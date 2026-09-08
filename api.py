"""
REST API Service for AI/ML Resume Scorer & ATS Analytics Engine
Built with FastAPI, Pydantic, and Uvicorn.
"""

import os
import io
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from analyzer.skills_taxonomy import ROLE_BENCHMARKS, CATEGORIZED_SKILLS
from analyzer.extractor import extract_text_from_file
from analyzer.scoring_engine import score_resume
from analyzer.report_builder import generate_pdf_report, generate_markdown_report
from analyzer.sample_resumes import SAMPLE_RESUMES

app = FastAPI(
    title="⚡ AI/ML Resume Scorer & ATS Intelligence API",
    description="Production-ready REST API for multi-dimensional NLP resume scoring, ATS compliance diagnostics, and AI/ML competency benchmarking.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web and microservice integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    resume_text: str = Field(..., description="Full text content of candidate resume", min_length=20)
    target_role: Optional[str] = Field(
        default=list(ROLE_BENCHMARKS.keys())[0],
        description="Target role benchmark identifier"
    )
    custom_jd_text: Optional[str] = Field(
        default=None,
        description="Optional custom Job Description text for tailored semantic matching"
    )

class BatchScoreItem(BaseModel):
    id: str = Field(..., description="Unique candidate or resume identifier")
    resume_text: str = Field(..., min_length=20)

class BatchScoreRequest(BaseModel):
    candidates: List[BatchScoreItem] = Field(..., description="List of candidates to score")
    target_role: Optional[str] = Field(default=list(ROLE_BENCHMARKS.keys())[0])
    custom_jd_text: Optional[str] = Field(default=None)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")

@app.get("/api/health", summary="Health Check", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Verify API health, loaded taxonomies, and system readiness."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "total_roles_benchmarked": len(ROLE_BENCHMARKS),
        "total_skill_categories": len(CATEGORIZED_SKILLS),
        "preloaded_samples_count": len(SAMPLE_RESUMES),
    }

@app.get("/api/roles", summary="List Available Roles", tags=["Taxonomy"])
def list_roles() -> Dict[str, Any]:
    """Retrieve list of all benchmark roles with their core and secondary skills."""
    roles_data = {}
    for role, data in ROLE_BENCHMARKS.items():
        roles_data[role] = {
            "core_skills_count": len(data.get("core_skills", [])),
            "secondary_skills_count": len(data.get("secondary_skills", [])),
            "core_skills": data.get("core_skills", []),
            "secondary_skills": data.get("secondary_skills", [])
        }
    return {"roles": roles_data}

@app.get("/api/samples", summary="List Sample Resumes", tags=["Taxonomy"])
def list_samples() -> Dict[str, Any]:
    """List preloaded realistic AI/ML sample resumes."""
    return {
        "samples": [
            {"id": key, "title": key, "text_preview": text[:200] + "..."}
            for key, text in SAMPLE_RESUMES.items()
        ]
    }

@app.post("/api/score", summary="Score Resume Text", tags=["Scoring"])
def score_text_endpoint(payload: ScoreRequest) -> Dict[str, Any]:
    """
    Score raw resume text against a target benchmark role or custom job description.
    Returns composite score, 5D dimensional breakdown, extracted skills, and recruiter diagnostics.
    """
    try:
        results = score_resume(
            resume_text=payload.resume_text,
            target_role_name=payload.target_role,
            custom_jd_text=payload.custom_jd_text
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {str(e)}"
        )

@app.post("/api/upload-and-score", summary="Upload & Score Resume Document", tags=["Scoring"])
async def upload_and_score(
    file: UploadFile = File(..., description="Resume file (.pdf, .docx, .txt)"),
    target_role: Optional[str] = Form(default=list(ROLE_BENCHMARKS.keys())[0]),
    custom_jd_text: Optional[str] = Form(default=None)
) -> Dict[str, Any]:
    """
    Upload resume file (PDF, DOCX, or TXT), automatically extract text, and perform full ATS evaluation.
    """
    filename = file.filename or "resume"
    extension = os.path.splitext(filename)[1].lower()
    
    if extension not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Only .pdf, .docx, and .txt are supported."
        )

    content = await file.read()
    file_obj = io.BytesIO(content)
    file_obj.name = filename

    try:
        extracted_text = extract_text_from_file(file_obj)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from file: {str(e)}"
        )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file contained no parseable text."
        )

    results = score_resume(
        resume_text=extracted_text,
        target_role_name=target_role,
        custom_jd_text=custom_jd_text
    )
    results["source_filename"] = filename
    return results

@app.post("/api/batch-score", summary="Batch Score Multiple Resumes", tags=["Scoring"])
def batch_score(payload: BatchScoreRequest) -> Dict[str, Any]:
    """
    Evaluate multiple resumes in batch, sorting by composite score to generate a ranked candidate leaderboard.
    """
    scored_candidates = []
    
    for item in payload.candidates:
        try:
            res = score_resume(
                resume_text=item.resume_text,
                target_role_name=payload.target_role,
                custom_jd_text=payload.custom_jd_text
            )
            scored_candidates.append({
                "id": item.id,
                "composite_score": res["composite_score"],
                "verdict": res["verdict"],
                "breakdown": res["scores_breakdown"],
                "matched_core_skills": res["skills_summary"]["matched_core"],
                "missing_core_skills": res["skills_summary"]["missing_core"],
                "impact_score": res["scores_breakdown"]["impact_score"],
                "full_result": res
            })
        except Exception as e:
            scored_candidates.append({
                "id": item.id,
                "error": str(e)
            })

    # Sort valid candidates by score descending
    valid_candidates = [c for c in scored_candidates if "composite_score" in c]
    valid_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Assign ranks
    for rank, c in enumerate(valid_candidates, 1):
        c["rank"] = rank

    avg_score = round(sum(c["composite_score"] for c in valid_candidates) / len(valid_candidates), 1) if valid_candidates else 0

    return {
        "target_role": payload.target_role,
        "total_processed": len(payload.candidates),
        "average_score": avg_score,
        "leaderboard": valid_candidates,
        "errors": [c for c in scored_candidates if "error" in c]
    }

@app.post("/api/report/pdf", summary="Generate PDF Report", tags=["Reports"])
def report_pdf_endpoint(payload: ScoreRequest):
    """Generate a downloadable PDF audit report from resume score evaluation."""
    results = score_resume(
        resume_text=payload.resume_text,
        target_role_name=payload.target_role,
        custom_jd_text=payload.custom_jd_text
    )
    pdf_bytes = generate_pdf_report(results)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Resume_Scorer_Report.pdf"}
    )

@app.post("/api/report/markdown", summary="Generate Markdown Report", tags=["Reports"])
def report_markdown_endpoint(payload: ScoreRequest) -> Dict[str, str]:
    """Generate markdown audit text from resume score evaluation."""
    results = score_resume(
        resume_text=payload.resume_text,
        target_role_name=payload.target_role,
        custom_jd_text=payload.custom_jd_text
    )
    md_text = generate_markdown_report(results)
    return {"markdown": md_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
