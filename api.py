"""
REST API Service for AI/ML Resume Scorer & ATS Analytics Engine
Built with FastAPI, Pydantic, and Uvicorn.

Security & Privacy Enhancements:
- Strict CORS configuration (explicit trusted origin whitelist, no credentialed wildcard).
- Upload size limits (5 MB limit) with HTTP 413 rejection.
- Content verification via binary magic bytes (protects against spoofed extensions / executables).
- Zero-retention data privacy policy and optional PII redaction.
- Full defensible scoring explanations and rubric validation telemetry.
"""

import os
import io
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from analyzer.skills_taxonomy import ROLE_BENCHMARKS, CATEGORIZED_SKILLS
from analyzer.extractor import extract_text_from_file, extract_text_from_pdf_with_diagnostics
from analyzer.scoring_engine import score_resume
from analyzer.report_builder import generate_pdf_report, generate_markdown_report
from analyzer.sample_resumes import SAMPLE_RESUMES
from analyzer.security import (
    MAX_UPLOAD_SIZE_BYTES,
    validate_file_content,
    redact_pii,
    PRIVACY_POLICY
)
from analyzer.evaluation_benchmark import run_rubric_validation

app = FastAPI(
    title="⚡ AI/ML Resume Scorer & ATS Intelligence API",
    description=(
        "Production-ready REST API for multi-dimensional NLP resume scoring, "
        "dense semantic embedding benchmarking, ATS compliance diagnostics, and candidate privacy protection."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ---------------------------------------------------------------------------
# Strict CORS Configuration
# Restricts access to trusted origins only; forbids insecure wildcard with credentials
# ---------------------------------------------------------------------------
DEFAULT_TRUSTED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

env_origins = os.getenv("ALLOWED_ORIGINS", "")
if env_origins:
    configured_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    # Reject wildcard if credentials are enabled
    trusted_origins = [o for o in configured_origins if o != "*"]
    if not trusted_origins:
        trusted_origins = DEFAULT_TRUSTED_ORIGINS
else:
    trusted_origins = DEFAULT_TRUSTED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=trusted_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
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
    redact_pii_before_scoring: Optional[bool] = Field(
        default=False,
        description="Optionally scrub candidate contact PII before analysis"
    )

class RedactPiiRequest(BaseModel):
    text: str = Field(..., description="Resume text to scrub of personal information", min_length=10)

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
        "version": "2.1.0",
        "total_roles_benchmarked": len(ROLE_BENCHMARKS),
        "total_skill_categories": len(CATEGORIZED_SKILLS),
        "preloaded_samples_count": len(SAMPLE_RESUMES),
        "cors_trusted_origins_count": len(trusted_origins),
        "max_upload_size_mb": MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
    }


@app.get("/api/privacy-policy", summary="Data Retention & Privacy Policy", tags=["Privacy & Security"])
def privacy_policy() -> Dict[str, Any]:
    """
    Returns explicit Zero-Retention Data Policy guarantees for sensitive applicant data.
    """
    return {
        "policy": PRIVACY_POLICY,
        "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,
        "allowed_extensions": [".pdf", ".docx", ".txt"]
    }


@app.get("/api/rubric-validation", summary="Validate Scoring Rubric", tags=["Methodology"])
def rubric_validation() -> Dict[str, Any]:
    """
    Returns statistical validation metrics (Pearson r, Spearman rho, MAE)
    comparing the heuristic scoring rubric against human-reviewed evaluation benchmarks.
    """
    return run_rubric_validation()


@app.post("/api/redact-pii", summary="Redact Candidate PII", tags=["Privacy & Security"])
def redact_pii_endpoint(payload: RedactPiiRequest) -> Dict[str, Any]:
    """
    Anonymize candidate contact info (emails, phones, LinkedIn, GitHub) before evaluation.
    """
    return redact_pii(payload.text)


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
    Returns composite score, 6D dimensional breakdown, extracted skills, and defensible score explanations.
    """
    text_to_score = payload.resume_text
    if payload.redact_pii_before_scoring:
        redaction_res = redact_pii(text_to_score)
        text_to_score = redaction_res["redacted_text"]

    try:
        results = score_resume(
            resume_text=text_to_score,
            target_role_name=payload.target_role,
            custom_jd_text=payload.custom_jd_text
        )
        if payload.redact_pii_before_scoring:
            results["pii_redacted"] = True
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {str(e)}"
        )


@app.post("/api/upload-and-score", summary="Upload & Score Resume Document", tags=["Scoring"])
async def upload_and_score(
    file: UploadFile = File(..., description="Resume file (.pdf, .docx, .txt) - Max 5MB"),
    target_role: Optional[str] = Form(default=list(ROLE_BENCHMARKS.keys())[0]),
    custom_jd_text: Optional[str] = Form(default=None),
    redact_pii_flag: Optional[bool] = Form(default=False)
) -> Dict[str, Any]:
    """
    Upload resume file with strict size checks (5MB limit), content magic-byte verification,
    and safe ephemeral in-memory processing.
    """
    filename = file.filename or "resume"

    # Enforce streaming upload size limit to prevent memory exhaustion / DoS
    content = bytearray()
    chunk_size = 1024 * 64  # 64 KB chunks
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum upload size limit of {max_mb} MB."
            )

    content_bytes = bytes(content)

    # Content and magic-byte verification
    is_valid, validation_msg = validate_file_content(content_bytes, filename)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_msg
        )

    # Multi-column & scanned document extraction
    if filename.lower().endswith(".pdf"):
        diag = extract_text_from_pdf_with_diagnostics(content_bytes)
        if diag.get("error"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=diag["error"]
            )
        if diag.get("is_scanned"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Scanned or image-only PDF detected. This document lacks a machine-readable text layer. "
                    "Applicant Tracking Systems (ATS) cannot parse scanned images without OCR."
                )
            )
        extracted_text = diag.get("text", "")
        extraction_warnings = diag.get("warnings", [])
    else:
        file_obj = io.BytesIO(content_bytes)
        file_obj.name = filename
        try:
            extracted_text = extract_text_from_file(file_obj)
            extraction_warnings = []
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to extract text from document: {str(e)}"
            )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file contained no parseable text."
        )

    if redact_pii_flag:
        redaction = redact_pii(extracted_text)
        extracted_text = redaction["redacted_text"]

    results = score_resume(
        resume_text=extracted_text,
        target_role_name=target_role,
        custom_jd_text=custom_jd_text
    )
    results["source_filename"] = filename
    results["extraction_warnings"] = extraction_warnings
    if redact_pii_flag:
        results["pii_redacted"] = True

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
                "explanations": res.get("explanations", {}),
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
        "rubric_disclaimer": (
            "Disclaimer: Candidate rankings are based on a simulated heuristic rubric for evaluation benchmarking. "
            "Scores represent heuristic alignment, not commercial ATS hiring filters."
        ),
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
