"""
Multi-Faceted Scoring Engine for AI/ML Resume Scorer
Combines Hard Skills, TF-IDF Semantic Similarity, Quantifiable Impact, MLOps Readiness, and ATS Health.
"""

import re
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analyzer.skills_taxonomy import ROLE_BENCHMARKS, CATEGORIZED_SKILLS, SKILL_LEARNING_RESOURCES
from analyzer.matcher import extract_skills_from_text, extract_skills_from_job_description
from analyzer.section_parser import parse_resume_sections, evaluate_section_completeness
from analyzer.impact_analyzer import analyze_impact_and_verbs
from analyzer.extractor import extract_contact_info

def compute_tfidf_similarity(resume_text: str, target_text: str) -> float:
    """
    Computes cosine similarity between resume text and target profile/JD using TF-IDF n-grams.
    Returns float between 0.0 and 1.0.
    """
    if not resume_text or not target_text:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, target_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        # Simple Jaccard fallback if TFIDF fails
        r_words = set(re.findall(r'\w+', resume_text.lower()))
        t_words = set(re.findall(r'\w+', target_text.lower()))
        if not t_words:
            return 0.0
        return len(r_words.intersection(t_words)) / len(t_words)


def calculate_ats_readability(text: str) -> Dict[str, Any]:
    """Calculates readability and formatting health metrics."""
    word_count = len(re.findall(r'\b\w+\b', text))
    char_count = len(text)
    
    # Try textstat if installed
    flesch_score = 60.0
    try:
        import textstat
        flesch_score = float(textstat.flesch_reading_ease(text))
    except Exception:
        # Native approximation
        sentences = max(1, len(re.split(r'[.!?]+', text)))
        words = max(1, word_count)
        avg_sentence_len = words / sentences
        flesch_score = max(0.0, min(100.0, 206.835 - 1.015 * avg_sentence_len - 15.0))

    # Readability health (ideal 45-75 for technical resumes)
    if 45 <= flesch_score <= 80:
        readability_status = "Optimal (Clear & Professional)"
        readability_rating = 95
    elif flesch_score > 80:
        readability_status = "Too Simple (Add more technical precision)"
        readability_rating = 75
    else:
        readability_status = "Dense / Complex (Improve sentence conciseness)"
        readability_rating = 70

    # Length health (ideal 350 - 1100 words)
    if 350 <= word_count <= 1100:
        length_status = "Ideal (1 - 2 Pages standard)"
        length_score = 100
    elif word_count < 350:
        length_status = "Too Brief (Expand on projects and metrics)"
        length_score = 60
    else:
        length_status = "Too Long (Consider condensing into 2 pages)"
        length_score = 75

    return {
        "word_count": word_count,
        "char_count": char_count,
        "flesch_score": round(flesch_score, 1),
        "readability_status": readability_status,
        "readability_rating": readability_rating,
        "length_status": length_status,
        "length_score": length_score
    }


def score_resume(
    resume_text: str,
    target_role_name: Optional[str] = None,
    custom_jd_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main analytics scoring function.
    Evaluates resume against a role benchmark or a custom job description.
    """
    if not resume_text or not resume_text.strip():
        return {"error": "Empty resume text provided"}

    # 1. Extract Candidate Skills
    extracted_skills_data = extract_skills_from_text(resume_text)
    candidate_skills = extracted_skills_data["all_matched"]
    candidate_by_category = extracted_skills_data["by_category"]

    # 2. Extract Sections, Contact Info & Impact
    sections = parse_resume_sections(resume_text)
    section_eval = evaluate_section_completeness(sections)
    contact_info = extract_contact_info(resume_text)
    impact_data = analyze_impact_and_verbs(resume_text)
    readability_data = calculate_ats_readability(resume_text)

    # 3. Determine Target Benchmark (Role vs Custom JD)
    if custom_jd_text and custom_jd_text.strip():
        jd_data = extract_skills_from_job_description(custom_jd_text)
        core_required = jd_data["core_skills"]
        secondary_required = jd_data["secondary_skills"]
        target_description = custom_jd_text
        role_label = "Custom Job Description"
        category_weights = {
            "Generative AI & LLMs": 0.25,
            "Deep Learning & AI Architectures": 0.25,
            "MLOps, Deployment & Infrastructure": 0.20,
            "Classical Machine Learning & Modeling": 0.15,
            "Software Engineering & Tools": 0.15
        }
    else:
        if not target_role_name or target_role_name not in ROLE_BENCHMARKS:
            target_role_name = list(ROLE_BENCHMARKS.keys())[0]
        
        benchmark = ROLE_BENCHMARKS[target_role_name]
        core_required = benchmark["core_skills"]
        secondary_required = benchmark["secondary_skills"]
        role_label = target_role_name
        target_description = benchmark["description"] + " " + " ".join(core_required + secondary_required)
        category_weights = benchmark.get("category_weights", {})

    # 4. Compute Skill Matches & Gaps
    matched_core = [s for s in core_required if s in candidate_skills]
    missing_core = [s for s in core_required if s not in candidate_skills]
    
    matched_secondary = [s for s in secondary_required if s in candidate_skills]
    missing_secondary = [s for s in secondary_required if s not in candidate_skills]

    # Additional skills candidate has beyond the target requirements
    target_all = set(core_required + secondary_required)
    bonus_skills = [s for s in candidate_skills if s not in target_all]

    # Hard Skill Match Calculation
    core_match_rate = len(matched_core) / max(1, len(core_required))
    sec_match_rate = len(matched_secondary) / max(1, len(secondary_required))
    hard_skill_score = round((core_match_rate * 75 + sec_match_rate * 25), 1)

    # 5. Semantic Similarity (TF-IDF Cosine Similarity)
    raw_sim = compute_tfidf_similarity(resume_text, target_description)
    # Calibrate TF-IDF score (a raw cosine similarity of 0.40+ in technical docs is very strong)
    semantic_score = round(min(100.0, (raw_sim / 0.45) * 100.0), 1)

    # 6. MLOps & Production Readiness Subscore
    mlops_candidate_skills = candidate_by_category.get("MLOps, Deployment & Infrastructure", [])
    mlops_score = min(100, int((len(mlops_candidate_skills) / 4) * 100))

    # 7. ATS Formatting & Completeness Score
    contact_score = 0
    if contact_info["email"]: contact_score += 35
    if contact_info["phone"]: contact_score += 25
    if contact_info["linkedin"] or contact_info["github"]: contact_score += 40

    ats_health_score = round(
        section_eval["score"] * 0.45 +
        contact_score * 0.30 +
        readability_data["readability_rating"] * 0.15 +
        readability_data["length_score"] * 0.10,
        1
    )

    # 8. Overall Composite ATS Score
    # Weights: Hard Skill (35%), Semantic (20%), Impact (15%), MLOps (15%), ATS Health (15%)
    composite_score = round(
        (hard_skill_score * 0.35) +
        (semantic_score * 0.20) +
        (impact_data["overall_impact_score"] * 0.15) +
        (mlops_score * 0.15) +
        (ats_health_score * 0.15)
    )
    composite_score = max(0, min(100, int(composite_score)))

    # Tier verdict
    if composite_score >= 85:
        verdict = "🏆 Elite Match (Interview Ready)"
        verdict_color = "#10B981"  # Emerald Green
        verdict_desc = "Your resume strongly aligns with top-tier AI/ML engineering standards with rich skills, clear metrics, and robust ATS formatting."
    elif composite_score >= 70:
        verdict = "🌟 Strong Match (High ATS Pass Rate)"
        verdict_color = "#3B82F6"  # Blue
        verdict_desc = "Solid candidate profile. Optimizing missing core skills and boosting quantifiable bullet impact will position you in the top 5%."
    elif composite_score >= 50:
        verdict = "⚡ Moderate Fit (Needs Optimization)"
        verdict_color = "#F59E0B"  # Amber
        verdict_desc = "Good foundation, but several key AI/ML competencies or quantifiable metrics are missing for this specific role."
    else:
        verdict = "⚠️ High Gap (Significant Optimization Needed)"
        verdict_color = "#EF4444"  # Red
        verdict_desc = "Low match against target role requirements. Review the skill gap checklist and follow the recommended learning resources."

    # 9. Radar Chart Multi-Dimensional Data
    radar_dimensions = {
        "Core AI/ML Skills": hard_skill_score,
        "Semantic Fit": semantic_score,
        "Quantifiable Impact": impact_data["overall_impact_score"],
        "MLOps & Deployment": mlops_score,
        "ATS Health & Structure": ats_health_score
    }

    # 10. Curated Learning Recommendations for Missing Skills
    learning_recommendations = []
    for skill in missing_core[:6]:
        resource = SKILL_LEARNING_RESOURCES.get(skill, {
            "title": f"Official {skill.title()} Documentation & Mastery",
            "link": f"https://www.google.com/search?q={skill.replace(' ', '+')}+tutorial+documentation"
        })
        learning_recommendations.append({
            "skill": skill,
            "title": resource["title"],
            "link": resource["link"],
            "priority": "High (Core Requirement)"
        })

    for skill in missing_secondary[:4]:
        resource = SKILL_LEARNING_RESOURCES.get(skill, {
            "title": f"Master {skill.title()} for Production AI",
            "link": f"https://www.google.com/search?q={skill.replace(' ', '+')}+guide"
        })
        learning_recommendations.append({
            "skill": skill,
            "title": resource["title"],
            "link": resource["link"],
            "priority": "Medium (Bonus Advantage)"
        })

    return {
        "target_role": role_label,
        "composite_score": composite_score,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "verdict_desc": verdict_desc,
        "scores_breakdown": {
            "hard_skill_score": hard_skill_score,
            "semantic_score": semantic_score,
            "impact_score": impact_data["overall_impact_score"],
            "mlops_score": mlops_score,
            "ats_health_score": ats_health_score
        },
        "radar_dimensions": radar_dimensions,
        "skills_summary": {
            "matched_core": matched_core,
            "missing_core": missing_core,
            "matched_secondary": matched_secondary,
            "missing_secondary": missing_secondary,
            "bonus_skills": bonus_skills[:15],
            "total_candidate_skills": len(candidate_skills),
            "by_category": candidate_by_category
        },
        "impact_analysis": impact_data,
        "sections": sections,
        "section_evaluation": section_eval,
        "contact_info": contact_info,
        "readability": readability_data,
        "learning_recommendations": learning_recommendations
    }
