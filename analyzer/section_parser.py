"""
Resume Section Segmentation Module
Identifies standard resume sections and chunks content appropriately.
"""

import re
from typing import Dict, List, Any

SECTION_HEADERS = {
    "Summary": [
        "summary", "professional summary", "executive summary", "about me",
        "profile", "personal profile", "career objective", "objective", "overview"
    ],
    "Experience": [
        "experience", "work experience", "professional experience", "employment history",
        "work history", "internships", "relevant experience", "industry experience", "career history"
    ],
    "Education": [
        "education", "academic background", "academic history", "degrees",
        "qualifications", "educational background", "university"
    ],
    "Skills": [
        "skills", "technical skills", "core competencies", "technologies",
        "tools & technologies", "technical proficiencies", "areas of expertise",
        "key skills", "skillset", "programming languages"
    ],
    "Projects": [
        "projects", "personal projects", "academic projects", "key projects",
        "open source", "selected projects", "technical projects", "portfolio",
        "projects & publications", "projects and publications", "key projects & achievements"
    ],
    "Certifications": [
        "certifications", "certificates", "licenses", "courses",
        "professional development", "training"
    ],
    "Publications": [
        "publications", "research papers", "patents", "conference proceedings",
        "journal articles", "research"
    ],
    "Awards": [
        "awards", "honors", "achievements", "accomplishments", "recognition"
    ]
}

def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Parses resume text into a dictionary of section names to content.
    Returns detected sections and an overall section presence scorecard.
    """
    if not text:
        return {}

    lines = text.split("\n")
    sections: Dict[str, List[str]] = {sec: [] for sec in SECTION_HEADERS}
    sections["Header"] = []
    
    current_section = "Header"
    
    # Pre-compile patterns
    header_patterns = {}
    for sec, aliases in SECTION_HEADERS.items():
        pattern = r'^(?:[\d\.\-\*#\s]*)(?:' + '|'.join(re.escape(a) for a in aliases) + r')(?:[:\-\s]*)$'
        header_patterns[sec] = re.compile(pattern, re.IGNORECASE)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        matched_section = None
        # Check if line looks like a header (short length, matches pattern)
        if len(line) < 45:
            for sec, pat in header_patterns.items():
                if pat.match(line.lower()):
                    matched_section = sec
                    break
        
        if matched_section:
            current_section = matched_section
        else:
            sections[current_section].append(raw_line)

    result = {sec: "\n".join(lines_list).strip() for sec, lines_list in sections.items() if lines_list}
    return result


def evaluate_section_completeness(sections: Dict[str, str]) -> Dict[str, Any]:
    """
    Evaluates presence of critical sections for ATS compliance.
    """
    critical = ["Experience", "Education", "Skills", "Projects"]
    recommended = ["Summary", "Certifications"]

    found_critical = [s for s in critical if s in sections and len(sections[s]) > 20]
    found_recommended = [s for s in recommended if s in sections and len(sections[s]) > 20]

    score = (len(found_critical) / len(critical)) * 70 + (len(found_recommended) / len(recommended)) * 30

    return {
        "score": round(score, 1),
        "found_sections": list(sections.keys()),
        "missing_critical": [s for s in critical if s not in found_critical],
        "missing_recommended": [s for s in recommended if s not in found_recommended]
    }
