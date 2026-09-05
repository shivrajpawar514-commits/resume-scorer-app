"""
Skill Extraction and NLP Phrase Matching Module
Uses SpaCy PhraseMatcher and boundary-aware regex to accurately extract multi-word skills.
"""

import re
from typing import List, Dict, Set, Tuple, Any
import spacy
from spacy.matcher import PhraseMatcher

from analyzer.skills_taxonomy import CATEGORIZED_SKILLS, ALL_TAXONOMY_SKILLS

_nlp_instance = None
_phrase_matcher = None

def get_spacy_nlp():
    """Lazy load and cache SpaCy model."""
    global _nlp_instance
    if _nlp_instance is None:
        try:
            _nlp_instance = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            # Fallback blank English model if en_core_web_sm is not available
            _nlp_instance = spacy.blank("en")
    return _nlp_instance


def get_skill_matcher():
    """Initializes and caches the SpaCy PhraseMatcher for all taxonomy skills."""
    global _phrase_matcher
    if _phrase_matcher is None:
        nlp = get_spacy_nlp()
        _phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        
        # Batch add patterns
        patterns = [nlp.make_doc(skill) for skill in ALL_TAXONOMY_SKILLS]
        _phrase_matcher.add("ALL_SKILLS", patterns)
        
    return _phrase_matcher


def extract_skills_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts all recognized skills from raw text using SpaCy PhraseMatcher
    and regex boundaries for acronyms and special characters (e.g. C++, CI/CD, A/B testing).
    """
    if not text:
        return {
            "all_matched": [],
            "by_category": {cat: [] for cat in CATEGORIZED_SKILLS},
            "skill_counts": {},
            "total_count": 0
        }

    nlp = get_spacy_nlp()
    matcher = get_skill_matcher()
    
    doc = nlp(text.lower())
    matches = matcher(doc)
    
    matched_skills_set: Set[str] = set()
    skill_counts: Dict[str, int] = {}
    
    for match_id, start, end in matches:
        span_text = doc[start:end].text.strip()
        matched_skills_set.add(span_text)
        skill_counts[span_text] = skill_counts.get(span_text, 0) + 1

    # Special handling for short acronyms / symbols that SpaCy tokenizer might split or miss
    special_symbols = ["c++", "r", "ci/cd", "a/b testing", "sql", "aws", "gcp", "nlp", "llm", "rag", "eda", "oop"]
    lower_text = " " + text.lower() + " "
    
    for sym in special_symbols:
        # Regex boundary check
        pattern = r'(?:^|[\s\(\),;/\[\]])' + re.escape(sym) + r'(?:[\s\(\),;/\[\]\.]|$)'
        count = len(re.findall(pattern, lower_text))
        if count > 0:
            matched_skills_set.add(sym)
            skill_counts[sym] = max(skill_counts.get(sym, 0), count)

    # Categorize matches
    by_category: Dict[str, List[str]] = {cat: [] for cat in CATEGORIZED_SKILLS}
    for skill in matched_skills_set:
        for cat, skills_in_cat in CATEGORIZED_SKILLS.items():
            if skill in skills_in_cat and skill not in by_category[cat]:
                by_category[cat].append(skill)

    return {
        "all_matched": sorted(list(matched_skills_set)),
        "by_category": by_category,
        "skill_counts": skill_counts,
        "total_count": len(matched_skills_set)
    }


def extract_skills_from_job_description(jd_text: str) -> Dict[str, Any]:
    """
    Analyzes custom Job Description text to identify key required skills,
    frequency, and taxonomy distribution.
    """
    extracted = extract_skills_from_text(jd_text)
    
    # Sort JD skills by frequency of occurrence in the JD text
    sorted_skills = sorted(
        extracted["all_matched"],
        key=lambda s: extracted["skill_counts"].get(s, 1),
        reverse=True
    )
    
    # Heuristic: top 60% frequency skills as core, rest as secondary
    split_idx = max(3, int(len(sorted_skills) * 0.6))
    core_jd_skills = sorted_skills[:split_idx]
    secondary_jd_skills = sorted_skills[split_idx:]
    
    return {
        "core_skills": core_jd_skills,
        "secondary_skills": secondary_jd_skills,
        "all_skills": sorted_skills,
        "by_category": extracted["by_category"],
        "skill_counts": extracted["skill_counts"]
    }
