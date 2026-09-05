"""
Resume Impact, Action Verbs, and Bullet Point Analyzer
Evaluates quantifiable metrics, action verb strength, and X-Y-Z formula compliance.
"""

import re
from typing import List, Dict, Any

STRONG_ACTION_VERBS = {
    # Engineering & Building
    "architected", "engineered", "developed", "built", "implemented", "designed",
    "constructed", "programmed", "authored", "spearheaded", "integrated",
    # AI/ML Specific
    "fine-tuned", "trained", "optimized", "benchmarked", "deployed", "scaled",
    "quantized", "pruned", "distilled", "evaluated", "containerized", "automated",
    # Data & Analytics
    "extracted", "transformed", "modeled", "analyzed", "forecasted", "visualized",
    "mined", "aggregated", "queried", "curated", "cleaned",
    # Leadership & Impact
    "orchestrated", "led", "directed", "accelerated", "maximized", "reduced",
    "slashed", "boosted", "delivered", "expanded", "generated", "pioneered"
}

WEAK_PASSIVE_PHRASES = [
    "responsible for", "duties included", "worked on", "helped with",
    "assisted in", "tasked with", "handled", "participated in", "involved in",
    "contributed to", "tried to", "supported"
]

def analyze_impact_and_verbs(text: str) -> Dict[str, Any]:
    """
    Analyzes resume text for action verbs, quantifiable metrics, and impact density.
    """
    if not text:
        return {
            "action_verb_score": 0,
            "metric_density_score": 0,
            "overall_impact_score": 0,
            "strong_verbs_found": [],
            "weak_phrases_found": [],
            "metrics_found": [],
            "bullet_analysis": []
        }

    lower_text = text.lower()
    
    # 1. Action Verbs Detection
    strong_verbs_found = []
    for verb in STRONG_ACTION_VERBS:
        # Match whole word
        if re.search(r'\b' + re.escape(verb) + r'\b', lower_text):
            strong_verbs_found.append(verb)
            
    # 2. Weak Passive Phrases Detection
    weak_phrases_found = []
    for phrase in WEAK_PASSIVE_PHRASES:
        if phrase in lower_text:
            weak_phrases_found.append(phrase)

    # 3. Quantifiable Metrics Detection
    # Regex patterns for: percentages, multipliers, latency, monetary amounts, large counts
    metric_patterns = [
        r'\b\d+(?:\.\d+)?%',                                   # 95%, 4.5%
        r'\b\d+(?:\.\d+)?x\b',                                  # 10x, 2.5x
        r'\$\s?\d+(?:,\d+)*(?:\.\d+)?[kKmMbB]?',               # $50k, $1.2M
        r'\b\d+(?:,\d+)*(?:\.\d+)?\s?(?:ms|sec|seconds|fps|tokens/sec|req/s|rps)\b',  # 40ms, 120fps
        r'\b\d+(?:,\d+)*\+?\s?(?:k|m|b|million|billion|users|records|samples|parameters|params)\b', # 10M users, 7B params
        r'\b(?:reduced|increased|improved|boosted|saved|accelerated)\s+by\s+\d+(?:\.\d+)?%?'
    ]
    
    metrics_found = []
    for pat in metric_patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        for m in matches:
            clean_m = m.strip()
            if clean_m and clean_m not in metrics_found:
                metrics_found.append(clean_m)

    # 4. Bullet Point Analysis
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 25]
    bullet_lines = []
    for l in lines:
        if l.startswith(("-", "•", "*", "–", "—", "1.", "2.", "3.", "4.", "5.")) or len(l.split()) >= 6:
            bullet_lines.append(l)

    bullet_analysis = []
    strong_bullets_count = 0
    
    for bullet in bullet_lines[:20]:  # Analyze up to top 20 bullets
        b_lower = bullet.lower()
        has_verb = any(re.search(r'\b' + re.escape(v) + r'\b', b_lower) for v in STRONG_ACTION_VERBS)
        has_weak = any(wp in b_lower for wp in WEAK_PASSIVE_PHRASES)
        has_metric = any(re.search(pat, bullet, flags=re.IGNORECASE) for pat in metric_patterns)
        
        # Categorize
        if has_verb and has_metric and not has_weak:
            rating = "🔥 Strong (X-Y-Z Aligned)"
            suggestion = "Great impactful statement with strong action and quantifiable outcome!"
            strong_bullets_count += 1
        elif has_verb or has_metric:
            rating = "⚡ Moderate"
            if not has_metric:
                suggestion = "Consider adding specific numbers (e.g. latency, %, volume) to show tangible outcome."
            else:
                suggestion = "Start with a stronger action verb (e.g. 'Architected', 'Engineered', 'Optimized')."
        else:
            rating = "⚠️ Needs Improvement"
            suggestion = "Rewrite using Google's X-Y-Z formula: 'Accomplished [X] measured by [Y], by doing [Z]'."

        bullet_analysis.append({
            "bullet": bullet,
            "rating": rating,
            "has_verb": has_verb,
            "has_metric": has_metric,
            "has_weak": has_weak,
            "suggestion": suggestion
        })

    # Scores (0 - 100)
    verb_score = min(100, int((len(strong_verbs_found) / 8) * 100))
    metric_score = min(100, int((len(metrics_found) / 6) * 100))
    
    # Impact composite
    penalty = len(weak_phrases_found) * 5
    overall_impact = max(0, min(100, int((verb_score * 0.5 + metric_score * 0.5) - penalty)))

    return {
        "action_verb_score": verb_score,
        "metric_density_score": metric_score,
        "overall_impact_score": overall_impact,
        "strong_verbs_found": sorted(strong_verbs_found),
        "weak_phrases_found": sorted(weak_phrases_found),
        "metrics_found": metrics_found[:12],
        "bullet_analysis": bullet_analysis,
        "total_bullets_analyzed": len(bullet_lines),
        "strong_bullets_count": strong_bullets_count
    }
