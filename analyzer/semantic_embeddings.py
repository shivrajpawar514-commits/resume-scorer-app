"""
Semantic Embeddings and Lexical Similarity Engine
Provides:
1. Genuine dense vector embedding comparisons using transformer-based models (FastEmbed / BAAI/bge-small-en-v1.5).
2. Lexical n-gram similarity via TF-IDF with accurate distinction between lexical vocabulary overlap and true semantic understanding.
"""

import re
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_dense_model_instance = None
_model_load_attempted = False

def get_dense_embedding_model():
    """
    Lazy loads and caches the dense embedding model.
    Uses FastEmbed's BAAI/bge-small-en-v1.5 (compact, high-performance ONNX transformer model).
    """
    global _dense_model_instance, _model_load_attempted
    if _dense_model_instance is None and not _model_load_attempted:
        _model_load_attempted = True
        try:
            from fastembed import TextEmbedding
            _dense_model_instance = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception:
            _dense_model_instance = None
    return _dense_model_instance


def compute_dense_embedding_similarity(text_a: str, text_b: str) -> Dict[str, Any]:
    """
    Computes genuine semantic similarity between two texts using dense transformer embeddings.
    Captures deep contextual meaning, synonyms (e.g. 'Kubernetes' and 'k8s', 'fine-tuning' and 'model adaptation'),
    and conceptual alignment that lexical keyword matching misses.
    
    Returns:
        Dict containing raw similarity [0.0, 1.0], calibrated score [0, 100], model used, and methodology description.
    """
    if not text_a or not text_b or not text_a.strip() or not text_b.strip():
        return {
            "raw_similarity": 0.0,
            "calibrated_score": 0.0,
            "engine": "none",
            "is_fallback": False,
            "explanation": "Empty text provided for semantic comparison."
        }

    # Clean text chunks for embedding
    clean_a = text_a[:4000].strip()
    clean_b = text_b[:4000].strip()

    model = get_dense_embedding_model()
    if model is not None:
        try:
            embeddings = list(model.embed([clean_a, clean_b]))
            vec_a = np.array(embeddings[0], dtype=np.float32)
            vec_b = np.array(embeddings[1], dtype=np.float32)

            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 0 and norm_b > 0:
                raw_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                raw_sim = float(np.clip(raw_sim, 0.0, 1.0))
                
                # Calibrate: BGE cosine similarity for domain-matching technical documents
                # typically spans 0.45 (baseline domain overlap) to 0.85+ (near perfect semantic match).
                # Rescale smoothly to 0-100 range:
                calibrated = max(0.0, min(100.0, ((raw_sim - 0.40) / (0.85 - 0.40)) * 100.0))
                
                return {
                    "raw_similarity": round(raw_sim, 4),
                    "calibrated_score": round(calibrated, 1),
                    "engine": "fastembed (BAAI/bge-small-en-v1.5)",
                    "is_fallback": False,
                    "explanation": (
                        f"Computed dense transformer embedding cosine similarity ({raw_sim:.3f}). "
                        f"Measures true semantic understanding, contextual nuance, and synonym alignment "
                        f"between candidate experience and role expectations."
                    )
                }
        except Exception as e:
            pass

    # Deterministic fallback: Sub-word character n-gram cosine + token Jaccard hybrid
    # Runs when dense transformer weights cannot be loaded in restricted environments.
    raw_sim = _compute_subword_hybrid_similarity(clean_a, clean_b)
    calibrated = round(min(100.0, (raw_sim / 0.50) * 100.0), 1)
    return {
        "raw_similarity": round(raw_sim, 4),
        "calibrated_score": calibrated,
        "engine": "subword-hybrid-fallback",
        "is_fallback": True,
        "explanation": (
            "Computed sub-word token similarity fallback. "
            "Evaluates morphological and keyword alignment when dense embedding weights are offline."
        )
    }


def compute_lexical_tfidf_similarity(text_a: str, text_b: str) -> Dict[str, Any]:
    """
    Computes lexical similarity using TF-IDF n-grams (1-2 words).
    IMPORTANT: This measures literal vocabulary overlap and exact keyword frequency,
    NOT semantic understanding or contextual meaning.
    
    Returns:
        Dict with raw cosine similarity [0.0, 1.0], calibrated score [0, 100], and explicit lexical disclaimer.
    """
    if not text_a or not text_b or not text_a.strip() or not text_b.strip():
        return {
            "raw_similarity": 0.0,
            "calibrated_score": 0.0,
            "engine": "tfidf",
            "is_lexical_only": True,
            "explanation": "Empty text provided for lexical comparison."
        }

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        raw_sim = float(np.clip(sim, 0.0, 1.0))
        # In technical resume corpus, a raw TF-IDF cosine of 0.40+ indicates very dense vocabulary overlap
        calibrated = round(min(100.0, (raw_sim / 0.45) * 100.0), 1)
        return {
            "raw_similarity": round(raw_sim, 4),
            "calibrated_score": calibrated,
            "engine": "tfidf (unigram + bigram)",
            "is_lexical_only": True,
            "explanation": (
                f"TF-IDF lexical cosine similarity: {raw_sim:.3f}. "
                "Notice: TF-IDF measures literal surface keyword frequency and vocabulary overlap (lexical similarity). "
                "It does not evaluate semantic meaning, conceptual mastery, or synonymous phrasing."
            )
        }
    except Exception:
        # Jaccard lexical fallback
        r_words = set(re.findall(r'\b\w+\b', text_a.lower()))
        t_words = set(re.findall(r'\b\w+\b', text_b.lower()))
        if not t_words:
            return {
                "raw_similarity": 0.0,
                "calibrated_score": 0.0,
                "engine": "jaccard-fallback",
                "is_lexical_only": True,
                "explanation": "No common tokens found."
            }
        jaccard = len(r_words.intersection(t_words)) / len(t_words)
        return {
            "raw_similarity": round(jaccard, 4),
            "calibrated_score": round(min(100.0, jaccard * 150), 1),
            "engine": "jaccard-token-fallback",
            "is_lexical_only": True,
            "explanation": "Lexical token overlap fallback."
        }


def _compute_subword_hybrid_similarity(text_a: str, text_b: str) -> float:
    """Helper for deterministic sub-word character n-gram cosine similarity."""
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000)
        matrix = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(np.clip(sim, 0.0, 1.0))
    except Exception:
        words_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w+\b', text_b.lower()))
        if not words_b:
            return 0.0
        return float(len(words_a.intersection(words_b)) / len(words_b))
