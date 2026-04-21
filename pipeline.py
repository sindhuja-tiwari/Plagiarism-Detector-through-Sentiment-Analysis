"""
backend/pipeline.py
-------------------
Core NLP pipeline — importable by the Flask API.

Stages
  1. BM25Retriever      — fast lexical candidate retrieval (rank_bm25)
  2. SBERTReranker      — semantic bi-encoder re-ranking (sentence-transformers)
  3. CrossEncoderScorer — fine-tuned cross-encoder for final similarity score
  4. SHAPExplainer      — token-level attribution via SHAP
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import shap
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # BM25
    bm25_top_k: int = 100

    # SBERT bi-encoder
    sbert_model: str = "all-MiniLM-L6-v2"

    # Cross-encoder (verified public, no HF token required)
    #   Options (fastest → most accurate):
    #     "cross-encoder/stsb-distilroberta-base"  ~300 MB
    #     "cross-encoder/stsb-roberta-base"        ~500 MB  ← default
    #     "cross-encoder/stsb-roberta-large"       ~1.4 GB
    cross_encoder_model: str = "cross-encoder/stsb-roberta-base"
    cross_encoder_max_len: int = 128

    # SHAP
    shap_max_evals: int = 150

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall to of in for on with at by "
    "from and or but not this that it its as into about than more also "
    "such their they which who how what when where".split()
)


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used by BM25."""
    return re.findall(r"\w+", text.lower())


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2]


# ── Stage 1: BM25 ─────────────────────────────────────────────────────────────

class BM25Retriever:
    """Thin wrapper around rank_bm25.BM25Okapi."""

    def __init__(self, corpus: List[str]) -> None:
        self.corpus = corpus
        self._bm25 = BM25Okapi([_tokenize(s) for s in corpus])

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus[i], float(scores[i])) for i in idx]

    def pairwise_score(self, text_a: str, text_b: str) -> float:
        """
        Normalised BM25 overlap: fraction of content tokens in B
        that also appear (as content tokens) in A.
        """
        tokens_a = set(_content_tokens(text_a))
        tokens_b = _content_tokens(text_b)
        if not tokens_b:
            return 0.0
        matches = sum(1 for t in tokens_b if t in tokens_a)
        return round(min(matches / len(tokens_b), 1.0), 4)


# ── Stage 2: SBERT ───────────────────────────────────────────────────────────

class SBERTReranker:
    """Bi-encoder cosine similarity via sentence-transformers."""

    def __init__(self, model_name: str, device: str) -> None:
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def similarity(self, text_a: str, text_b: str) -> float:
        embs = self.encode([text_a, text_b])
        return round(float(embs[0] @ embs[1]), 4)


# ── Stage 3: Cross-encoder ───────────────────────────────────────────────────

_FALLBACK_MODELS = [
    "cross-encoder/stsb-roberta-base",
    "cross-encoder/stsb-roberta-large",
    "cross-encoder/stsb-distilroberta-base",
]


class CrossEncoderScorer:
    """
    Cross-encoder that jointly encodes (A, B) and outputs a calibrated
    similarity score in [0, 1] via sigmoid.
    """

    def __init__(self, model_name: str, device: str, max_len: int = 128) -> None:
        self.device = device
        self.max_len = max_len
        self._model_name: str = ""

        candidates = list(dict.fromkeys([model_name] + _FALLBACK_MODELS))
        last_err: Exception = RuntimeError("No models available.")

        for candidate in candidates:
            try:
                self._tok = AutoTokenizer.from_pretrained(candidate)
                self._model = AutoModelForSequenceClassification.from_pretrained(candidate)
                self._model.to(device).eval()
                self._pipe = hf_pipeline(
                    "text-classification",
                    model=self._model,
                    tokenizer=self._tok,
                    device=0 if device == "cuda" else -1,
                    function_to_apply="sigmoid",
                    return_all_scores=False,
                )
                self._model_name = candidate
                break
            except Exception as exc:
                last_err = exc
        else:
            raise RuntimeError(f"All cross-encoders failed. Last: {last_err}")

    @torch.no_grad()
    def score(self, text_a: str, text_b: str) -> float:
        enc = self._tok(
            text_a, text_b,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        logit = self._model(**enc).logits.squeeze()
        return round(torch.sigmoid(logit).item(), 4)


# ── Stage 4: SHAP ─────────────────────────────────────────────────────────────

class SHAPExplainer:
    """Token-level SHAP attributions using the cross-encoder pipeline."""

    def __init__(self, scorer: CrossEncoderScorer, max_evals: int = 150) -> None:
        self._scorer = scorer
        self._max_evals = max_evals
        self._explainer = shap.Explainer(
            scorer._pipe,
            masker=shap.maskers.Text(scorer._tok),
            output_names=["similarity"],
        )

    def explain(self, text_a: str, text_b: str) -> List[Dict]:
        """
        Returns a list of dicts sorted by |shap_value| descending:
          { "token": str, "shap_value": float }
        """
        combined = f"{text_a} [SEP] {text_b}"
        sv = self._explainer([combined], max_evals=self._max_evals)
        tokens = list(sv.data[0])
        values = sv.values[0]
        if values.ndim > 1:
            values = values[:, 0]

        pairs = sorted(
            zip(tokens, values.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return [
            {"token": tok, "shap_value": round(val, 5)}
            for tok, val in pairs[:10]
            if tok.strip() and tok not in ("[SEP]", "[CLS]", "<s>", "</s>")
        ]


# ── Assembled pipeline ────────────────────────────────────────────────────────

class PlagiarismPipeline:
    """
    Facade that wires all three stages together.
    Call .analyse(text_a, text_b) to get a full result dict.
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.bm25: Optional[BM25Retriever] = None  # built lazily per-request
        self.sbert = SBERTReranker(self.cfg.sbert_model, self.cfg.device)
        self.cross = CrossEncoderScorer(
            self.cfg.cross_encoder_model,
            self.cfg.device,
            self.cfg.cross_encoder_max_len,
        )
        self._shap: Optional[SHAPExplainer] = None

    @property
    def shap_explainer(self) -> SHAPExplainer:
        if self._shap is None:
            self._shap = SHAPExplainer(self.cross, self.cfg.shap_max_evals)
        return self._shap

    def analyse(
        self,
        text_a: str,
        text_b: str,
        run_shap: bool = True,
    ) -> Dict:
        """
        Full 3-stage analysis of a sentence / document pair.

        Returns
        -------
        {
          "bm25_score":    float,   # lexical overlap [0,1]
          "sbert_score":   float,   # semantic cosine [0,1]
          "ce_score":      float,   # cross-encoder   [0,1]
          "shap_tokens":   list,    # top token attributions (if run_shap)
          "model_used":    str,     # which cross-encoder was loaded
        }
        """
        bm25_score = BM25Retriever([text_b]).pairwise_score(text_a, text_b)
        sbert_score = self.sbert.similarity(text_a, text_b)
        ce_score = self.cross.score(text_a, text_b)

        shap_tokens: List[Dict] = []
        if run_shap:
            try:
                shap_tokens = self.shap_explainer.explain(text_a, text_b)
            except Exception:
                shap_tokens = []

        return {
            "bm25_score": bm25_score,
            "sbert_score": sbert_score,
            "ce_score": ce_score,
            "shap_tokens": shap_tokens,
            "model_used": self.cross._model_name,
        }