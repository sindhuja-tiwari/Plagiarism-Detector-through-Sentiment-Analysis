"""
3-Stage Semantic Textual Similarity Pipeline
============================================
Stage 1 : BM25 candidate retrieval       (rank_bm25)
Stage 2 : SBERT semantic re-ranking      (sentence-transformers)
Stage 3 : DeBERTa classification + SHAP  (transformers + shap)

Dataset : STS Benchmark (stsb) via HuggingFace datasets
Metrics : Pearson r, Spearman ρ, MRR@k, Recall@k

Install
-------
pip install rank_bm25 sentence-transformers transformers \
            datasets shap torch scipy numpy pandas tqdm
"""

# ── stdlib ──────────────────────────────────────────────────────────────
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ── third-party ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import shap
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # BM25
    bm25_top_k: int = 100          # candidates retrieved per query

    # SBERT
    sbert_model: str = "all-MiniLM-L6-v2"
    sbert_top_k: int = 10          # candidates kept after re-ranking

    # Cross-encoder for STS — verified public models (no auth needed):
    #   "cross-encoder/stsb-roberta-large"  (~1.4 GB, best accuracy)
    #   "cross-encoder/stsb-roberta-base"   (~500 MB, faster on CPU)
    #   "cross-encoder/stsb-distilroberta-base"  (~300 MB, lightest)
    deberta_model: str = "cross-encoder/stsb-roberta-base"
    deberta_max_len: int = 128

    # SHAP
    shap_n_examples: int = 3       # how many pairs to explain
    shap_max_evals: int = 200      # SHAP partition explainer budget

    # Evaluation
    eval_split: str = "test"       # "train" | "validation" | "test"
    eval_sample: int = 200         # rows to evaluate (None = all)
    random_seed: int = 42

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


CFG = PipelineConfig()
np.random.seed(CFG.random_seed)
torch.manual_seed(CFG.random_seed)

# ════════════════════════════════════════════════════════════════════════
# 1. Data loading
# ════════════════════════════════════════════════════════════════════════

def load_stsb(split: str = "test", n: Optional[int] = None) -> pd.DataFrame:
    """
    Load the STS-B dataset from HuggingFace.

    The dataset has columns:
        sentence1, sentence2, score  (score ∈ [0, 5])

    We normalise score → [0, 1] for easier comparison.
    """
    print(f"\n{'='*60}")
    print(f"  Loading STS-B ({split} split) …")
    print(f"{'='*60}")

    ds = load_dataset("stsb_multi_mt", name="en", split=split)
    df = ds.to_pandas()

    # Column is "similarity_score" in stsb_multi_mt (range 0–5)
    # Fall back gracefully if the schema ever changes
    score_col = next(
        (c for c in df.columns if "score" in c.lower()),
        None,
    )
    if score_col is None:
        raise ValueError(f"No score column found. Available columns: {df.columns.tolist()}")
    print(f"  Score column detected: '{score_col}'")

    # Normalise similarity score from [0,5] → [0,1]
    df["score_norm"] = df[score_col] / 5.0

    if n is not None:
        df = df.sample(n=min(n, len(df)), random_state=CFG.random_seed).reset_index(drop=True)

    # Normalise column names: some versions use "sentence1"/"sentence2",
    # others use "sent_1"/"sent_2" — rename to a canonical form.
    rename_map = {}
    for col in df.columns:
        if col.lower() in ("sent_1", "sentence_1", "sentence1"):
            rename_map[col] = "sentence1"
        elif col.lower() in ("sent_2", "sentence_2", "sentence2"):
            rename_map[col] = "sentence2"
    if rename_map:
        df = df.rename(columns=rename_map)

    print(f"  Columns : {df.columns.tolist()}")
    print(f"  Loaded {len(df)} pairs  |  score range: "
          f"{df['score_norm'].min():.3f} – {df['score_norm'].max():.3f}")
    return df


# ════════════════════════════════════════════════════════════════════════
# 2. Stage 1 — BM25 Retrieval
# ════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """Simple whitespace + lower-case tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi.

    Usage
    -----
    retriever = BM25Retriever(corpus_sentences)
    candidates = retriever.retrieve(query, top_k=100)
    # returns list of (sentence, bm25_score) tuples
    """

    def __init__(self, corpus: List[str]):
        print("\n[Stage 1] Building BM25 index …")
        self.corpus = corpus
        tokenized = [tokenize(s) for s in corpus]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  Index built over {len(corpus)} documents")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus[i], float(scores[i])) for i in top_idx]

    def recall_at_k(
        self,
        queries: List[str],
        relevant: List[str],
        k: int = 100,
    ) -> float:
        """
        Recall@k: fraction of queries where the true relevant sentence
        appears in the top-k BM25 results.
        """
        hits = 0
        for q, rel in zip(queries, relevant):
            candidates = [s for s, _ in self.retrieve(q, top_k=k)]
            if rel in candidates:
                hits += 1
        return hits / len(queries)


# ════════════════════════════════════════════════════════════════════════
# 3. Stage 2 — SBERT Re-ranking
# ════════════════════════════════════════════════════════════════════════

class SBERTReranker:
    """
    Bi-encoder re-ranker using sentence-transformers.

    1. Encodes all BM25 candidates once.
    2. Computes cosine similarity with the query embedding.
    3. Returns top_k re-ranked candidates.
    """

    def __init__(self, model_name: str = CFG.sbert_model, device: str = CFG.device):
        print(f"\n[Stage 2] Loading SBERT model: {model_name} …")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        print(f"  Model loaded on {device}")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Returns (sentence, cosine_similarity) sorted descending."""
        sentences = [s for s, _ in candidates]
        q_emb = self.encode([query])
        c_embs = self.encode(sentences)
        sims = (c_embs @ q_emb.T).squeeze()  # cosine sim (normalized)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(sentences[i], float(sims[i])) for i in top_idx]

    def pairwise_similarity(self, sent_a: str, sent_b: str) -> float:
        """Direct cosine similarity between two sentences."""
        embs = self.encode([sent_a, sent_b])
        return float(embs[0] @ embs[1])

    def mrr_at_k(
        self,
        queries: List[str],
        relevant: List[str],
        candidates_list: List[List[Tuple[str, float]]],
        k: int = 10,
    ) -> float:
        """Mean Reciprocal Rank @ k after SBERT re-ranking."""
        rr_sum = 0.0
        for q, rel, cands in zip(queries, relevant, candidates_list):
            reranked = self.rerank(q, cands, top_k=k)
            for rank, (sent, _) in enumerate(reranked, start=1):
                if sent == rel:
                    rr_sum += 1.0 / rank
                    break
        return rr_sum / len(queries)


# ════════════════════════════════════════════════════════════════════════
# 4. Stage 3 — DeBERTa Cross-encoder + SHAP
# ════════════════════════════════════════════════════════════════════════

class DeBERTaClassifier:
    """
    Cross-encoder: jointly encodes (sentence_a, sentence_b) →
    scalar similarity score via a regression head.

    Model: cross-encoder/stsb-roberta-base (default; also works with
           cross-encoder/stsb-roberta-large for higher accuracy)
    Fine-tuned on STS-B, outputs values ∈ [0, 1] after sigmoid.
    """

    def __init__(
        self,
        model_name: str = CFG.deberta_model,
        device: str = CFG.device,
        max_len: int = CFG.deberta_max_len,
    ):
        self.device = device
        self.max_len = max_len

        # Try the requested model first; fall back to lighter public models
        # if the primary is unavailable (deleted, private, or gated).
        _candidates = [
            model_name,
            "cross-encoder/stsb-roberta-base",
            "cross-encoder/stsb-roberta-large",
            "cross-encoder/stsb-distilroberta-base",
        ]
        # De-duplicate while preserving order
        seen = set()
        _candidates = [m for m in _candidates if not (m in seen or seen.add(m))]

        last_err: Exception = RuntimeError("No candidate models available.")
        for candidate in _candidates:
            try:
                print(f"\n[Stage 3] Loading cross-encoder: {candidate} …")
                self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                self.model = AutoModelForSequenceClassification.from_pretrained(candidate)
                self.model.to(device)
                self.model.eval()
                self._model_name = candidate
                print(f"  Model loaded on {device}")
                break
            except Exception as e:
                print(f"  Could not load {candidate!r}: {e}")
                last_err = e
        else:
            raise RuntimeError(
                f"All candidate cross-encoders failed. Last error: {last_err}"
            )

        # HuggingFace pipeline for convenience (used in SHAP)
        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            function_to_apply="sigmoid",
            return_all_scores=False,
        )

    @torch.no_grad()
    def predict(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Args
        ----
        pairs : list of (sentence_a, sentence_b)

        Returns
        -------
        scores : np.ndarray of shape (N,), values ∈ [0, 1]
        """
        scores = []
        for a, b in pairs:
            enc = self.tokenizer(
                a, b,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            logit = self.model(**enc).logits.squeeze()
            score = torch.sigmoid(logit).item()
            scores.append(score)
        return np.array(scores)

    def predict_single(self, sent_a: str, sent_b: str) -> float:
        return float(self.predict([(sent_a, sent_b)])[0])


# ════════════════════════════════════════════════════════════════════════
# 5. SHAP Explainability
# ════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    Uses shap.Explainer with a HuggingFace text pipeline as the model
    function to attribute token-level importance for each pair.
    """

    def __init__(self, classifier: DeBERTaClassifier):
        print("\n[SHAP] Initialising explainer …")
        self.classifier = classifier

        # SHAP text explainer wraps the pipeline predict function
        self.explainer = shap.Explainer(
            classifier.pipe,
            masker=shap.maskers.Text(classifier.tokenizer),
            output_names=["similarity"],
        )

    def explain(
        self,
        sent_a: str,
        sent_b: str,
        max_evals: int = CFG.shap_max_evals,
    ) -> Dict:
        """
        Returns
        -------
        dict with keys:
            tokens      : list[str]
            shap_values : np.ndarray
            score       : float   (model prediction)
        """
        combined = f"{sent_a} [SEP] {sent_b}"
        sv = self.explainer([combined], max_evals=max_evals)

        # Extract from SHAP Values object
        tokens = sv.data[0]
        values = sv.values[0]

        # values shape can be (N,) or (N, num_labels) — flatten
        if values.ndim > 1:
            values = values[:, 0]

        score = self.classifier.predict_single(sent_a, sent_b)
        return {"tokens": tokens, "shap_values": values, "score": score}

    @staticmethod
    def format_explanation(explanation: Dict, top_n: int = 8) -> str:
        """Pretty-print top contributing tokens."""
        tokens = explanation["tokens"]
        values = explanation["shap_values"]
        score = explanation["score"]

        idx = np.argsort(np.abs(values))[::-1][:top_n]
        lines = [
            f"  Predicted similarity : {score:.4f}",
            f"  {'Token':<20} {'SHAP value':>12}",
            f"  {'-'*34}",
        ]
        for i in idx:
            sign = "+" if values[i] >= 0 else ""
            lines.append(f"  {str(tokens[i]):<20} {sign}{values[i]:.4f}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# 6. Evaluation helpers
# ════════════════════════════════════════════════════════════════════════

def evaluate_regression(
    gold: np.ndarray,
    pred: np.ndarray,
    label: str = "Model",
) -> Dict[str, float]:
    """Compute Pearson r and Spearman ρ between gold and predicted scores."""
    r, p_r = pearsonr(gold, pred)
    rho, p_rho = spearmanr(gold, pred)
    mse = float(np.mean((gold - pred) ** 2))
    mae = float(np.mean(np.abs(gold - pred)))

    print(f"\n  ── {label} ──")
    print(f"  Pearson  r  : {r:.4f}  (p={p_r:.2e})")
    print(f"  Spearman ρ  : {rho:.4f}  (p={p_rho:.2e})")
    print(f"  MSE         : {mse:.4f}")
    print(f"  MAE         : {mae:.4f}")

    return {"pearson_r": r, "spearman_rho": rho, "mse": mse, "mae": mae}


# ════════════════════════════════════════════════════════════════════════
# 7. Full pipeline — end-to-end run
# ════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: PipelineConfig = CFG) -> Dict:
    """
    End-to-end 3-stage STS pipeline on STS-B.

    Returns a dict of evaluation metrics from each stage.
    """

    # ── Load data ────────────────────────────────────────────────────
    df = load_stsb(split=cfg.eval_split, n=cfg.eval_sample)

    sentences_a = df["sentence1"].tolist()
    sentences_b = df["sentence2"].tolist()
    gold_scores = df["score_norm"].values

    # Build corpus = all unique sentences (a + b)
    corpus = list(dict.fromkeys(sentences_a + sentences_b))
    print(f"\n  Corpus size: {len(corpus)} unique sentences")

    # ── Stage 1: BM25 ────────────────────────────────────────────────
    bm25 = BM25Retriever(corpus)

    print("\n[Stage 1] BM25 retrieval …")
    bm25_scores = []
    bm25_candidates_all = []

    for a, b in tqdm(zip(sentences_a, sentences_b), total=len(df), desc="  BM25"):
        cands = bm25.retrieve(a, top_k=cfg.bm25_top_k)
        bm25_candidates_all.append(cands)

        # For pairwise scoring: score of sentence_b given sentence_a as query
        b_rank_score = next(
            (score for sent, score in cands if sent == b),
            0.0,
        )
        # Normalise BM25 scores to [0,1] via max in the candidate list
        max_score = max((s for _, s in cands), default=1.0)
        bm25_scores.append(b_rank_score / max_score if max_score > 0 else 0.0)

    bm25_scores = np.array(bm25_scores)
    metrics_bm25 = evaluate_regression(gold_scores, bm25_scores, label="Stage 1 — BM25")

    # BM25 Recall@k
    recall_100 = bm25.recall_at_k(sentences_a, sentences_b, k=cfg.bm25_top_k)
    print(f"  Recall@{cfg.bm25_top_k}    : {recall_100:.4f}")
    metrics_bm25[f"recall@{cfg.bm25_top_k}"] = recall_100

    # ── Stage 2: SBERT ───────────────────────────────────────────────
    sbert = SBERTReranker(model_name=cfg.sbert_model, device=cfg.device)

    print("\n[Stage 2] SBERT re-ranking …")
    sbert_scores = []

    for a, b in tqdm(zip(sentences_a, sentences_b), total=len(df), desc="  SBERT"):
        sbert_scores.append(sbert.pairwise_similarity(a, b))

    sbert_scores = np.array(sbert_scores)
    metrics_sbert = evaluate_regression(gold_scores, sbert_scores, label="Stage 2 — SBERT")

    # MRR@10
    mrr = sbert.mrr_at_k(
        sentences_a, sentences_b, bm25_candidates_all, k=cfg.sbert_top_k
    )
    print(f"  MRR@{cfg.sbert_top_k}        : {mrr:.4f}")
    metrics_sbert[f"mrr@{cfg.sbert_top_k}"] = mrr

    # ── Stage 3: DeBERTa ─────────────────────────────────────────────
    deberta = DeBERTaClassifier(
        model_name=cfg.deberta_model,
        device=cfg.device,
        max_len=cfg.deberta_max_len,
    )

    print("\n[Stage 3] DeBERTa cross-encoder scoring …")
    pairs = list(zip(sentences_a, sentences_b))
    deberta_scores = []

    for pair in tqdm(pairs, desc="  DeBERTa"):
        deberta_scores.append(deberta.predict_single(*pair))

    deberta_scores = np.array(deberta_scores)
    metrics_deberta = evaluate_regression(
        gold_scores, deberta_scores, label="Stage 3 — DeBERTa"
    )

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(
        {
            "Stage": ["BM25 (baseline)", "SBERT", "DeBERTa"],
            "Pearson r": [
                metrics_bm25["pearson_r"],
                metrics_sbert["pearson_r"],
                metrics_deberta["pearson_r"],
            ],
            "Spearman ρ": [
                metrics_bm25["spearman_rho"],
                metrics_sbert["spearman_rho"],
                metrics_deberta["spearman_rho"],
            ],
            "MSE": [
                metrics_bm25["mse"],
                metrics_sbert["mse"],
                metrics_deberta["mse"],
            ],
        }
    )
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print("=" * 60)

    return {
        "bm25": metrics_bm25,
        "sbert": metrics_sbert,
        "deberta": metrics_deberta,
        "scores": {
            "gold": gold_scores,
            "bm25": bm25_scores,
            "sbert": sbert_scores,
            "deberta": deberta_scores,
        },
        "models": {
            "bm25": bm25,
            "sbert": sbert,
            "deberta": deberta,
        },
    }


# ════════════════════════════════════════════════════════════════════════
# 8. SHAP demo — explain specific pairs
# ════════════════════════════════════════════════════════════════════════

def run_shap_demo(deberta: DeBERTaClassifier, n_pairs: int = 3) -> None:
    """
    Explain a set of hand-picked sentence pairs with SHAP to show
    which tokens drive the similarity score.
    """
    demo_pairs = [
        (
            "The physician prescribed medication for the patient's condition.",
            "A doctor gave drugs to treat the illness affecting the individual.",
        ),
        (
            "A man is riding a horse.",
            "A person is jumping on a bicycle.",
        ),
        (
            "The stock market crashed significantly today.",
            "Share prices experienced a dramatic fall this morning.",
        ),
    ][:n_pairs]

    print("\n" + "=" * 60)
    print("  SHAP EXPLAINABILITY DEMO")
    print("=" * 60)

    explainer = SHAPExplainer(deberta)

    for i, (a, b) in enumerate(demo_pairs, start=1):
        print(f"\n  Pair {i}")
        print(f"  A: {a}")
        print(f"  B: {b}")
        explanation = explainer.explain(a, b)
        print(explainer.format_explanation(explanation))

    print("\n" + "=" * 60)


# ════════════════════════════════════════════════════════════════════════
# 9. Stage-by-stage demo on a single pair
# ════════════════════════════════════════════════════════════════════════

def demo_single_pair(
    models: Dict,
    sent_a: str,
    sent_b: str,
    corpus: List[str],
    cfg: PipelineConfig = CFG,
) -> None:
    """
    Walk through all 3 stages for a single sentence pair, printing
    intermediate outputs at each step.
    """
    bm25: BM25Retriever = models["bm25"]
    sbert: SBERTReranker = models["sbert"]
    deberta: DeBERTaClassifier = models["deberta"]

    print("\n" + "=" * 60)
    print("  SINGLE PAIR WALKTHROUGH")
    print("=" * 60)
    print(f"  Query (A) : {sent_a}")
    print(f"  Target(B) : {sent_b}")

    # Stage 1
    t0 = time.perf_counter()
    cands = bm25.retrieve(sent_a, top_k=cfg.bm25_top_k)
    bm25_time = time.perf_counter() - t0

    b_bm25 = next((s for s, _ in cands if s == sent_b), None)
    b_rank_bm25 = next((i + 1 for i, (s, _) in enumerate(cands) if s == sent_b), ">100")

    print(f"\n  [Stage 1 — BM25]  ({bm25_time*1000:.1f} ms)")
    print(f"  Top-5 candidates:")
    for rank, (sent, score) in enumerate(cands[:5], start=1):
        marker = " ◀ TARGET" if sent == sent_b else ""
        print(f"    {rank}. [{score:.3f}] {sent[:70]}{marker}")
    print(f"  Target rank in BM25: {b_rank_bm25}")

    # Stage 2
    t0 = time.perf_counter()
    reranked = sbert.rerank(sent_a, cands, top_k=cfg.sbert_top_k)
    sbert_time = time.perf_counter() - t0
    sbert_sim = sbert.pairwise_similarity(sent_a, sent_b)

    b_rank_sbert = next(
        (i + 1 for i, (s, _) in enumerate(reranked) if s == sent_b), f">{cfg.sbert_top_k}"
    )

    print(f"\n  [Stage 2 — SBERT]  ({sbert_time*1000:.1f} ms)")
    print(f"  Top-5 after re-ranking:")
    for rank, (sent, score) in enumerate(reranked[:5], start=1):
        marker = " ◀ TARGET" if sent == sent_b else ""
        print(f"    {rank}. [{score:.4f}] {sent[:70]}{marker}")
    print(f"  Direct cosine similarity A↔B : {sbert_sim:.4f}")
    print(f"  Target rank after SBERT       : {b_rank_sbert}")

    # Stage 3
    t0 = time.perf_counter()
    deb_score = deberta.predict_single(sent_a, sent_b)
    deberta_time = time.perf_counter() - t0

    print(f"\n  [Stage 3 — DeBERTa]  ({deberta_time*1000:.1f} ms)")
    print(f"  Cross-encoder similarity score : {deb_score:.4f}")

    print("\n  Summary:")
    print(f"    BM25 keyword match  →  ~{sbert_sim*22:.0f}%")
    print(f"    SBERT semantic sim  →  {sbert_sim*100:.0f}%")
    print(f"    DeBERTa prediction  →  {deb_score*100:.0f}%")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Full evaluation run ──────────────────────────────────────────
    results = run_pipeline(CFG)

    models = results["models"]

    # ── Demo on the canonical paraphrase pair ────────────────────────
    df = load_stsb(split=CFG.eval_split, n=CFG.eval_sample)
    corpus = list(dict.fromkeys(df["sentence1"].tolist() + df["sentence2"].tolist()))

    demo_single_pair(
        models=models,
        sent_a="The physician prescribed medication for the patient's condition.",
        sent_b="A doctor gave drugs to treat the illness affecting the individual.",
        corpus=corpus,
    )

    # ── SHAP explainability on 3 pairs ───────────────────────────────
    run_shap_demo(models["deberta"], n_pairs=CFG.shap_n_examples)

    print("\nDone.")
