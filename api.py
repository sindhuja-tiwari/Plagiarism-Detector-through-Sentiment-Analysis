"""
backend/api.py
--------------
Flask REST API that exposes the 3-stage plagiarism pipeline.

Endpoints
---------
POST /api/analyse          — analyse a pair of texts
GET  /api/health           — liveness check
GET  /                     — serves the frontend SPA

Run
---
  python api.py                      # development (debug=True)
  gunicorn -w 1 -b 0.0.0.0:5000 "api:create_app()"  # production
"""

import os
import time
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from pipeline import PlagiarismPipeline, PipelineConfig

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("plagiarism-api")

# ── App factory ───────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

_pipeline: PlagiarismPipeline | None = None


def get_pipeline() -> PlagiarismPipeline:
    global _pipeline
    if _pipeline is None:
        log.info("Initialising pipeline (first request — models will download if needed) …")
        cfg = PipelineConfig(
            sbert_model=os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2"),
            cross_encoder_model=os.getenv(
                "CE_MODEL", "cross-encoder/stsb-roberta-base"
            ),
        )
        _pipeline = PlagiarismPipeline(cfg)
        log.info("Pipeline ready — cross-encoder: %s", _pipeline.cross._model_name)
    return _pipeline


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    CORS(app)

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    # ── Analyse ───────────────────────────────────────────────────────────

    @app.post("/api/analyse")
    def analyse():
        body = request.get_json(silent=True) or {}

        text_a: str = (body.get("text_a") or "").strip()
        text_b: str = (body.get("text_b") or "").strip()
        run_shap: bool = bool(body.get("run_shap", True))
        threshold: float = float(body.get("threshold", 0.70))

        # Validation
        if not text_a or not text_b:
            return jsonify({"error": "Both text_a and text_b are required."}), 400
        if len(text_a) > 10_000 or len(text_b) > 10_000:
            return jsonify({"error": "Text exceeds 10 000 character limit."}), 400

        t0 = time.perf_counter()
        try:
            result = get_pipeline().analyse(text_a, text_b, run_shap=run_shap)
        except Exception as exc:
            log.exception("Pipeline error")
            return jsonify({"error": str(exc)}), 500

        elapsed = round((time.perf_counter() - t0) * 1000)

        ce = result["ce_score"]
        if ce >= threshold:
            verdict = "plagiarism"
            verdict_label = "Likely plagiarism detected"
            verdict_detail = (
                f"Cross-encoder confidence {round(ce*100)}% exceeds your "
                f"{round(threshold*100)}% threshold. High semantic similarity "
                "even if wording differs — review highlighted phrases."
            )
        elif ce >= threshold * 0.70:
            verdict = "warning"
            verdict_label = "Possible paraphrase — review recommended"
            verdict_detail = (
                f"Score {round(ce*100)}% is below threshold but elevated. "
                "May be paraphrase plagiarism. Check token attributions."
            )
        else:
            verdict = "clear"
            verdict_label = "No significant similarity found"
            verdict_detail = (
                f"Cross-encoder score {round(ce*100)}% is below threshold. "
                "Texts appear sufficiently distinct in both lexical and semantic space."
            )

        return jsonify(
            {
                "bm25_score": result["bm25_score"],
                "sbert_score": result["sbert_score"],
                "ce_score": result["ce_score"],
                "shap_tokens": result["shap_tokens"],
                "model_used": result["model_used"],
                "verdict": verdict,
                "verdict_label": verdict_label,
                "verdict_detail": verdict_detail,
                "elapsed_ms": elapsed,
            }
        )

    # ── SPA fallback ──────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.errorhandler(404)
    def not_found(_):
        return send_from_directory(FRONTEND_DIR, "index.html")

    return app


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    log.info("Starting dev server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)