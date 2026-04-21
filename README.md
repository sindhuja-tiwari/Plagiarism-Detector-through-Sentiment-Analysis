# Plagiarism-Detector-through-Sentiment-Analysis
A dark-themed, browser-based plagiarism detection tool that simulates a multi-stage NLP pipeline — combining lexical retrieval, semantic similarity, cross-encoder scoring, and SHAP-style token attribution — all running client-side with zero dependencies.
Live demo: your-vercel-url.vercel.app

Overview
PlagiarismIQ analyses two pieces of text and determines the likelihood of plagiarism or paraphrasing through a four-stage pipeline. Unlike simple keyword matchers, it detects semantic similarity — catching paraphrased content that shares meaning but not exact words.

Pipeline
BM25  →  SBERT  →  Cross-encoder  →  SHAP

Features
Three-tier verdict — Plagiarism Likely / Possible Paraphrase / No Match Found
Four live score meters — BM25, SBERT, Cross-encoder, and Semantic Lift (CE − BM25)
Phrase-level highlighting — colour-coded as Exact, Stem, or Semantic match
SHAP attribution panel — ranked phrases with positive/negative influence scores
Sentence-level breakdown table — each sentence in B matched to its closest sentence in A, with a risk pill (High / Med / Low)
Adjustable threshold — slide from 40% to 90% to tune sensitivity
SHAP toggle — disable attribution step for faster analysis
Load sample — pre-loaded paraphrase example for instant demo

How Scoring Works
BM25
Classic information retrieval ranking. Measures term frequency and inverse document frequency across the two texts. High BM25 = lots of shared keywords.
SBERT (simulated)
Token-overlap similarity weighted by stem matching and a synonym map. Approximates semantic cosine similarity — two words with the same meaning score partially even without exact match.
Cross-encoder
The primary confidence signal. Computed as a weighted blend:
CE = SBERT × 0.58 + BM25 × 0.28 + TokenSim × 0.16 + bias
Semantic Lift (Δ)
CE − BM25 — a positive lift means the texts are more semantically similar than their surface keywords suggest, which is the hallmark of deliberate paraphrasing.
SHAP Attribution
Each sentence in the suspect text is scored against the reference. Sentences are ranked by attribution score and labelled positive (drives plagiarism verdict) or negative (pulls away from it).
