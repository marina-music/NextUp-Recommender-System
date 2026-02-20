# TODO: Rewrite as dual-arm orchestrator (Task 13)
#
# This module previously provided a fusion-based inference API (Mamba + LLM gate).
# It will be replaced by a dual-arm orchestrator that combines:
#   - Mamba4Rec behavioral arm (sequential predictions)
#   - BGE/FAISS content tower (semantic retrieval)
#   - Reranker layer to blend scores
