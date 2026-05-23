# NextUp Recommender — Checkpoint (May 23, 2026)

A snapshot of where the project stands after a multi-month pause. Use this as a briefing document to bring a future Claude session up to speed and to seed a refreshed project plan.

> **How to use this file later:** Feed it back to Claude with a prompt like _"Read checkpoint_may_23.md and produce an updated project plan with concrete next-step tasks, suggested milestones, and a recommended order of work."_ The checkboxes below are the single source of truth for current progress.

---

## 1. Project At-A-Glance

**NextUp** is a movie recommendation system that combines two complementary "arms":

1. **Behavioral arm** — a Mamba4Rec sequential model trained on MovieLens-32M user histories.
2. **Content arm** — BGE-large embeddings of ~190K Wikipedia movie plots, indexed in FAISS for semantic retrieval driven by natural-language user queries.

A **reranker** adaptively blends the two arms with a dynamic α weight (specific mood/genre/era queries → favor content; general browsing or strong user history → favor Mamba). A **graduation pipeline** is intended to migrate cold-start movies from content-only into the behavioral catalog once they accumulate enough interactions.

The current public-facing posture (README, resume.md) frames the system as a **dual-arm architecture**. Internally there is older material (`LLMintegrationEGG.md`, `IntegrationPlan.md/pdf`) referencing a more deeply coupled gated-fusion approach where LLM mood vectors were injected directly into Mamba's hidden space. That direction was **abandoned in favor of the dual-arm/reranker design** that is now in the repo — see Section 6 for the historical context worth keeping in mind.

---

## 2. System Architecture (concise)

```
                       ┌────────────────────────────────┐
   User query  ──────▶ │  IntentParser + LLMEncoder     │── query_emb (1024-d, BGE) ──┐
                       │  (llm_encoder.py)              │                              │
                       └────────────────────────────────┘                              │
                                                                                       ▼
   User history ──────────────────────┐                              ┌──────────────────────────┐
                                       ▼                              │  ContentTower (FAISS)    │
                              ┌──────────────────┐                    │  content_tower.py        │
                              │  Mamba4Rec       │                    │  + plots_metadata.parquet│
                              │  mamba4rec.py    │                    └──────────────────────────┘
                              │  (BPR/CE losses) │                                  │
                              └──────────────────┘                                  │
                                       │                                            │
                              mamba_scores                                content_scores
                                       │                                            │
                                       ▼                                            ▼
                              ┌────────────────────────────────────────────────────────┐
                              │  Reranker (reranker.py)                                 │
                              │  • min-max normalizes Mamba scores                      │
                              │  • computes α from query specificity / profile presence │
                              │  • blends: α·content + (1-α)·mamba                      │
                              │  • group ranking: mean − λ·std (fairness-weighted)      │
                              └────────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                          Top-K recommendations
                                                       │
                                                       ▼
                              ┌────────────────────────────────────┐
                              │  ChatProvider (chat_provider.py)   │  → natural-language reply
                              │  OpenAI / Claude / Gemini backends │
                              └────────────────────────────────────┘

   Side systems:
   • EmbeddingManager (embedding_store.py)  – session mood store + persistent profile store
                                              (in-memory + Redis backends)
   • GraduationManager (graduation.py)      – tracks cold-start interactions, decides retrain triggers
   • Data pipeline (pipeline/)              – Wikipedia dump → plot extraction → Wikidata filter
                                              → MovieLens join → TMDB/Wikipedia backfill → consolidate
                                              → BGE encode → FAISS index
```

### Module map

| File                    | Role                                                                                |
|-------------------------|-------------------------------------------------------------------------------------|
| `mamba4rec.py`          | Pure-sequential Mamba model (RecBole-compatible). Has `MambaPureTorch` CPU fallback.|
| `content_tower.py`      | Wraps FAISS index + metadata; supports query/profile blended search and add-movie.  |
| `reranker.py`           | `compute_alpha`, `Reranker.blend`, `Reranker.rank_group`.                           |
| `inference.py`          | `DualArmEngine` orchestrator: `recommend`, `recommend_group`, `record_interaction`. |
| `llm_encoder.py`        | `LLMEncoder` (BGE wrapper with cache) + keyword-based `IntentParser`.               |
| `embedding_store.py`    | Abstract + in-memory + Redis stores; `EmbeddingManager` ties mood + profile.        |
| `chat_provider.py`      | Provider-agnostic chat factory (OpenAI / Anthropic / Gemini) + prompt formatter.    |
| `graduation.py`         | Threshold-based new-movie graduation queue with persistence.                         |
| `train.py`              | Single-phase Mamba training; supports embedding expansion for retraining.            |
| `pipeline/`             | 4-stage ETL: extract → filter/join → backfill → consolidate → encode/index.          |
| `tests/`                | Per-module unit tests + `test_integration.py` for end-to-end dual-arm flow.          |
| `config_ml32m.yaml`     | Live config used by `train.py`.                                                      |
| `example_config.yaml`   | Annotated reference config (kept in sync conceptually).                              |

---

## 3. Progress Checkboxes

Treat each box as the authoritative state. Edit in place as work progresses; the resolution of a box is "code merged + tests passing for that piece."

### 3.1 Core models & retrieval
- [x] Mamba4Rec sequential model (`mamba4rec.py`) — pure-sequential, no fusion remnants
- [x] CPU fallback `MambaPureTorch` for environments without `mamba_ssm`
- [x] Single-phase training script (`train.py`) with embedding-expansion support
- [x] ContentTower over BGE-large (1024-d) FAISS `IndexFlatIP`
- [x] Query + profile blended search with α weighting
- [x] Reranker with adaptive α from query specificity
- [x] Group recommendation via fairness-weighted aggregation (mean − λ·std)
- [ ] **End-to-end training run on MovieLens-32M committed with reported metrics** (script exists, no recorded run)
- [ ] **FAISS index actually built and saved** under `data/plots.faiss` (build path exists, no artifact)

### 3.2 LLM / chatbot layer
- [x] BGE encoder wrapper (`LLMEncoder`) with LRU-ish cache
- [x] Keyword-based `IntentParser` (mood / genre / era / constraints)
- [x] Provider-agnostic `ChatProvider` factory (OpenAI / Claude / Gemini)
- [x] Prompt-formatting helper for recommendation responses
- [ ] **Real LLM-driven intent parsing** (current `IntentParser` is pure keyword matching — not using any LLM)
- [ ] **Multi-turn conversation state** (each call is currently stateless)
- [ ] **LLM-generated taste summaries** (mentioned in `resume.md` future-work; not started)

### 3.3 Embedding / profile storage
- [x] In-memory mood store with TTL
- [x] In-memory profile store with decay + adaptive LR
- [x] Redis mood store implementation
- [x] `EmbeddingManager` unified facade
- [ ] **PostgreSQL + pgvector profile store** (resume.md claims this exists; only Redis is implemented)
- [ ] **Redis profile store** (only mood is wired to Redis)
- [ ] **Profile-warm-start strategy** for users with <N interactions

### 3.4 Data pipeline
- [x] Wikipedia dump download (`pipeline/download.py`)
- [x] Plot section extraction with wikitext cleanup (`extract_plots.py`)
- [x] Wikidata SPARQL bridge for movie/TV classification (`wikidata_bridge.py`, `filter_plots.py`)
- [x] MovieLens links join (`join_movielens.py`)
- [x] Async Wikipedia/TMDB backfill for unmatched MovieLens IDs (`backfill.py`)
- [x] Consolidation + final reports (`consolidate.py`)
- [x] BGE encoding + FAISS index builder (`encode_plots.py`)
- [x] Orchestrator (`run_pipeline.py`)
- [ ] **End-to-end pipeline executed against real Wikipedia dump + outputs committed/published** (190K figure is documented but no artifact in repo)
- [ ] **Resumability / incremental re-runs** beyond per-stage parquet caches
- [ ] **CI smoke test** running the pipeline on a tiny fixture dump

### 3.5 Graduation / retraining loop
- [x] `GraduationManager` with persistent queue and thresholds
- [x] Embedding-table expansion in `train.py` for adding graduated items
- [ ] **Hooking `DualArmEngine.record_interaction` into `GraduationManager`** (record path exists for profiles, not for graduation)
- [ ] **Automated retraining trigger** (`should_retrain_by_threshold` exists; nothing calls it)
- [ ] **Item-ID remapping** when MovieLens IDs ≠ Mamba's internal item IDs after graduation
- [ ] **Evaluation comparing pre/post-graduation recall on cold-start items**

### 3.6 Serving / API
- [ ] **REST/FastAPI layer** (README claims "RESTful API architecture" — no `api.py` / `server.py` exists)
- [ ] **Authentication / session management** wiring `session_id` through `EmbeddingManager`
- [ ] **Async I/O for content + Mamba scoring**
- [ ] **Containerization** (resume.md mentions Docker; no `Dockerfile`)
- [ ] **Deployment guide / production playbook**

### 3.7 Evaluation & benchmarking
- [x] Unit tests for each module under `tests/` (>10 files, see `tests/`)
- [x] Integration test covering content-only flow, profile build, graduation, group rec, intent parsing, α spectrum
- [ ] **Offline metrics on MovieLens-32M** (Hit@K, NDCG@K, MRR@K) committed somewhere
- [ ] **Transformer baseline (SASRec / BERT4Rec) for the "why Mamba" comparison** promised in README
- [ ] **Cold-start evaluation** (recall on items absent from training)
- [ ] **Group-recommendation evaluation harness**
- [ ] **A/B-testing harness** for α strategies (mentioned in resume.md future-work)

### 3.8 Documentation
- [x] README with architecture diagram (`NextUpRecommenderArchitecture.png`)
- [x] `example_config.yaml` annotated reference config
- [x] Legacy `IntegrationPlan.md` + PDFs covering the abandoned gated-fusion design
- [x] `resume.md` framing the project for job applications
- [ ] **Up-to-date design doc reflecting the current dual-arm architecture** (the legacy IntegrationPlan is stale)
- [ ] **CHANGELOG / decision log** capturing the pivot away from gated fusion
- [ ] **Demo notebook / quickstart that actually runs end-to-end on a tiny fixture dataset**

---

## 4. Quick "what's missing to ship a demo" summary

If the goal is a credible demo, the gating items are roughly:
1. **Build the FAISS index** from a real run of the pipeline (or a curated subset).
2. **Train Mamba** on ML-32M (or ML-1M as a smaller pilot) and check in a checkpoint or download link.
3. **Wire a thin API** (FastAPI) around `DualArmEngine` so the chatbot can call it.
4. **Replace keyword `IntentParser` with an LLM call** so the natural-language framing in the README is honest.
5. **Stand up at least an in-memory demo** without Postgres/Redis — the in-memory stores already cover this.

Items 1–4 are independently doable; #4 is the smallest but highest narrative-impact item.

---

## 5. Known inconsistencies between docs and code

These are worth resolving so the project plan is grounded in reality, not aspiration:

- **README** says intent parsing handles natural-language queries → **Reality:** `IntentParser` is regex/keyword.
- **README** "RESTful API architecture for frontend integration" → **Reality:** no API layer exists.
- **resume.md** claims Postgres+pgvector profile storage → **Reality:** only in-memory and Redis-mood implementations exist.
- **resume.md** claims Dockerized microservices → **Reality:** no Dockerfile committed.
- **README** says "190K+ Wikipedia movie plots encoded" → **Reality:** the pipeline can produce this; there is no committed artifact or recorded run.

Decide per item: build it, or update the docs to match what's true today.

---

## 6. Historical context (do not lose)

- `LLMintegrationEGG.md` and `IntegrationPlan.md/pdf` describe an earlier **gated-fusion** design where LLM mood/intent vectors were injected directly into Mamba's hidden space, with a 3-phase training plan (freeze Mamba → train fusion+projection → joint fine-tune). That direction was **superseded by the dual-arm + reranker architecture** now in the repo.
- The folder name `fusion_git` is a relic of that original direction.
- The 3-phase training plan is no longer relevant; current training is single-phase Mamba only.
- Anything in the old IntegrationPlan referring to `LLMprojection.py`, "fusion modules," or LLM-vector alignment with Mamba's hidden_size is **historical and should not be re-implemented** unless we explicitly revive that direction.

---

## 7. Suggested prompts for the next session

Pick the one that matches your goal when you return:

- **For a refreshed plan:** _"Read `checkpoint_may_23.md` and produce a prioritized roadmap with concrete next-step tasks. Group by 'unblock a demo' vs. 'long-term polish'. For each task, name the file(s) involved and rough effort (S/M/L)."_
- **For doc/code reconciliation:** _"Read `checkpoint_may_23.md` Section 5. For each inconsistency, propose either the minimal code to make the docs true or the minimal doc edit to make the docs honest, and recommend which is the better call."_
- **For evaluation focus:** _"Read `checkpoint_may_23.md`. Design an evaluation plan to back up the README's claims about dual-arm > content-only > Mamba-only. Specify datasets, metrics, baselines, and the scripts to add."_
- **For shipping a demo:** _"Read `checkpoint_may_23.md` Section 4. Walk through the five gating items in order and propose the smallest possible implementation of each, with the explicit goal of getting a one-click local demo running."_
