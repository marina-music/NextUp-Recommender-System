# NextUp Recommender — Desktop Checkpoint (2026-05-23)

A reconciled snapshot built on the desktop machine, where the **large untracked data and ML artifacts actually live**. The laptop-side `checkpoint_may_23.md` is structurally correct but undercounts progress because the data files (FAISS index, raw Wikipedia dump, MovieLens-32M, trained checkpoint) are gitignored and never made it onto the laptop. Treat *this* file as the authoritative state.

> **How to use this file later:** Hand it back to Claude with a prompt like:
> _"Read `checkpoint_desktop.md` and produce an updated, prioritized project plan. Group work by 'unblock a demo' vs 'long-term polish', name the files involved, and give rough effort (S/M/L)."_
> Other prompt ideas are in Section 8.

---

## 1. Elevator Pitch

**NextUp** is a movie recommender that combines two arms:

- **Behavioral arm** — Mamba4Rec sequential model trained on MovieLens-32M user histories.
- **Content arm** — BGE-large embeddings (1024-d) of Wikipedia movie/TV plot summaries, indexed in FAISS for semantic retrieval driven by natural-language queries.

A **reranker** dynamically blends the two arms with α weight (specific mood/genre/era → favor content; strong user history → favor Mamba). A **graduation pipeline** is intended to migrate cold-start movies into the behavioral catalog once they accumulate enough interactions.

> **Historical note (don't lose):** Older docs (`LLMintegrationEGG.md`, `IntegrationPlan.md/pdf`, the folder name `fusion`) describe an abandoned **gated-fusion** design where LLM mood vectors were injected directly into Mamba's hidden space (3-phase training: freeze → train fusion → joint fine-tune). That direction was superseded by the current dual-arm + reranker design. Do not re-implement `LLMprojection`, fusion modules, or hidden-size alignment unless we explicitly revive that line.

---

## 2. System Architecture (concise)

```
                       ┌────────────────────────────────┐
   User query  ──────▶ │  IntentParser + LLMEncoder     │── query_emb (1024-d, BGE) ──┐
                       │  (llm_encoder.py)              │                              │
                       └────────────────────────────────┘                              ▼
   User history ──────────────────────┐                              ┌──────────────────────────┐
                                       ▼                              │  ContentTower (FAISS)    │
                              ┌──────────────────┐                    │  content_tower.py        │
                              │  Mamba4Rec       │                    │  + plots_metadata.parquet│
                              │  mamba4rec.py    │                    └──────────────────────────┘
                              │  (BPR / CE loss) │                                  │
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
   • GraduationManager (graduation.py)      – tracks cold-start interactions, retrain triggers
   • Data pipeline (pipeline/)              – Wikipedia dump → plot extract → Wikidata filter
                                              → MovieLens join → TMDB/Wiki backfill
                                              → consolidate → BGE encode → FAISS index
```

### Module map

| File / dir            | Role |
|-----------------------|------|
| `mamba4rec.py`        | Pure-sequential Mamba model (RecBole-compatible). `MambaPureTorch` CPU fallback. |
| `content_tower.py`    | Wraps FAISS index + metadata; query/profile blended search; add-movie API. |
| `reranker.py`         | `compute_alpha`, `Reranker.blend`, `Reranker.rank_group`. |
| `inference.py`        | `DualArmEngine` orchestrator: `recommend`, `recommend_group`, `record_interaction`. |
| `llm_encoder.py`      | `LLMEncoder` (BGE wrapper + cache) + keyword-based `IntentParser`. |
| `embedding_store.py`  | Abstract + in-memory + Redis stores; `EmbeddingManager` ties mood + profile. |
| `chat_provider.py`    | Provider-agnostic chat factory (OpenAI / Anthropic / Gemini) + prompt formatter. |
| `graduation.py`       | Threshold-based new-movie graduation queue with persistence. |
| `train.py`            | Single-phase Mamba training; supports embedding expansion for retraining. |
| `pipeline/`           | 4-stage ETL: extract → filter/join → backfill → consolidate → encode/index. |
| `tests/`              | Per-module unit tests + `test_integration.py` for end-to-end dual-arm flow. |
| `config_ml32m.yaml`   | Live config used by `train.py`. |
| `example_config.yaml` | Annotated reference config. |

---

## 3. What's actually on this disk (the bit the laptop checkpoint missed)

These files are **gitignored** (`data/raw/`, `data/*.parquet`, `*.pt`, `saved/`, `checkpoints/`, `log/`, `log_tensorboard/`), so the laptop has no idea they exist. They represent real, non-trivial progress:

### Data artifacts present
| Path | Size | Meaning |
|---|---|---|
| `data/raw/enwiki-latest-pages-articles-multistream.xml.bz2` | **25.9 GB** | Wikipedia dump (Feb 18 2026). |
| `data/raw/ml-32m/` (ratings.csv, movies.csv, links.csv, tags.csv) | **~950 MB** | MovieLens-32M, unpacked. |
| `data/wiki_plots_raw.parquet` | 148 MB | Stage-1 extracted plots from the dump. |
| `data/wikidata_mapping.parquet` | 7.3 MB | Cached Wikidata SPARQL movie/TV classification. |
| `data/wiki_plots_movies_tv.parquet` | 90.6 MB | Plots filtered to movies/TV entities. |
| `data/reports/stage2_matched_movies.parquet` | 46.2 MB | MovieLens ↔ Wikipedia matches. |
| `data/reports/stage2_unmatched_wikiplots.parquet` | 47.8 MB | Plots without a MovieLens link. |
| `data/reports/stage3_checkpoint.parquet` | 160 KB | Backfill in-progress checkpoint. |
| `data/reports/stage3_failed_lookups.parquet` | 189 KB | TMDB/Wikipedia backfill misses. |
| `data/reports/stage3_short_plots.parquet` | 15 KB | Plots rejected for length. |
| `data/reports/stage4_no_plot_movies.parquet` | 1.3 MB | Final list of MovieLens movies without a plot. |
| `data/movie_plots.parquet` | 51.3 MB | **Final consolidated movie-plot table.** |
| `data/plots_metadata.parquet` | 1.9 MB | Metadata aligned with the FAISS index. |
| `data/plots.faiss` | **353.9 MB** | **Built FAISS `IndexFlatIP` over BGE-large embeddings.** |

### Model artifacts present
| Path | Size | Meaning |
|---|---|---|
| `checkpoints/mamba_phase1.pt` | 1.2 MB | Mamba checkpoint — **trained on `ml-latest-small`, not `ml-32m`.** |
| `saved/Mamba4RecFusion-Feb-15-2026_*.pth` | ~3.6 MB ×2 | RecBole-formatted training checkpoints (old "Fusion" naming, see §6). |
| `log/Mamba4RecFusion/*.log` | small | Training logs (Feb 15–17 2026). |
| `log_tensorboard/Mamba4RecFusion-*/` | small | TensorBoard scalars for the same runs. |

### Recorded training metrics (from `log/Mamba4RecFusion/...19-55-09-...log`)
- **Dataset:** `ml-latest-small` (small fixture — not the 32M target).
- **Best epoch:** 10.
- **Best valid:** hit@10 = **0.1689**, ndcg@10 = **0.0905**, mrr@10 = **0.0667**.
- **Test:** hit@10 = **0.1246**, ndcg@10 = **0.0645**, mrr@10 = **0.0463**.
- **Trainable params:** 305,152. **Frozen:** 0. **Loss:** CE.

### What did NOT succeed
- A `ml-32m` training run was attempted on **Feb 17 2026 20:45:49** (`Mamba4RecFusion-ml32m-...log`), but the log contains only the Phase-1 banner — **zero epochs completed**. Likely OOM or environment failure; needs investigation before claiming "trained on ML-32M."

---

## 4. Progress checkboxes (reconciled against actual desktop state)

Boxes are "code in repo + supporting artifact on disk." Boxes marked ⚠ deserve attention — they are checked off but with caveats.

### 4.1 Core models & retrieval
- [x] Mamba4Rec sequential model (`mamba4rec.py`) — pure-sequential, no fusion remnants
- [x] CPU fallback `MambaPureTorch` for environments without `mamba_ssm`
- [x] Single-phase training script (`train.py`) with embedding-expansion support
- [x] ContentTower over BGE-large (1024-d) FAISS `IndexFlatIP`
- [x] Query + profile blended search with α weighting
- [x] Reranker with adaptive α from query specificity
- [x] Group recommendation via fairness-weighted aggregation (mean − λ·std)
- [x] **FAISS index built and saved on disk** (`data/plots.faiss`, 354 MB; `data/plots_metadata.parquet`, 1.9 MB)
- [x] ⚠ **End-to-end training run completed** — but only on `ml-latest-small`, metrics in §3. The `ml-32m` run never produced an epoch.
- [ ] **Successful training run on MovieLens-32M with committed metrics**
- [ ] **Reproducible eval script** that loads `checkpoints/mamba_phase1.pt` + `data/plots.faiss` and reports numbers

### 4.2 LLM / chatbot layer
- [x] BGE encoder wrapper (`LLMEncoder`) with LRU-ish cache
- [x] Keyword-based `IntentParser` (mood / genre / era / constraints)
- [x] Provider-agnostic `ChatProvider` factory (OpenAI / Claude / Gemini)
- [x] Prompt-formatting helper for recommendation responses
- [ ] **Real LLM-driven intent parsing** (current parser is pure keyword — README implies LLM)
- [ ] **Multi-turn conversation state** (currently stateless)
- [ ] **LLM-generated taste summaries** (in `resume.md` "future work")

### 4.3 Embedding / profile storage
- [x] In-memory mood store with TTL
- [x] In-memory profile store with decay + adaptive LR
- [x] Redis mood store implementation
- [x] `EmbeddingManager` unified facade
- [ ] **PostgreSQL + pgvector profile store** (`resume.md` claims it; not in code)
- [ ] **Redis profile store** (only mood is Redis-backed)
- [ ] **Profile-warm-start strategy** for users with <N interactions

### 4.4 Data pipeline
- [x] Wikipedia dump download (`pipeline/download.py`) — **dump file present (25.9 GB)**
- [x] Plot section extraction with wikitext cleanup (`extract_plots.py`) — **`wiki_plots_raw.parquet` present**
- [x] Wikidata SPARQL bridge (`wikidata_bridge.py`, `filter_plots.py`) — **`wikidata_mapping.parquet` present**
- [x] MovieLens links join (`join_movielens.py`) — **`stage2_matched_movies.parquet` present**
- [x] Async Wikipedia/TMDB backfill (`backfill.py`) — **`stage3_*` reports present**
- [x] Consolidation + final reports (`consolidate.py`) — **`movie_plots.parquet` (51 MB) + reports present**
- [x] BGE encoding + FAISS index builder (`encode_plots.py`) — **`plots.faiss` + `plots_metadata.parquet` present**
- [x] Orchestrator (`run_pipeline.py`)
- [x] **End-to-end pipeline executed against the real Wikipedia dump** (artifacts above prove it)
- [ ] **Documented row counts / coverage numbers** (need to load `plots_metadata.parquet` and report — the "190K" figure in README is unverified against current outputs)
- [ ] **Resumability / incremental re-runs** beyond per-stage parquet caches
- [ ] **CI smoke test** running the pipeline on a tiny fixture dump

### 4.5 Graduation / retraining loop
- [x] `GraduationManager` with persistent queue and thresholds
- [x] Embedding-table expansion in `train.py` for adding graduated items
- [ ] **Hook `DualArmEngine.record_interaction` into `GraduationManager`** (record path covers profiles, not graduation)
- [ ] **Automated retraining trigger** (`should_retrain_by_threshold` exists; nothing calls it)
- [ ] **Item-ID remapping** when MovieLens IDs ≠ Mamba's internal item IDs after graduation
- [ ] **Evaluation comparing pre/post-graduation recall on cold-start items**

### 4.6 Serving / API
- [ ] **REST/FastAPI layer** (README claims it; no `api.py` / `server.py`)
- [ ] **Authentication / session management** wiring `session_id` through `EmbeddingManager`
- [ ] **Async I/O for content + Mamba scoring**
- [ ] **Containerization** (`resume.md` mentions Docker; no `Dockerfile`)
- [ ] **Deployment guide / production playbook**

### 4.7 Evaluation & benchmarking
- [x] Unit tests for each module under `tests/` (10 files)
- [x] Pipeline-stage unit tests under `tests/pipeline/` (6 files)
- [x] Integration test (`test_integration.py`) — content-only flow, profile build, graduation, group rec, intent parsing, α spectrum
- [ ] **Offline metrics on MovieLens-32M** (Hit@K, NDCG@K, MRR@K) — small-dataset numbers exist (§3) but ML-32M does not
- [ ] **Transformer baseline (SASRec / BERT4Rec)** for the "why Mamba" comparison in README
- [ ] **Cold-start evaluation** (recall on items absent from training)
- [ ] **Group-recommendation evaluation harness**
- [ ] **A/B-testing harness for α strategies** (resume.md future-work)

### 4.8 Documentation
- [x] README with architecture diagram (`NextUpRecommenderArchitecture.png`)
- [x] `example_config.yaml` annotated reference config
- [x] Legacy `IntegrationPlan.md` + PDFs covering the abandoned gated-fusion design
- [x] `resume.md` framing the project for job applications
- [x] `checkpoint_may_23.md` (laptop snapshot — code only, no data context)
- [x] `checkpoint_desktop.md` (this file — adds data + artifact reality)
- [ ] **Up-to-date design doc reflecting the current dual-arm architecture** (legacy IntegrationPlan is stale)
- [ ] **CHANGELOG / decision log** capturing the pivot away from gated fusion
- [ ] **Demo notebook / quickstart that runs end-to-end on a tiny fixture dataset**

---

## 5. Reality vs claims (what's true today)

| Claim | Source | Reality on this desktop |
|---|---|---|
| "190K+ Wikipedia movie plots encoded" | README | Pipeline ran end-to-end; row count needs to be re-read from `plots_metadata.parquet` and confirmed. The artifact is real; the exact number is unverified. |
| "RESTful API architecture" | README | No `api.py` / `server.py` / FastAPI code exists. |
| "Trained on MovieLens-32M" | implied by `config_ml32m.yaml` | A run was started Feb 17 and **did not produce a single epoch**. The only successful run is on `ml-latest-small`. |
| "PostgreSQL + pgvector profile storage" | `resume.md` | Not implemented — in-memory and Redis-mood only. |
| "Dockerized microservices" | `resume.md` | No `Dockerfile`. |
| "LLM-driven natural-language intent parsing" | README tone | `IntentParser` is regex/keyword matching; no LLM call. |
| Mamba checkpoint available | implied | `checkpoints/mamba_phase1.pt` exists (1.2 MB) but is from the small-dataset run. |

Decide per item: **build it**, or **edit the docs to be honest about today's state**.

---

## 6. Historical context (do not lose)

- `LLMintegrationEGG.md` + `IntegrationPlan.md/pdf` describe a **gated-fusion** design: LLM mood/intent vectors injected into Mamba's hidden space, 3-phase training (freeze Mamba → train fusion+projection → joint fine-tune). **Abandoned.**
- The folder name `fusion` and the old artifact prefix `Mamba4RecFusion-*` are relics of that direction.
- Current training is **single-phase, pure Mamba** — no fusion module, no `LLMprojection.py`.
- Anything in legacy docs referring to `LLMprojection.py`, fusion modules, or aligning LLM vectors with Mamba's `hidden_size` is **historical**.

---

## 7. "Smallest path to a credible demo"

If the goal is a one-click local demo (no Postgres, no Docker, no cloud), the gating items in order:

1. **Confirm the FAISS+metadata artifacts load correctly** — `plots.faiss` (354 MB) and `plots_metadata.parquet` are on disk; sanity-check row count, dims, and a sample query through `ContentTower`. (S)
2. **Get an `ml-32m` training run to actually complete** — find out why the Feb 17 attempt died at the banner, fix it, commit metrics. Alternatively pilot on `ml-1m` to validate the loop end-to-end. (M)
3. **Wire a thin FastAPI layer** around `DualArmEngine.recommend` and `recommend_group`. (S–M)
4. **Replace keyword `IntentParser` with an LLM call** via the existing `ChatProvider` — small change, biggest narrative-impact. (S)
5. **Write a `scripts/demo.py`** that loads checkpoint + FAISS, takes a query, prints top-K. (S)

Items 1, 3, 4, 5 are independently doable. Item 2 is the only "real ML work" required.

---

## 8. Suggested prompts for the next session

Pick the one that matches your goal:

- **Refreshed plan:** _"Read `checkpoint_desktop.md` and produce a prioritized roadmap with concrete next-step tasks. Group by 'unblock a demo' vs 'long-term polish'. For each task, name file(s) involved and rough effort (S/M/L)."_
- **Doc/code reconciliation:** _"Read `checkpoint_desktop.md` Section 5. For each row, propose either the minimal code to make the docs true or the minimal doc edit to make the docs honest, and recommend which is the better call."_
- **Evaluation focus:** _"Read `checkpoint_desktop.md`. Design an evaluation plan to back up the README's claims about dual-arm > content-only > Mamba-only. Specify datasets, metrics, baselines, and the scripts to add."_
- **Ship a demo:** _"Read `checkpoint_desktop.md` Section 7. Walk through the five gating items in order and propose the smallest possible implementation of each, with the explicit goal of a one-click local demo."_
- **Debug ML-32M training:** _"Read `checkpoint_desktop.md` §3 and §4.1. The Feb 17 ML-32M training attempt logged the banner and stopped. Investigate likely causes (memory, RecBole config, data loading) and propose a fix or smaller pilot (e.g. ML-1M)."_
- **Verify pipeline outputs:** _"Read `checkpoint_desktop.md` §3. Load `data/plots_metadata.parquet` and `data/plots.faiss` and report: row count, embedding dim, index size, a sample top-5 nearest-neighbor query, and whether the README's '190K' claim is accurate."_
