# Dual-Arm Recommendation Architecture — Design Document

> **Purpose:** Replace the current Mamba+LLM gated fusion architecture with a cleaner dual-arm system where Mamba (behavioral) and a Content Tower (semantic) operate independently and are combined at the ranking layer. This enables cold-start handling for new movies, mood-based chatbot recommendations, group watch features, and a clear graduation path for new movies into the Mamba catalog.

---

## 1. Architecture Overview

Two independent arms, no fusion layer. Each arm does what it's best at.

```
ARM 1: Mamba (Behavioral)              ARM 2: Content Tower
┌────────────────────────┐             ┌──────────────────────┐
│ Pure Mamba4Rec          │             │ BAAI/bge-large-en-v1.5│
│ No fusion layers        │             │ 1024-dim embeddings   │
│ No LLM projection       │             │ FAISS index           │
│ Trained on ML-32M       │             │ ~130-150k movies/TV   │
│ 87k movie catalog       │             │ (filtered from wiki   │
│                         │             │  + TMDB plots)        │
└───────────┬─────────────┘             └──────────┬───────────┘
            │                                      │
            └──────────────┬───────────────────────┘
                           ▼
                    ┌──────────────┐
                    │   Reranker   │
                    │  alpha blend │
                    │  + group     │
                    └──────┬───────┘
                           ▼
                   Final recommendations
                           │
                           ▼
                   Chatbot LLM formats
                   response to user
```

### Operating Modes Per Movie

| Movie Status | Has Interactions | Has Plot | Scoring Mode |
|---|---|---|---|
| Established | Yes | Yes | Mamba + Content (reranked) |
| Established, no plot | Yes | No | Mamba only |
| New movie | No | Yes | Content only |

### What Gets Removed

- `fusion.py` — All fusion layers (PreferenceFusion, AdaptivePreferenceFusion, TemporalPreferenceFusion)
- `llm_projection.py` — LLM-to-Mamba projection no longer needed
- Phases 2 and 3 of training (alignment, EWC fine-tuning)
- All fusion logic in `mamba4rec.py` (`_apply_fusion`, `use_llm_fusion`, LLM embedding handling)

### What Stays (Modified)

- `mamba4rec.py` — stripped to pure Mamba4Rec
- `llm_encoder.py` — swapped to BAAI/bge-large-en-v1.5, simplified interface
- `embedding_store.py` — adapted for 1024-dim content profiles
- `inference.py` — rewritten as dual-arm orchestrator
- `train_phases.py` → renamed `train.py`, single-phase training + retraining support

### What's New

- `content_tower.py` — FAISS index, plot encoding, content retrieval
- `reranker.py` — score combination, implicit alpha, group recommendations
- `graduation.py` — interaction tracking, retraining triggers
- `pipeline/filter_plots.py` — filter wiki plots to movies/TV via Wikidata
- `pipeline/encode_plots.py` — BGE encoding + FAISS index building

---

## 2. Content Tower

### Embedding Model

**BAAI/bge-large-en-v1.5** (local, free, no API dependency)
- 1024-dim embeddings
- MTEB score: 64.2
- Runs locally — no external API calls for encoding
- Same model used for plot encoding AND user query encoding (must be same space)

### Plot Source and Filtering

The raw `wiki_plots_raw.parquet` (192,837 articles) contains books, video games, anime, operas, etc. Filter to movies/TV only using Wikidata SPARQL:

```sparql
SELECT ?articleTitle ?instanceLabel WHERE {
  ?article schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?articleTitle ;
           schema:about ?item .
  ?item wdt:P31 ?instance .
  VALUES ?instance {
    wd:Q11424      wd:Q24856      wd:Q5398426
    wd:Q21191270   wd:Q24862      wd:Q506240
    wd:Q1261214    wd:Q63952888   wd:Q220898
  }
}
```

Types included: film, film series, TV series, TV episode, short film, TV film, TV special, anime film, anime series.

**Output:** `wiki_plots_movies_tv.parquet` (~100-120k articles)

### Combined Plot Index

```
wiki_plots_movies_tv.parquet  (~100-120k)
  + TMDB overviews for ML-32M movies not in wiki (~35k)
  deduplicated on title
  ─────────────────
  ~130-150k unique movies/TV with plot text
```

### Metadata Prepending (Normalization)

To normalize embedding quality between long Wikipedia plots and short TMDB overviews, prepend uniform metadata to all plot texts before encoding:

```
"Film. Action, Sci-Fi. 1999. The Matrix. [plot text]"
"TV Series. Drama, Crime. 2008. Breaking Bad. [plot text]"
```

This gives every entry a consistent baseline of structured signal regardless of plot length. Short TMDB overviews become more distinctive in embedding space.

### FAISS Index

- Index type: `IndexFlatIP` (exact inner product — fast enough for <200k vectors)
- Built once offline, loaded at app startup
- New movies added at runtime via `index.add()`
- Persisted to disk as `data/plots.faiss` alongside `data/plots_metadata.parquet` (movie_id → index position mapping)

### Content Retrieval

```python
def search(
    query_embedding=None,      # from user's text query
    profile_embedding=None,    # from user's taste profile
    alpha=0.5,                 # blend weight (query vs profile)
    top_k=50,
) → List[candidate_ids]
```

- Query only: search with query embedding
- Profile only: search with profile embedding (home feed, no query)
- Both: `search_vector = alpha * query + (1 - alpha) * profile`

---

## 3. Reranker

### Score Combination

Both arms produce scores in different scales. The reranker normalizes and blends them.

**Step 1:** Gather candidates from both arms (union of top-50 each → ~100 unique).

**Step 2:** Score every candidate with both arms where possible:
- `content_score` = cosine similarity (0 to 1)
- `mamba_score` = dot product → min-max normalized to 0-1 across candidate set

**Step 3:** Blend:
```python
final_score = alpha * content_score + (1 - alpha) * mamba_score
```

For new movies (no Mamba score): `final_score = alpha * content_score`
For movies with no plot: `final_score = (1 - alpha) * mamba_score`

### Implicit Alpha from Query Specificity

Instead of a fixed alpha, derive it from how specific the user's query is:

```python
def compute_alpha(query_text, query_embedding, faiss_top1_score):
    word_count = len(query_text.split())
    distance_from_centroid = norm(query_embedding - mean_embedding)
    top_similarity = faiss_top1_score

    alpha = sigmoid(w1 * word_count + w2 * distance + w3 * top_sim + bias)
    return clamp(alpha, min=0.3, max=0.9)
```

| Situation | Alpha | Behavior |
|---|---|---|
| Home feed, no query, no profile | 0.0 | Pure Mamba |
| Home feed, has profile, no query | 0.2 | Mostly Mamba + new discoveries |
| Chatbot, vague query | ~0.4 | Balanced |
| Chatbot, specific query | ~0.8 | Trust content |

### Group Recommendations

For group watch scenarios (date night, family night), the reranker accepts multiple user contexts:

**Candidate gathering:** Union of all users' Mamba top-K + content search with shared query.

**Per-candidate scoring:** Each user scores every candidate individually (Mamba + content profile).

**Aggregation — fairness-weighted:**
```python
group_score = mean(user_scores) - lambda * std(user_scores)
```

This ensures high average satisfaction while penalizing movies that one person loves but another hates.

**Query role in group mode:** The shared text query ("something fun for the family") filters the candidate pool via FAISS. Individual profiles and Mamba scores determine which query-matching movies the group collectively agrees on.

```python
def rank_group(self, users: list, query_text=None, top_k=10):
    # Gather candidates from all users + shared query
    all_candidates = set()
    for user in users:
        all_candidates |= set(self.mamba.top_k(user.history, k=50))
        if user.has_profile:
            all_candidates |= set(self.content.search(profile=user.profile, k=30))
    if query_text:
        all_candidates |= set(self.content.search(query=query_text, k=50))

    # Score each candidate with each user
    for movie in all_candidates:
        user_scores = []
        for user in users:
            m = normalize(mamba_score(user, movie)) if movie.in_mamba else 0
            c = content_profile_sim(user.profile, movie) if movie.has_plot else 0
            score = alpha * c + (1 - alpha) * m
            user_scores.append(score)

        movie.group_score = mean(user_scores) - 0.5 * std(user_scores)

    return sorted(all_candidates, key=lambda m: -m.group_score)[:top_k]
```

---

## 4. User Content Profiles (Level 1)

Weighted average of plot embeddings from movies the user has rated/swiped on. Built from local vector math — no API calls, no privacy concerns.

```python
# Weight = rating - 3 (so 5-star → +2, 1-star → -2)
# For swipe data: right = +1, left = -1
profile = sum(weight_i * plot_embedding_i) / sum(|weight_i|)
```

Updated incrementally on each interaction via EMA:
```python
new_profile = decay * old_profile + lr * weighted_embedding
```

Profile vectors are 1024-dim (BGE space), stored in `embedding_store.py`.

### Future Extension (Level 2 — Not Implemented Now)

Feed watch history to a local LLM to generate a natural language taste summary, then encode with BGE. Richer than averaging but adds complexity. The architecture supports this without changes — the FAISS index accepts any 1024-dim query vector regardless of how it was produced.

---

## 5. Mamba Training

### Pure Mamba, Single Phase

No fusion layers, no alignment training, no EWC. Just train Mamba4Rec on interaction sequences.

### Binary Thresholding for ML-32M

MovieLens 32M uses explicit ratings (0.5-5.0). The app will collect implicit binary feedback (swipe right/left). To align formats:

```python
# Convert ML-32M ratings to binary implicit feedback
positive_interactions = ratings.filter(pl.col("rating") >= THRESHOLD)
# Threshold TBD — typically 3.5
# Only positive interactions form user sequences
```

Left swipes from the app can serve as explicit negative samples for BPR training.

### Interaction Types (UI Decision Pending)

The data layer should support multiple interaction types, as the team has not yet decided the full UI:

| Interaction | Signal | Use |
|---|---|---|
| Right swipe | Positive intent | Add to Mamba sequence |
| Left swipe | Negative intent | BPR negative sample |
| Watch confirmation | Positive consumption | Stronger positive signal |
| Post-watch rating | Graded feedback | Content profile weighting |

The system records all interaction types with timestamps. Which types feed into Mamba sequences vs content profiles vs negative sampling is configurable.

### Retraining for Graduated Movies

When retraining Mamba with graduated movies:
1. Expand the item embedding table (add rows for new movie IDs)
2. Initialize new embeddings randomly (or from content — future optimization)
3. Retrain on full dataset (ML-32M + app interactions)
4. New movies now have learned Mamba embeddings

---

## 6. Graduation Mechanism

### Interaction Counter

Track interaction count per movie. When a new movie (not in Mamba's catalog) crosses the graduation threshold, add it to the retraining queue.

```python
GRADUATION_THRESHOLD = 50  # tunable
```

### Three Trigger Modes (All Supported)

**Manual:** Operator triggers retraining via CLI or admin endpoint.
```python
graduation_manager.trigger_retrain(reason="manual")
```

**Periodic:** Scheduled retraining (daily/weekly/monthly). Cron-compatible.
```python
# Config
retrain_schedule: "weekly"  # or "daily", "monthly"
```

**Threshold:** Retrain when N movies have graduated since last training.
```python
# Config
retrain_on_graduation_count: 100  # retrain when 100+ movies queued
```

Triggers are not mutually exclusive — periodic weekly retraining AND early trigger if 100+ movies graduate in a burst.

### Retraining Flow

```
1. Export: current interaction log → training dataset
   (ML-32M + all app interactions since last training)

2. Retrain: Mamba on full dataset
   (new movies now have IDs in embedding table)

3. Update: movie catalog (in_mamba = True for graduated movies)

4. Rebuild: Mamba item embedding cache for reranker

5. Content arm: unchanged — already knows about these movies
```

### Queue Persistence

```json
// data/retraining_queue.json
{
  "pending": [
    {"movie_id": "tt9999001", "graduated_at": "2026-03-01", "interaction_count": 73},
    {"movie_id": "tt9999005", "graduated_at": "2026-03-03", "interaction_count": 58}
  ],
  "completed": [
    {"movie_id": "tt9998001", "retrained_at": "2026-02-15", "batch": "retrain_003"}
  ]
}
```

---

## 7. Chatbot LLM (Provider-Agnostic)

The chatbot LLM is a text formatter at the end of the pipeline. It receives ranked results and generates a natural response. Completely decoupled from the recommendation logic.

```python
class ChatProvider:
    def generate(self, prompt: str) -> str: ...

class OpenAIChat(ChatProvider): ...
class ClaudeChat(ChatProvider): ...
class GeminiChat(ChatProvider): ...
```

Switching providers is a config change. No retraining, no re-encoding. The prompt template is provider-agnostic:

```
The user asked: "{query}"
Based on their taste profile and history, here are the top recommendations:
1. {title} ({year}) - {genre} - {plot_snippet}
2. ...
Generate a natural, conversational response presenting these recommendations.
```

---

## 8. File Changes Summary

### Remove
| File | Reason |
|---|---|
| `fusion.py` | Fusion layers replaced by reranker |
| `llm_projection.py` | No longer projecting into Mamba space |

### Simplify
| File | Changes |
|---|---|
| `mamba4rec.py` | Strip fusion logic, phase support. Pure Mamba. ~250 lines. |
| `train_phases.py` → `train.py` | Single-phase training + retraining support. Remove Phase 2/3, EWC. |
| `llm_encoder.py` | Swap to BAAI/bge-large-en-v1.5. Simplify to encode/encode_query. Keep IntentParser. |
| `embedding_store.py` | Profile vectors → 1024-dim (BGE space). Keep mood storage, EMA updates. |
| `config.yaml` | Remove fusion config. Add content tower, graduation, alpha settings. |

### Rewrite
| File | Changes |
|---|---|
| `inference.py` | Dual-arm orchestrator. Single-user + group recommendations. |

### Create
| File | Purpose |
|---|---|
| `content_tower.py` | FAISS index management, plot encoding, content retrieval |
| `reranker.py` | Score normalization, alpha blending, group ranking |
| `graduation.py` | Interaction counting, retraining triggers (manual/periodic/threshold) |
| `pipeline/filter_plots.py` | Wikidata SPARQL filtering of wiki plots to movies/TV |
| `pipeline/encode_plots.py` | BGE batch encoding + FAISS index building with metadata prepend |
| `chat_provider.py` | Provider-agnostic chatbot LLM interface |

---

## 9. Configuration

```yaml
# Mamba (behavioral arm)
dataset: ml-32m
hidden_size: 64
num_layers: 1
dropout_prob: 0.2
d_state: 32
d_conv: 4
expand: 2
loss_type: CE
epochs: 20
train_batch_size: 2048
learning_rate: 0.001
rating_threshold: 3.5  # binary thresholding for ML-32M

# Content tower
embedding_model: BAAI/bge-large-en-v1.5
embedding_dim: 1024
faiss_index_path: data/plots.faiss
plots_metadata_path: data/plots_metadata.parquet

# Reranker
alpha_mode: implicit  # "implicit" (from query specificity) or "fixed"
alpha_fixed: 0.5      # used when alpha_mode=fixed
alpha_min: 0.3
alpha_max: 0.9
home_feed_alpha: 0.2  # alpha when no query but profile exists
group_fairness_lambda: 0.5

# Graduation
graduation_threshold: 50
retrain_trigger: all          # "manual", "periodic", "threshold", or "all"
retrain_schedule: weekly
retrain_on_graduation_count: 100
retraining_queue_path: data/retraining_queue.json

# Chatbot
chat_provider: openai         # "openai", "claude", "gemini"
chat_model: gpt-4o

# User profiles
profile_decay: 0.95
profile_learning_rate: 0.1
profile_dim: 1024
mood_ttl: 1800
```
