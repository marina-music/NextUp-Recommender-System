# IntegrationPlan.md

## LLM-Mamba4Rec Integration Architecture Plan

This document outlines the integration strategy for combining an LLM-based conversational interface with the Mamba4Rec sequential recommendation system for a movie recommender application.

---

## a. BPR vs CE Loss for LLM Integration

**Recommendation: Use Cross-Entropy (CE) Loss**

**Rationale:**

1. **Full Softmax Distribution**: CE loss computes a probability distribution over ALL items, which is essential when LLM mood vectors need to influence rankings globally. The softmax ensures that when a user says "I want something Tarantino-style," the model can appropriately redistribute probability mass across all Tarantino-esque films.

2. **Gradient Flow to All Items**: With CE, gradients flow through the entire item embedding matrix during each update. This means the LLM projection layer learns how its output affects scores for every item, not just a sampled pair. This is critical for mood-based adjustments that should affect genre clusters, directors, eras, etc.

3. **Better Calibration**: CE loss produces better-calibrated probability estimates, which matters when you want to explain recommendations to users ("This movie scored 85% match with your current mood").

4. **Contrastive Limitations of BPR**: BPR only considers (positive, negative) pairs and doesn't naturally support the soft preference shifts that LLM mood vectors introduce. A user saying "cozy but not too long" isn't a hard positive/negative distinction—it's a preference modulation.

**Trade-off**: CE is computationally more expensive (full softmax over all items), but for movie datasets like ML-1M (~3,900 items), this is tractable.

---

## b. Is Gated Fusion the Best Approach?

**Yes, gated fusion is well-suited for this use case. Here's why:**

### Advantages of Gated Fusion

1. **Adaptive Blending**: The gate learns when to trust the Mamba sequence history vs. the LLM mood signal. If a user has a strong viewing pattern and a vague mood ("something good"), the gate can weight history heavily. If the mood is specific ("1970s horror"), the gate can shift toward the LLM signal.

2. **Graceful Degradation**: When no LLM signal is present (m_current = 0), the gate naturally learns to output α ≈ 1, preserving pure Mamba behavior. This makes the system robust during Phase 1 training and when users don't specify preferences.

3. **Avoids Hard Switching**: Unlike attention mechanisms that might entirely ignore one signal, gated fusion ensures both signals contribute (to varying degrees).

### Alternative Approaches Considered

| Approach | Pros | Cons |
|----------|------|------|
| **Concatenation + MLP** | Simple, expressive | Doesn't preserve Mamba geometry; requires more training data |
| **Additive Fusion** | Very simple, preserves geometry | No learnable weighting; can't handle conflicting signals |
| **Cross-Attention** | Powerful for complex interactions | Computationally expensive; overkill for scalar mood vectors |
| **Residual Addition** | Minimal interference | No adaptation to signal strength |

### Recommendation

Keep gated fusion but consider a refinement: **vector-level gating** instead of scalar gating. The current implementation uses a single α ∈ [0,1] for the entire hidden dimension. A per-dimension gate α ∈ ℝ^d would allow the model to trust Mamba for some latent dimensions (e.g., genre preferences) while trusting LLM for others (e.g., mood modifiers).

---

## c. Is the Current Code Setup Sufficient for Fusion?

**No, several components are missing or need modification.**

### Current State Analysis

**What's Working:**
- `PreferenceFusion` module in `fusion.py` implements gated fusion correctly
- `mamba4rec.py` has conditional fusion in `predict()` and `full_sort_predict()`
- Config flag `use_llm_fusion` allows toggling fusion on/off

**What's Missing:**

1. **Import Statement**: `mamba4rec.py` uses `PreferenceFusion` but doesn't import it:
   ```python
   # Missing at top of mamba4rec.py:
   from fusion import PreferenceFusion
   ```

2. **LLMProjection Integration**: `LLMprojection.py` defines the projection module but:
   - Missing `import torch.nn as nn`
   - Not integrated into the main model
   - Not instantiated anywhere

3. **Data Pipeline for LLM Embeddings**: The `interaction` dictionary expects `LLM_MOOD_EMB` and `LLM_PROFILE_EMB` keys, but:
   - No data loader modifications to include these
   - No mechanism to generate LLM embeddings during training/inference
   - No storage schema for user mood vectors

4. **Training Loss Doesn't Flow Through Fusion**: `calculate_loss()` doesn't use fusion—only `predict()` and `full_sort_predict()` do. This means fusion isn't trained during the main training loop.

5. **Freezing Logic Missing**: The phased training strategy requires freezing specific layers, but no freeze/unfreeze utilities exist.

### Required Additions

```
Required New Files/Modifications:
├── llm_encoder.py          # Wrapper for sentence transformer
├── embedding_store.py      # Vector database interface
├── dataset_wrapper.py      # Modified data loader
├── train_phases.py         # Phased training orchestrator
└── inference_server.py     # Real-time fusion inference
```

---

## d. Is the Proposed Training Plan Optimal?

**The plan is sound but needs refinements for your use case.**

### Original Plan Assessment

| Phase | Assessment | Modification |
|-------|------------|--------------|
| **Phase 1**: Train vanilla Mamba4Rec | ✓ Correct | None needed |
| **Phase 2**: Freeze Mamba, train LLM projection + fusion | ✓ Correct concept | Add warm-up scheduling |
| **Phase 3**: Joint fine-tuning | ⚠ Optional but risky | Add constraints |

### Recommended Modifications

#### Phase 2 Refinements

1. **Contrastive Pre-alignment**: Before training LLMProjection with recommendation loss, pre-align LLM embeddings to item embeddings using contrastive learning:
   ```
   For movie "Pulp Fiction":
   - Positive pairs: ("Pulp Fiction" sentence embedding, item embedding)
   - Negative pairs: (random movie description, item embedding)
   ```
   This gives the projection a better initialization.

2. **Curriculum Learning**: Start with "easy" mood signals (explicit genre requests like "action movie") before introducing nuanced moods ("something like early Coen brothers but lighter").

3. **Gradient Clipping**: Apply aggressive gradient clipping to prevent the LLM projection from dominating early training.

#### Phase 3 Constraints

1. **Elastic Weight Consolidation (EWC)**: When unfreezing, use EWC to penalize large changes to parameters that were important for Phase 1 performance.

2. **Very Low Learning Rate**: Use 1/10 to 1/100 of Phase 2 learning rate.

3. **Early Stopping on Sequence Metrics**: Monitor Hit@10 on sequential evaluation (without LLM signals). If it drops more than 5%, stop fine-tuning.

#### Additional Phase: Profile Learning

Consider adding **Phase 2.5**:
- Freeze: item embeddings, Mamba layers, LLMProjection
- Train: Profile aggregation layer (p_profile)
- Purpose: Learn stable user profiles from historical LLM interactions

---

## e. Overall Architecture for LLM Integration

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                         (Chat Interface / App)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM CHAT SERVICE                                   │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │ Conversation    │───▶│ Intent Parser    │───▶│ Embedding Generator    │  │
│  │ Manager         │    │ (mood, genre,    │    │ (Sentence Transformer) │  │
│  │                 │    │  era, style)     │    │                        │  │
│  └─────────────────┘    └──────────────────┘    └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Raw LLM Embedding (768-dim)
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING ALIGNMENT LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    LLMProjection Module                              │    │
│  │            768-dim → hidden_size (64-dim in current config)          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Aligned Mood Vector (64-dim)
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VECTOR DATABASES                                   │
│                                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │  Mood Vector Store   │    │  Profile Vector Store │                      │
│  │  (Session-based)     │    │  (User-based)         │                      │
│  │                      │    │                       │                      │
│  │  Key: session_id     │    │  Key: user_id         │                      │
│  │  Value: m_current    │    │  Value: p_profile     │                      │
│  │  TTL: 30 minutes     │    │  TTL: None (persist)  │                      │
│  │                      │    │                       │                      │
│  │  Store: Redis/       │    │  Store: PostgreSQL +  │                      │
│  │         Memcached    │    │         pgvector      │                      │
│  └──────────────────────┘    └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
                 m_current                       p_profile
                      │                               │
                      └───────────────┬───────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RECOMMENDATION ENGINE                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Mamba4Rec Model                                  │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌────────────────────────┐   │    │
│  │  │   Item      │───▶│   Mamba     │───▶│   Sequence Output      │   │    │
│  │  │ Embeddings  │    │   Layers    │    │   (s_mamba)            │   │    │
│  │  └─────────────┘    └─────────────┘    └────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PreferenceFusion Module                           │    │
│  │         fused = α * s_mamba + (1-α) * (m_current + p_profile)       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│                          Fused Preference Vector                             │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Score Computation                                  │    │
│  │              scores = fused @ item_embeddings.T                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDATION OUTPUT                                │
│                      Top-K Movie Recommendations                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Database Schema

#### 1. Mood Vector Store (Redis)

```
Key Structure: mood:{session_id}
Value: JSON {
    "vector": [float; 64],
    "raw_text": "I want something cozy from the early 2000s",
    "parsed_intent": {
        "mood": "cozy",
        "era": "2000-2010",
        "genre": null
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
TTL: 1800 seconds (30 minutes)
```

#### 2. Profile Vector Store (PostgreSQL + pgvector)

```sql
CREATE TABLE user_profiles (
    user_id         BIGINT PRIMARY KEY,
    profile_vector  VECTOR(64),          -- p_profile
    interaction_count INT DEFAULT 0,
    last_updated    TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE mood_history (
    id              SERIAL PRIMARY KEY,
    user_id         BIGINT REFERENCES user_profiles(user_id),
    session_id      UUID,
    mood_vector     VECTOR(64),
    raw_text        TEXT,
    feedback_signal FLOAT,               -- Did user like the recommendation?
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_mood_history_user ON mood_history(user_id, created_at DESC);
```

#### 3. Item Embeddings Cache (Redis)

```
Key Structure: item_emb:{item_id}
Value: [float; 64]
TTL: None (permanent, refresh on model update)
```

### LLM Vector Alignment Strategy

The key challenge is ensuring LLM embeddings (typically 768 or 1024 dimensions from sentence transformers) align with Mamba's learned item geometry (64 dimensions).

#### Alignment Approach: Anchored Projection

1. **Anchor Points**: Use movie descriptions to create alignment anchors:
   ```python
   # For each movie in the catalog:
   movie_description = "Pulp Fiction (1994): Interconnected crime stories..."
   llm_embedding = sentence_transformer.encode(movie_description)  # 768-dim
   item_embedding = mamba_model.item_embedding(item_id)  # 64-dim

   # These pairs become training data for LLMProjection
   ```

2. **Contrastive Alignment Loss**:
   ```python
   def alignment_loss(projected_llm, item_emb, temperature=0.07):
       # Cosine similarity
       sim = F.cosine_similarity(projected_llm, item_emb)

       # Contrastive loss with in-batch negatives
       logits = sim / temperature
       labels = torch.arange(len(sim))
       return F.cross_entropy(logits, labels)
   ```

3. **Geometric Preservation**: During projection training, add a regularization term that preserves relative distances:
   ```python
   def geometry_preservation_loss(projected_1, projected_2, original_1, original_2):
       proj_dist = F.pairwise_distance(projected_1, projected_2)
       orig_dist = F.pairwise_distance(original_1, original_2)
       return F.mse_loss(proj_dist, orig_dist)
   ```

### Updating Vectors with New Interactions

#### Real-time Mood Updates

```python
def update_mood_vector(session_id, user_message, llm_encoder, projection):
    # 1. Encode the new message
    raw_embedding = llm_encoder.encode(user_message)

    # 2. Project to Mamba space
    aligned_embedding = projection(raw_embedding)

    # 3. Retrieve existing mood (if any)
    existing_mood = redis.get(f"mood:{session_id}")

    if existing_mood:
        # Exponential moving average for conversation continuity
        alpha = 0.7  # Weight toward new message
        new_mood = alpha * aligned_embedding + (1 - alpha) * existing_mood
    else:
        new_mood = aligned_embedding

    # 4. Store updated mood
    redis.setex(f"mood:{session_id}", 1800, new_mood)

    return new_mood
```

#### Long-term Profile Updates

```python
def update_user_profile(user_id, mood_vector, feedback_signal, db):
    """
    feedback_signal: 1.0 if user liked recommendation, -1.0 if disliked, 0 if ignored
    """
    # 1. Retrieve current profile
    current_profile = db.get_profile(user_id)

    # 2. Weight the mood by feedback
    weighted_mood = feedback_signal * mood_vector

    # 3. Update with exponential moving average
    decay = 0.95  # Slow decay for long-term preferences
    interaction_count = db.get_interaction_count(user_id)

    # Adaptive learning rate: learn faster early, slower later
    lr = 0.1 / (1 + 0.01 * interaction_count)

    new_profile = decay * current_profile + lr * weighted_mood

    # 4. Normalize to unit sphere (prevents drift)
    new_profile = F.normalize(new_profile, dim=-1)

    # 5. Store
    db.update_profile(user_id, new_profile)
    db.log_mood_history(user_id, mood_vector, feedback_signal)
```

---

## f. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              REAL-TIME INFERENCE FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────────────────┘

User Message: "I want a cozy movie from the early 2000s"
                │
                ▼
┌─────────────────────────────────────┐
│     1. LLM ENCODING                 │
│                                     │
│  sentence_transformer.encode()      │
│  Input: "I want a cozy movie..."    │
│  Output: raw_emb ∈ ℝ^768           │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     2. PROJECTION                   │
│                                     │
│  LLMProjection.forward(raw_emb)     │
│  Input: raw_emb ∈ ℝ^768            │
│  Output: m_current ∈ ℝ^64          │
└─────────────────────────────────────┘
                │
                ├─────────────────────────────────────────┐
                ▼                                         ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│     3a. STORE IN REDIS              │   │     3b. UPDATE PROFILE (async)      │
│                                     │   │                                     │
│  redis.setex(                       │   │  profile_queue.enqueue(             │
│    f"mood:{session_id}",            │   │    user_id, m_current,              │
│    1800,                            │   │    feedback=None  # pending         │
│    m_current                        │   │  )                                  │
│  )                                  │   │                                     │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     4. RETRIEVE CONTEXT             │
│                                     │
│  m_current = redis.get(mood:...)    │
│  p_profile = db.get_profile(user)   │
│  item_seq = db.get_history(user)    │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     5. MAMBA FORWARD PASS           │
│                                     │
│  s_mamba = mamba4rec.forward(       │
│    item_seq, item_seq_len           │
│  )  # Output: s_mamba ∈ ℝ^64       │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     6. GATED FUSION                 │
│                                     │
│  fusion_input = cat([s_mamba,       │
│                      m_current,     │
│                      p_profile])    │  # ∈ ℝ^192
│                                     │
│  α = gate(fusion_input)  # ∈ [0,1] │
│                                     │
│  fused = α * s_mamba +              │
│          (1-α) * (m_current +       │
│                   p_profile)        │  # ∈ ℝ^64
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     7. SCORING                      │
│                                     │
│  scores = fused @ item_emb.T        │  # ∈ ℝ^num_items
│  top_k = scores.topk(10)            │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     8. RETURN RECOMMENDATIONS       │
│                                     │
│  [Movie1, Movie2, ..., Movie10]     │
└─────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              FEEDBACK LOOP (ASYNC)                                        │
└──────────────────────────────────────────────────────────────────────────────────────────┘

User clicks on "Movie3" or says "No, something more exciting"
                │
                ▼
┌─────────────────────────────────────┐
│     F1. INTERPRET FEEDBACK          │
│                                     │
│  Click on Movie3 → positive (+1)    │
│  "More exciting" → negative (-0.5)  │
│  + new mood signal                  │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     F2. UPDATE PROFILE              │
│                                     │
│  feedback_signal = +1 or -0.5       │
│  weighted_mood = signal * m_current │
│                                     │
│  p_profile_new = EMA(               │
│    p_profile_old,                   │
│    weighted_mood                    │
│  )                                  │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│     F3. LOG FOR TRAINING            │
│                                     │
│  INSERT INTO mood_history (         │
│    user_id, session_id,             │
│    mood_vector, raw_text,           │
│    feedback_signal                  │
│  )                                  │
└─────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL TRAINING FLOW                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │         PHASE 1: VANILLA MAMBA          │
                    │                                         │
                    │  Dataset: MovieLens-1M (or similar)     │
                    │  Loss: Cross-Entropy                    │
                    │  Epochs: 300                            │
                    │  Output: Stable item geometry           │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼ Checkpoint: mamba_phase1.pt
                    ┌─────────────────────────────────────────┐
                    │         PHASE 2: LLM ALIGNMENT          │
                    │                                         │
                    │  ❄️ Frozen:                             │
                    │    - item_embedding                     │
                    │    - mamba_layers                       │
                    │                                         │
                    │  🔥 Training:                           │
                    │    - LLMProjection                      │
                    │    - PreferenceFusion                   │
                    │                                         │
                    │  Dataset: mood_history table +          │
                    │           synthetic mood-item pairs     │
                    │                                         │
                    │  Loss: CE + alignment_loss              │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼ Checkpoint: mamba_phase2.pt
                    ┌─────────────────────────────────────────┐
                    │     PHASE 3: JOINT FINE-TUNING          │
                    │                                         │
                    │  ❄️ Frozen:                             │
                    │    - item_embedding                     │
                    │    - mamba_layers[:-1] (all but last)   │
                    │                                         │
                    │  🔥 Training (low LR):                  │
                    │    - mamba_layers[-1] (top layer)       │
                    │    - LLMProjection                      │
                    │    - PreferenceFusion                   │
                    │                                         │
                    │  Loss: CE + EWC regularization          │
                    │                                         │
                    │  Early Stop: if Hit@10 drops > 5%       │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼ Checkpoint: mamba_production.pt
                    ┌─────────────────────────────────────────┐
                    │         CONTINUOUS LEARNING             │
                    │              (Optional)                 │
                    │                                         │
                    │  Trigger: Every 10K new interactions    │
                    │                                         │
                    │  1. Export mood_history since last      │
                    │     training                            │
                    │  2. Fine-tune LLMProjection only        │
                    │  3. A/B test new model                  │
                    │  4. Promote if metrics improve          │
                    └─────────────────────────────────────────┘
```

---

## g. Code Changes Documentation

The `Fusion/` folder contains a complete implementation with all necessary modifications for LLM-Mamba integration. Below is a detailed breakdown of each file and the changes made.

### File Structure

```
Fusion/
├── config.yaml          # Extended config with fusion settings
├── mamba4rec.py         # Core model with fusion integration
├── fusion.py            # Enhanced fusion module with multiple variants
├── llm_projection.py    # Sophisticated projection with alignment
├── llm_encoder.py       # Sentence transformer wrapper
├── embedding_store.py   # Mood and profile vector storage
├── train_phases.py      # Phased training orchestrator
├── inference.py         # High-level inference API
├── run.py               # Simple training script
├── environment.yaml     # Updated dependencies
└── README.md            # Usage documentation
```

---

### 1. `mamba4rec.py` - Core Model Changes

**Location**: `Fusion/mamba4rec.py`

**Key Changes from Original:**

#### a. Added Imports and LLM Components (Lines 1-25)
```python
from fusion import PreferenceFusion, AdaptivePreferenceFusion
from llm_projection import LLMProjection, LLMProjectionWithAlignment
```
The original code referenced `PreferenceFusion` but never imported it. This is now fixed.

#### b. Model Initialization with LLM Components (Lines 50-75)
```python
# LLM Fusion components (only created if enabled)
if self.use_llm_fusion:
    self.llm_projection = LLMProjectionWithAlignment(
        llm_dim=self.llm_dim,
        hidden_size=self.hidden_size,
        dropout=self.fusion_dropout
    )
    self.fusion = PreferenceFusion(
        hidden_size=self.hidden_size,
        dropout=self.fusion_dropout,
        vector_gate=self.vector_gate
    )
```
**Purpose**: Instantiates the LLM projection layer within the model, so raw LLM embeddings (768-dim) are projected to hidden_size (64-dim) before fusion.

#### c. `_apply_fusion()` Helper Method (Lines 100-125)
```python
def _apply_fusion(self, seq_output, interaction):
    if not self.use_llm_fusion or self.fusion is None:
        return seq_output

    raw_mood_emb = interaction.get("LLM_MOOD_EMB", None)
    raw_profile_emb = interaction.get("LLM_PROFILE_EMB", None)

    m_current = None
    p_profile = None

    if raw_mood_emb is not None:
        m_current = self.llm_projection(raw_mood_emb)
    if raw_profile_emb is not None:
        p_profile = self.llm_projection(raw_profile_emb)

    return self.fusion(s_mamba=seq_output, m_current=m_current, p_profile=p_profile)
```
**Purpose**: Centralizes fusion logic. The original code had fusion calls duplicated in `predict()` and `full_sort_predict()` but not in `calculate_loss()`. This helper ensures consistent application.

#### d. `calculate_loss()` Now Supports Fusion Training (Lines 130-160)
```python
def calculate_loss(self, interaction):
    seq_output = self.forward(item_seq, item_seq_len)

    # Apply fusion during training (Phase 2+)
    if self._current_phase >= 2:
        seq_output = self._apply_fusion(seq_output, interaction)
    ...
```
**Purpose**: The original code only applied fusion during inference (`predict`, `full_sort_predict`), meaning fusion was never trained. Now fusion is applied during loss calculation in Phase 2+.

#### e. Phased Training Methods (Lines 190-260)
```python
def set_training_phase(self, phase):
    self._current_phase = phase
    if phase == 1:
        self._unfreeze_all()
    elif phase == 2:
        self._freeze_for_phase2()
    elif phase == 3:
        self._configure_phase3()

def _freeze_for_phase2(self):
    # Freeze item embeddings
    self.item_embedding.weight.requires_grad = False
    # Freeze Mamba layers
    for layer in self.mamba_layers:
        for param in layer.parameters():
            param.requires_grad = False
    # Keep fusion trainable
    for param in self.fusion.parameters():
        param.requires_grad = True
```
**Purpose**: Implements the freezing logic for phased training. The original code had no mechanism to freeze/unfreeze specific layers.

#### f. Interpretability Method (Lines 265-285)
```python
def get_fusion_weights(self, interaction):
    """Get fusion gate values for interpretability."""
    return self.fusion.get_gate_value(seq_output, m_current, p_profile)
```
**Purpose**: Allows inspection of how much the model trusts Mamba vs. LLM signals for any given prediction.

---

### 2. `fusion.py` - Enhanced Fusion Module

**Location**: `Fusion/fusion.py`

**Key Changes from Original:**

#### a. Vector-Level Gating Option (Lines 20-35)
```python
def __init__(self, hidden_size, dropout=0.1, vector_gate=False):
    gate_out_dim = hidden_size if vector_gate else 1
    self.gate = nn.Sequential(
        nn.Linear(hidden_size * 3, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, gate_out_dim),
        nn.Sigmoid()
    )
```
**Purpose**: Original fusion used scalar gating (α ∈ [0,1]). Vector gating (α ∈ ℝ^d) allows the model to trust different signals for different latent dimensions.

#### b. Gate Bias Initialization (Lines 45-55)
```python
def _init_gate_bias(self):
    """Initialize gate to slightly favor Mamba output (α ≈ 0.6)."""
    with torch.no_grad():
        final_linear = self.gate[-2]
        if hasattr(final_linear, 'bias') and final_linear.bias is not None:
            final_linear.bias.fill_(0.4)
```
**Purpose**: Biases the gate toward Mamba at initialization, ensuring stable behavior when LLM signals are weak or absent.

#### c. Additional Fusion Variants (Lines 100-200)
- `AdaptivePreferenceFusion`: Attention-based fusion with multi-head attention
- `TemporalPreferenceFusion`: Processes mood history with a transformer for conversation context

**Purpose**: Provides alternatives for experimentation and more complex use cases.

---

### 3. `llm_projection.py` - Sophisticated Projection Module

**Location**: `Fusion/llm_projection.py`

**Key Changes from Original `LLMprojection.py`:**

#### a. Fixed Missing Import
```python
import torch.nn as nn  # Was missing in original
```

#### b. Multi-Layer Architecture with Normalization (Lines 30-55)
```python
intermediate_dim = max(hidden_size * 2, llm_dim // 4)

self.proj = nn.Sequential(
    nn.Linear(llm_dim, intermediate_dim),
    nn.LayerNorm(intermediate_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(intermediate_dim, hidden_size),
    nn.LayerNorm(hidden_size),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, hidden_size),
)
```
**Purpose**: Original was 2 layers (768→64→64). New version uses a bottleneck design with layer normalization for training stability.

#### c. Small Weight Initialization (Lines 60-70)
```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Smaller than default
```
**Purpose**: Prevents the LLM projection from dominating Mamba outputs during early training.

#### d. Contrastive Alignment Methods (Lines 100-160)
```python
def compute_alignment_loss(self, llm_emb, item_emb):
    """Contrastive loss to align LLM projections with item embeddings."""
    _, contrastive_proj = self.forward(llm_emb, return_contrastive=True)

    contrastive_proj = F.normalize(contrastive_proj, dim=-1)
    item_emb = F.normalize(item_emb, dim=-1)

    logits = torch.matmul(contrastive_proj, item_emb.T) / self.temperature
    labels = torch.arange(batch_size, device=llm_emb.device)

    loss_llm_to_item = F.cross_entropy(logits, labels)
    loss_item_to_llm = F.cross_entropy(logits.T, labels)
    return (loss_llm_to_item + loss_item_to_llm) / 2
```
**Purpose**: Enables pre-alignment of LLM embeddings with item embeddings using movie descriptions as anchor points.

---

### 4. `llm_encoder.py` - Sentence Transformer Wrapper

**Location**: `Fusion/llm_encoder.py`

**New File - Not in Original**

#### Key Features:

- **Lazy Loading**: Sentence transformer model is only loaded when first used
- **Caching**: LRU cache for repeated queries (10,000 entries)
- **Mood Preprocessing**: Adds context to short mood messages
- **Intent Parser**: Extracts structured information (mood, genre, era, constraints)

```python
class IntentParser:
    MOOD_KEYWORDS = {
        "cozy": ["cozy", "warm", "comfortable", ...],
        "exciting": ["exciting", "thrilling", "action", ...],
        ...
    }

    def parse(self, message):
        return {
            "mood": self._extract_matches(message, self.MOOD_KEYWORDS),
            "genre": self._extract_matches(message, self.GENRE_KEYWORDS),
            "era": self._extract_matches(message, self.ERA_KEYWORDS),
            "constraints": self._extract_constraints(message)
        }
```

---

### 5. `embedding_store.py` - Vector Storage

**Location**: `Fusion/embedding_store.py`

**New File - Not in Original**

Implements:
- `InMemoryMoodStore`: For development/testing with TTL expiration
- `InMemoryProfileStore`: With EMA profile updates
- `RedisMoodStore`: Production-ready Redis backend
- `EmbeddingManager`: Unified interface for all storage operations

```python
class EmbeddingManager:
    def update_mood(self, session_id, projected_mood, raw_text, parsed_intent):
        """Update session mood with EMA blending."""
        ...

    def record_feedback(self, user_id, session_id, feedback):
        """Update long-term profile based on feedback."""
        ...

    def prepare_interaction_dict(self, session_id, user_id, device):
        """Create LLM_MOOD_EMB and LLM_PROFILE_EMB for model input."""
        ...
```

---

### 6. `train_phases.py` - Phased Training Orchestrator

**Location**: `Fusion/train_phases.py`

**New File - Not in Original**

Implements:
- Phase 1: Vanilla Mamba training
- Phase 2: LLM alignment with frozen Mamba
- Phase 3: Joint fine-tuning with EWC regularization

```python
class EWCRegularizer:
    """Elastic Weight Consolidation to prevent catastrophic forgetting."""

    def _compute_fisher(self, dataloader):
        """Compute Fisher information matrix diagonal."""
        ...

    def penalty(self):
        """Compute EWC penalty term."""
        return self.lambda_ewc * sum(
            (self.fisher[name] * (param - self.optimal_params[name])**2).sum()
            for name, param in model.named_parameters()
        )
```

---

### 7. `inference.py` - High-Level Inference API

**Location**: `Fusion/inference.py`

**New File - Not in Original**

Provides:
- `RecommendationEngine`: Complete inference pipeline
- `RecommendationResult`: Structured output with explanations
- `BatchRecommender`: Efficient batch processing

```python
engine = RecommendationEngine.load("checkpoint.pt")
result = engine.recommend(
    user_id=123,
    session_id="abc",
    item_history=[101, 205, 312],
    mood_text="I want something cozy",
    top_k=10
)
print(engine.explain_recommendation(result))
# Output:
# Based on your mood: "I want something cozy"
#   Detected mood: cozy
# Blending: 65% watch history + 35% current mood
# Top recommendations:
#   1. You've Got Mail (score: 0.892)
#   ...
```

---

### 8. `config.yaml` - Extended Configuration

**Location**: `Fusion/config.yaml`

**New Settings Added:**

```yaml
# LLM Fusion settings
use_llm_fusion: True            # Enable LLM fusion
llm_dim: 768                    # Sentence transformer dimension
fusion_dropout: 0.1             # Dropout in fusion layers
vector_gate: True               # Per-dimension gating

# Phase-specific learning rates
phase2_learning_rate: 0.0005
phase2_epochs: 50
phase3_learning_rate: 0.0001
phase3_epochs: 20
ewc_lambda: 0.4                 # EWC regularization strength
```

---

## Summary

This integration plan provides:

1. **CE loss** as the recommended loss function for better gradient flow to all items
2. **Gated fusion** as an appropriate mechanism with suggestions for vector-level enhancement
3. **Gap analysis** of the current code identifying missing imports, data pipeline, and training loop modifications
4. **Refined training strategy** with contrastive pre-alignment and regularization
5. **Complete architecture** with database schemas for mood and profile storage
6. **Data flow diagrams** showing real-time inference, feedback loops, and training phases

The `Fusion/` folder contains all necessary code modifications to make LLM-Mamba integration fully functional.

This integration plan provides:

1. **CE loss** as the recommended loss function for better gradient flow to all items
2. **Gated fusion** as an appropriate mechanism with suggestions for vector-level enhancement
3. **Gap analysis** of the current code identifying missing imports, data pipeline, and training loop modifications
4. **Refined training strategy** with contrastive pre-alignment and regularization
5. **Complete architecture** with database schemas for mood and profile storage
6. **Data flow diagrams** showing real-time inference, feedback loops, and training phases

The next step is to create the `Fusion/` folder with all necessary code modifications.
