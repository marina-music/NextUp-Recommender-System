# NextUp Recommender System

A mood-aware recommendation system that fuses sequential behavior modeling (Mamba4Rec) with LLM-derived signals for personalized recommendations.

## Overview

This is an extended version of [Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec) that integrates LLM-based mood and intent signals for enhanced personalization.

The fusion architecture combines:
- **Mamba4Rec**: Sequential recommendation via Selective State Space Models
- **LLM Encoder**: Sentence transformer for mood/intent understanding
- **Gated Fusion**: Adaptive blending of sequential history and current mood

### Key Features

- Three-phase training for stable learning without catastrophic forgetting
- Multiple fusion architectures (gated, attention-based, temporal)
- Intent parsing from natural language (mood, genre, era, constraints)
- Pluggable embedding storage (in-memory for dev, Redis for production)
- Batch inference support for offline processing

## Installation

### Using Conda (recommended)

```bash
conda env create -f environment.yaml
conda activate mamba4rec-fusion
```

### Using UV

```bash
uv sync
```

### Using pip

```bash
pip install -e .

# For GPU support (Linux only)
pip install -e ".[gpu]"

# For production Redis storage
pip install -e ".[redis]"
```

## Quick Start

### 1. Prepare Your Dataset

Place your dataset in RecBole format under `./data/your_dataset/`:
- `your_dataset.inter` - User-item interactions (user_id, item_id, timestamp)
- `your_dataset.item` - Item features (optional)
- `your_dataset.user` - User features (optional)

The default config uses MovieLens-1M (`ml-1m`).

### 2. Train the Model

**Phased Training (Recommended)**

```bash
# Phase 1: Train vanilla Mamba4Rec
python train_phases.py --phase 1 --config config.yaml --save_dir ./checkpoints

# Phase 2: Train fusion layers (Mamba frozen)
python train_phases.py --phase 2 --config config.yaml \
  --checkpoint ./checkpoints/mamba_phase1.pt --save_dir ./checkpoints

# Phase 3: Joint fine-tuning with EWC
python train_phases.py --phase 3 --config config.yaml \
  --checkpoint ./checkpoints/mamba_phase2.pt --save_dir ./checkpoints
```

**Quick Training (Single Pass)**

```bash
python run.py                # Vanilla Mamba4Rec
python run.py --fusion       # With fusion enabled
```

### 3. Run Inference

```python
from inference import RecommendationEngine

# Load trained model
engine = RecommendationEngine.load("./checkpoints/mamba_production.pt")

# Get recommendations
result = engine.recommend(
    user_id=123,
    session_id="session-abc",
    item_history=[101, 205, 312, 89],
    mood_text="I want something cozy and nostalgic from the 90s",
    top_k=10
)

# Explain the recommendation
print(engine.explain_recommendation(result))
```

## Architecture

```
User Message --> LLM Encoder --> LLMProjection --> m_current
                                                       |
                                                       v
User History --> Mamba Layers --> seq_output --> [PreferenceFusion] --> Recommendations
                                                       ^
User Profile ----------------------------------------> p_profile
```

## Training Phases

### Phase 1: Vanilla Mamba4Rec
- Trains the sequential model without fusion
- Establishes stable item geometry in embedding space
- Uses full learning rate (0.001)

### Phase 2: Fusion Alignment
- Freezes Mamba layers and item embeddings
- Trains LLM projection and fusion layers
- Aligns LLM embeddings to item space
- Lower learning rate (0.0005)

### Phase 3: Joint Fine-tuning
- Unfreezes top Mamba layer
- Uses EWC regularization to prevent forgetting
- Very low learning rate (0.0001)

## Project Structure

```
fusion/
├── mamba4rec.py        # Core model with Mamba layers and fusion
├── fusion.py           # Fusion mechanisms (gated, attention, temporal)
├── llm_encoder.py      # Text encoding and intent parsing
├── llm_projection.py   # Projects LLM embeddings to Mamba space
├── embedding_store.py  # Storage backends for mood/profile vectors
├── train_phases.py     # Three-phase training orchestrator
├── inference.py        # Inference API and batch processing
├── run.py              # Quick training runner
├── config.yaml         # Model and training configuration
└── environment.yaml    # Conda environment specification
```

## Configuration

See `config.yaml` for all options. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 64 | Mamba hidden dimension |
| `num_layers` | 1 | Number of Mamba layers |
| `llm_dim` | 768 | LLM embedding dimension |
| `use_llm_fusion` | true | Enable LLM fusion |
| `vector_gate` | true | Per-dimension gating |
| `fusion_dropout` | 0.1 | Dropout in fusion layers |
| `loss_type` | CE | Loss function (BPR or CE) |
| `ewc_lambda` | 0.4 | EWC regularization strength |

## Supported Intents

The intent parser recognizes:

- **Moods**: cozy, exciting, scary, funny, sad, romantic, thoughtful, relaxing
- **Genres**: action, comedy, drama, horror, scifi, fantasy, thriller, romance, documentary, animation
- **Eras**: classic, 80s, 90s, 2000s, 2010s, recent
- **Constraints**: duration limits, family-friendly, mature content

## Development

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format .
```

## Requirements

- Python 3.8+ (3.11 recommended for UV)
- PyTorch >= 2.0.0
- RecBole >= 1.2.0
- sentence-transformers >= 2.2.0
- mamba-ssm >= 1.1.4 (Linux + CUDA only)
- causal-conv1d >= 1.2.0 (Linux + CUDA only)

## Citation

```bibtex
@article{liu2024mamba4rec,
  title={Mamba4rec: Towards efficient sequential recommendation with selective state space models},
  author={Liu, Chengkai and Lin, Jianghao and Wang, Jianling and Liu, Hanzhou and Caverlee, James},
  journal={arXiv preprint arXiv:2403.03900},
  year={2024}
}
```

## License

[Add your license here]
