# NextUp Recommender System

A next-generation movie recommendation system that puts users back in the driver's seat. By combining cutting-edge sequential modeling with intelligent content understanding, NextUp solves the cold-start problem while delivering personalized, mood-aware recommendations through natural language interaction.

## The Problem

Traditional recommendation systems face two critical challenges:
1. **Cold-start problem**: New movies can't be recommended until users interact with them
2. **Loss of user agency**: Users are passive recipients of algorithmic predictions rather than active participants

NextUp addresses both through a novel dual-arm architecture that blends behavioral signals with semantic content understanding.

## Overview

NextUp implements a **dual-arm recommendation architecture** that intelligently combines two complementary approaches:

### Behavioral Arm (Mamba4Rec)
Learns from user interaction histories using **Mamba4Rec**, a cutting-edge sequential recommendation model based on Selective State Space Models (SSMs). Unlike Transformer-based approaches that suffer from quadratic computational complexity, Mamba4Rec efficiently handles long user behavior sequences with linear complexity—making it ideal for real-world production systems.

> **Why Mamba4Rec?** Published in March 2024 and awarded Best Paper at RelKD@KDD 2024, Mamba4Rec is the first work to apply selective SSMs to sequential recommendation. It defeats both RNN and attention-based models in effectiveness AND efficiency, especially for long interaction sequences.

### Content Arm (Semantic Search)
Uses **BAAI/bge-large-en-v1.5** embeddings and FAISS vector indexing over **190,000+ Wikipedia movie plots** to enable semantic search based on user queries like "something cozy and nostalgic from the early 2000s."

### Intelligent Blending Layer
A reranker dynamically blends behavioral and content signals based on query specificity:
- Specific mood/genre queries → Higher weight on content arm
- General browsing → Higher weight on behavioral arm
- Cold-start items → Surfaced through content arm, graduated into behavioral catalog

## Key Features

### 🎯 User-Driven Recommendations
Natural language queries for mood, genre, era, and constraints put users in control rather than forcing them to passively consume algorithmic suggestions.

### ❄️ Cold-Start Solution
New movies are immediately recommendable through semantic plot similarity, then "graduate" into the behavioral model's catalog as users interact with them.

### 👥 Group-Watch Blending
Intelligently merges multiple user profiles for shared viewing experiences—perfect for couples, families, or watch parties.

### ⚡ Production-Ready Design
- RESTful API architecture for frontend integration
- Efficient data pipelines for plot extraction and encoding
- Scalable vector storage (in-memory for dev, Redis/PostgreSQL for production)
- Modular design supporting pluggable embedding stores

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NextUp Recommendation System             │
└─────────────────────────────────────────────────────────────┘

User Query: "I want something cozy from the 90s"
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         v                  v                  v
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Intent Parser  │  │ Behavioral Arm │  │  Content Arm   │
│                │  │                │  │                │
│ Extract mood,  │  │  Mamba4Rec     │  │ BGE Embeddings │
│ genre, era     │  │  Sequential    │  │ 190K+ plots    │
│                │  │  Model         │  │ FAISS Index    │
└────────────────┘  └────────────────┘  └────────────────┘
         │                  │                  │
         │                  v                  v
         │          ┌───────────────────────────────┐
         └────────> │   Adaptive Reranker (α)       │
                    │                               │
                    │ Blends scores based on:       │
                    │ - Query specificity           │
                    │ - Item cold-start status      │
                    │ - User history depth          │
                    └───────────────────────────────┘
                                 │
                                 v
                    ┌────────────────────────┐
                    │ Top-K Recommendations  │
                    └────────────────────────┘
```

## Dataset & Scale

- **Behavioral Training**: MovieLens-32M (32 million ratings, 162,000 movies)
- **Content Search**: 190,000+ Wikipedia movie/TV plot summaries encoded with BGE-large embeddings
- **Data Pipeline**: Automated Wikidata SPARQL filtering → Wikipedia plot extraction → BGE encoding with caching

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yaml
conda activate nextup-recommender-system
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

# For data pipeline
pip install -e ".[data]"

# For production storage backends
pip install -e ".[redis]"
```

## Quick Start

### 1. Train the Behavioral Arm

```bash
python train.py --config config_ml32m.yaml --save_dir ./checkpoints
```

### 2. Build the Content Index

```bash
# Run the data pipeline to extract and encode plots
python pipeline/run_pipeline.py --output ./data/plots.parquet

# Build FAISS index
python -c "from content_tower import ContentTower; tower = ContentTower(); tower.build_index()"
```

### 3. Serve Recommendations (API)

```python
from inference import RecommendationEngine

engine = RecommendationEngine.load("./checkpoints/mamba_trained.pt")

# Single user recommendation
result = engine.recommend(
    user_id=12345,
    query="I want something cozy and nostalgic from the 90s",
    top_k=10
)

# Group-watch blending
result = engine.recommend_group(
    user_ids=[12345, 67890],
    query="family-friendly comedy",
    top_k=10
)
```

## Project Status

🚧 **In Active Development** 🚧

This project is currently under development as part of a research initiative to push the boundaries of recommendation system design. Core components implemented:

- ✅ Mamba4Rec behavioral model (stripped fusion, pure sequential)
- ✅ Wikipedia plot extraction pipeline (190K+ encoded)
- ✅ BGE-based content tower with FAISS indexing
- ✅ Intent parsing for mood/genre/era extraction
- ✅ Reranker with adaptive blending
- 🚧 API integration layer (in progress)
- 🚧 Group-watch blending (in progress)
- 🚧 Graduation mechanism (cold-start → behavioral catalog)
- 🚧 End-to-end evaluation and benchmarking

## Technical Highlights

### Why Dual-Arm Architecture?

Traditional single-model approaches force a tradeoff:
- **Collaborative filtering alone**: Can't recommend new items (cold-start)
- **Content-based alone**: Ignores personalization and user behavior patterns

The dual-arm approach gets the best of both worlds through intelligent blending.

### Why Mamba over Transformers?

| Model Type | Complexity | Long Sequences | Production Ready |
|------------|------------|----------------|------------------|
| Transformer | O(n²) | Struggles with 100+ items | High memory cost |
| Mamba4Rec | O(n) | Handles 1000+ items efficiently | Production viable |

For users with long interaction histories (power users, binge-watchers), this efficiency difference is critical.

### Adaptive Blending (α Parameter)

The reranker computes a dynamic blending weight α ∈ [0,1]:
- High specificity query ("cyberpunk thriller from 2019") → α → 0.8 (favor content)
- General query ("something good") → α → 0.3 (favor behavioral)
- Cold-start item → Boosted through content arm, flagged for graduation

## Project Structure

```
nextup-recommender-system/
├── mamba4rec.py           # Core Mamba sequential model
├── content_tower.py       # BGE embedding + FAISS search
├── reranker.py            # Adaptive blending logic
├── llm_encoder.py         # Intent parsing from natural language
├── embedding_store.py     # Pluggable storage (in-memory/Redis)
├── inference.py           # Recommendation API orchestrator
├── train.py               # Single-phase Mamba training
├── graduation.py          # Cold-start → behavioral catalog promotion
├── chat_provider.py       # LLM chat providers for conversational interface
├── pipeline/              # Data pipeline for plot extraction
│   ├── download.py        # Wikipedia dump downloader
│   ├── extract_plots.py   # Plot text extraction
│   ├── filter_plots.py    # Wikidata SPARQL filtering (movies/TV only)
│   ├── encode_plots.py    # BGE encoding with caching
│   └── join_movielens.py  # Bridge MovieLens IDs to Wikipedia
└── tests/                 # Comprehensive test suite
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
ruff check .
ruff format .
```

## Requirements

- Python 3.10+ (3.11 recommended)
- PyTorch >= 2.0.0
- RecBole >= 1.1.1
- sentence-transformers >= 2.2.0
- faiss-cpu >= 1.7.0
- mamba-ssm >= 1.0.0 (Linux + CUDA only, optional)

## Roadmap

- [ ] Complete API integration layer
- [ ] Implement group-watch profile blending
- [ ] Build graduation mechanism (content → behavioral)
- [ ] End-to-end evaluation on MovieLens-32M
- [ ] Benchmark against Transformer baselines
- [ ] Deploy demo application
- [ ] Production-ready deployment guide

## Citation

This project builds upon Mamba4Rec:

```bibtex
@article{liu2024mamba4rec,
  title={Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models},
  author={Liu, Chengkai and Lin, Jianghao and Wang, Jianling and Liu, Hanzhou and Caverlee, James},
  journal={arXiv preprint arXiv:2403.03900},
  year={2024}
}
```

## Acknowledgments

- [Mamba4Rec](https://github.com/chengkai-liu/Mamba4Rec) for the sequential recommendation foundation
- [RecBole](https://recbole.io/) for the recommendation framework
- [BAAI](https://huggingface.co/BAAI) for the BGE embedding models

---

**Note**: This is a research and development project. Code and documentation are actively evolving. Contributions and feedback welcome!
