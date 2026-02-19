# IMDb/Wikipedia Plot Data Pipeline Design

## Goal

Build a Polars-based pipeline that produces `data/movie_plots.parquet` — a clean dataset mapping MovieLens 32M movie IDs to Wikipedia plot summaries for use in ML training.

## Data Sources

### Primary: Kaggle Wikipedia Movie Plots (~35k movies)

- Source: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots
- Contains: IMDb IDs, full Wikipedia plot text, title, year, genre, director, origin
- Format: CSV download
- Key advantage: IMDb IDs are pre-linked (no matching needed)

### Linking: MovieLens 32M `links.csv`

- Source: https://grouplens.org/datasets/movielens/
- Contains: `movieId, imdbId, tmdbId`
- 87,585 movies total
- Direct join key: `imdbId`

### Backfill: Wikipedia API (for gaps)

- Wikidata SPARQL endpoint: https://query.wikidata.org/
  - Property P345 maps IMDb IDs to Wikidata items
  - Wikidata items link to Wikipedia article titles
- Wikipedia REST API: `action=parse` to extract Plot sections
- Rate limit: ~50 requests/second (polite use)

### Why not TMDB?

TMDB overviews are 1-3 sentences (30-80 words) — too short for meaningful sentence transformer embeddings. Wikipedia plots are 200-2000 words and capture mood, tone, and narrative arc. TMDB could serve as a last-resort fallback but is not a primary source.

## Pipeline Stages

### Stage 1: Load & Join

1. Load Kaggle CSV with `pl.read_csv()`
2. Load MovieLens `links.csv` with `pl.read_csv()`
3. Normalize IMDb IDs to consistent format: `tt` prefix + zero-padded 7 digits
4. Inner join on `imdbId`
5. Output: `matched_df` with `movieId, imdbId, title, year, plot_text, genres`
6. Expected match: ~25-30k movies

**Reports:**
- `data/reports/stage1_join_report.json` — total Kaggle rows, total MovieLens rows, matched count, match rate
- `data/reports/stage1_unmatched_kaggle.parquet` — Kaggle movies not in MovieLens
- `data/reports/stage1_id_mismatches.parquet` — rows where imdbId normalization changed the value

### Stage 2: Gap Identification

1. Anti-join `links.csv` against `matched_df` on `imdbId`
2. Output: `unmatched_df` — MovieLens movies with no plot text (~57-62k)

**Reports:**
- `data/reports/stage2_gap_report.json` — total unmatched count, breakdown by year/decade
- `data/reports/stage2_unmatched_movies.parquet` — full list of movies needing backfill

### Stage 3: Wikipedia Backfill (async)

1. Batch query Wikidata SPARQL to map `imdbId -> Wikipedia article title`
   - Use `VALUES` clauses, chunks of 5000 IDs per query
2. For each matched Wikipedia title, fetch the "Plot" section via `action=parse&prop=wikitext`
3. Clean wikitext markup (strip `[[links]]`, `{{templates}}`, references, etc.)
4. Use `aiohttp` with `asyncio.Semaphore(50)` for rate-limited concurrency
5. Exponential backoff on 429/5xx responses
6. Checkpoint intermediate results every 5000 movies to `data/reports/stage3_checkpoint.parquet`
7. Expected yield: ~20-30k additional plots

**Reports:**
- `data/reports/stage3_backfill_report.json` — total attempted, Wikidata matches, Wikipedia articles with Plot section, success count, failure reasons breakdown
- `data/reports/stage3_failed_lookups.parquet` — movies that failed at each stage with failure reason
- `data/reports/stage3_short_plots.parquet` — movies with fetched plot under 50 words

### Stage 4: Consolidate & Store

1. Concatenate `matched_df` + backfill results
2. Deduplicate on `movieId`
3. Add `plot_source` column: `"kaggle"` or `"wikipedia_api"`
4. Add `plot_length` column: word count
5. Write `data/movie_plots.parquet`

**Reports:**
- `data/reports/stage4_final_report.json`:
  - Total MovieLens 32M movies
  - Movies with plot text (count and percentage)
  - Breakdown by source (Kaggle vs Wikipedia API)
  - Still missing count and percentage
  - Plot length distribution: min, p25, median, p75, max (word count)
  - Per-genre match rate
  - Per-decade match rate
- `data/reports/stage4_no_plot_movies.csv` — human-readable list of all unmatched movies with: `imdbId, title, year, genres` (sorted by year descending)
- `data/reports/stage4_no_plot_movies.parquet` — same data in Parquet for programmatic use

## Output Schema

### `data/movie_plots.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `movieId` | UInt32 | MovieLens ID |
| `imdbId` | Utf8 | IMDb tconst (e.g., "tt0114709") |
| `title` | Utf8 | Movie title |
| `year` | UInt16 | Release year |
| `plot_text` | Utf8 | Full plot summary |
| `plot_source` | Utf8 | "kaggle" or "wikipedia_api" |
| `genres` | Utf8 | Pipe-delimited genres |
| `plot_length` | UInt32 | Word count |

Expected size: ~50-100MB compressed Parquet.

## Tech Stack

- **Python 3.11**
- **Polars** — all dataframe operations, CSV/Parquet I/O
- **aiohttp** — async HTTP for Wikipedia API backfill (Stage 3)
- **tqdm** — progress bars for all stages
- **json** — report file generation

## ML Loading Pattern

```python
import polars as pl

# Load all plots with sufficient length
df = (
    pl.scan_parquet("data/movie_plots.parquet")
    .filter(pl.col("plot_length") > 50)
    .collect()
)

# Join with training data by movieId for embedding generation
```

## Key Decisions

1. **Kaggle dataset as primary** — has IMDb IDs pre-linked, avoids matching complexity
2. **Wikidata as bridge for backfill** — canonical IMDb-to-Wikipedia mapping, more reliable than title fuzzy matching
3. **Async backfill with checkpointing** — handles 50-60k API calls efficiently, survives interruptions
4. **Parquet storage** — native Polars format, columnar compression, predicate pushdown for ML loading
5. **Comprehensive reporting** — every stage produces quality reports; final CSV of unmatched movies for manual inspection
