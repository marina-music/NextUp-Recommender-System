# Wikipedia Plot Data Pipeline Design

## Goal

Build a Polars-based pipeline that produces `data/movie_plots.parquet` — a clean dataset mapping MovieLens 32M movie IDs to Wikipedia plot summaries for use in ML training.

## Data Sources

### Primary: Fresh Wikipedia Dump (Feb 2026)

- Source: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
- Size: ~24GB compressed
- Contains: all English Wikipedia articles in XML/wikitext format
- We extract all articles with a "Plot" section using `mwparserfromhell`
- Expected yield: ~130k+ plots (up-to-date through Feb 2026)

### Linking: Wikidata SPARQL

- Endpoint: https://query.wikidata.org/
- Property P345: IMDb ID stored on Wikidata items
- English Wikipedia article titles are linked to Wikidata items
- Query maps: `Wikipedia article title → Wikidata item → IMDb ID`
- This is the bridge between extracted plots and MovieLens imdbIds

### Linking: MovieLens 32M

- Source: https://grouplens.org/datasets/movielens/
- `links.csv`: `movieId, imdbId, tmdbId` (87,585 movies)
- `movies.csv`: `movieId, title, genres` (title includes year, e.g., "Toy Story (1995)")
- Join key: `imdbId` (after Wikidata resolves Wikipedia titles to imdbIds)

### Backfill: Wikipedia API (for gaps)

- For MovieLens movies not matched via the dump extraction, query Wikipedia directly
- Use Wikidata SPARQL: `imdbId → Wikipedia article title`
- Then fetch Plot section via Wikipedia `action=parse` API
- Rate limit: ~50 requests/second with async concurrency

## Pipeline Stages

### Stage 1: Download & Extract Plots from Wikipedia Dump

1. Download `enwiki-latest-pages-articles-multistream.xml.bz2` (~24GB)
2. Stream-parse the XML using `mwxml` (avoids loading full dump into memory)
3. For each article, parse wikitext with `mwparserfromhell`
4. Identify sections with heading containing "Plot" (case-insensitive)
5. Extract plain text from the Plot section, strip markup
6. Output: `wiki_plots_raw.parquet` with `wiki_title, plot_text`
7. Processing time: ~2-4 hours

**Reports:**
- `data/reports/stage1_extract_report.json` — total articles scanned, articles with Plot sections found, extraction rate

### Stage 2: Build Wikidata Bridge & Join with MovieLens

1. Batch query Wikidata SPARQL to build mapping: `Wikipedia article title → IMDb ID`
   - Query all film/TV items with both an English Wikipedia sitelink and IMDb ID (P345)
   - Single bulk query returns the full mapping (~500k entries)
2. Join extracted plots with Wikidata mapping on `wiki_title`
3. Load MovieLens `links.csv` and `movies.csv`
4. Normalize IMDb IDs to consistent format: `tt` prefix + zero-padded 7 digits
5. Inner join with `links.csv` on `imdbId`
6. Enrich with title/genres from `movies.csv` via `movieId`
7. Output: `matched_df` with `movieId, imdbId, title, year, plot_text, genres`

**Reports:**
- `data/reports/stage2_bridge_report.json` — WikiPlots with Wikidata match, WikiPlots+MovieLens match, match rates at each step
- `data/reports/stage2_unmatched_wikiplots.parquet` — extracted plots with no Wikidata/IMDb match
- `data/reports/stage2_matched_movies.parquet` — all successfully matched movies

### Stage 3: Gap Identification & Wikipedia API Backfill (async)

1. Anti-join MovieLens `links.csv` against `matched_df` on `imdbId` to find gaps
2. For unmatched MovieLens movies, query Wikidata SPARQL: `imdbId → Wikipedia article title`
3. For each resolved Wikipedia title, fetch the "Plot" section via `action=parse&prop=wikitext`
4. Clean wikitext markup (strip `[[links]]`, `{{templates}}`, references, etc.)
5. Use `aiohttp` with `asyncio.Semaphore(50)` for rate-limited concurrency
6. Exponential backoff on 429/5xx responses
7. Checkpoint intermediate results every 5000 movies to `data/reports/stage3_checkpoint.parquet`

**Reports:**
- `data/reports/stage3_gap_report.json` — total unmatched count, breakdown by year/decade
- `data/reports/stage3_backfill_report.json` — total attempted, Wikidata matches, Wikipedia articles with Plot section, success count, failure reasons breakdown
- `data/reports/stage3_failed_lookups.parquet` — movies that failed at each stage with failure reason
- `data/reports/stage3_short_plots.parquet` — movies with fetched plot under 50 words

### Stage 4: Consolidate & Store

1. Concatenate `matched_df` (from dump extraction) + backfill results (from Wikipedia API)
2. Deduplicate on `movieId`
3. Add `plot_source` column: `"wiki_dump"` or `"wikipedia_api"`
4. Add `plot_length` column: word count
5. Write `data/movie_plots.parquet`

**Reports:**
- `data/reports/stage4_final_report.json`:
  - Total MovieLens 32M movies
  - Movies with plot text (count and percentage)
  - Breakdown by source (wiki_dump vs wikipedia_api)
  - Still missing count and percentage
  - Plot length distribution: min, p25, median, p75, max (word count)
  - Per-genre match rate
  - Per-decade match rate
- `data/reports/stage4_no_plot_movies.csv` — human-readable list with: `imdbId, title, year, genres` (sorted by year descending)
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
| `plot_source` | Utf8 | "wiki_dump" or "wikipedia_api" |
| `genres` | Utf8 | Pipe-delimited genres |
| `plot_length` | UInt32 | Word count |

Expected size: ~50-150MB compressed Parquet.

## Tech Stack

- **Python 3.11**
- **Polars** — all dataframe operations, CSV/Parquet I/O
- **mwxml** — streaming XML parser for Wikipedia dumps
- **mwparserfromhell** — wikitext parser for extracting clean text from markup
- **aiohttp** — async HTTP for Wikidata SPARQL and Wikipedia API backfill
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

1. **Fresh Wikipedia dump** — up-to-date through Feb 2026, largest possible coverage. All pre-built datasets are stale (2017-2018).
2. **Wikidata as bridge** — canonical Wikipedia-to-IMDb mapping via property P345. More reliable than title fuzzy matching.
3. **Async API backfill with checkpointing** — handles remaining gaps after dump extraction, survives interruptions.
4. **Parquet storage** — native Polars format, columnar compression, predicate pushdown for ML loading.
5. **Comprehensive reporting** — every stage produces quality reports; final CSV of unmatched movies for manual inspection.
