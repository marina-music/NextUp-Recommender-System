# Wikipedia Plot Data Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Polars pipeline that extracts movie plots from a fresh Wikipedia dump, links them to MovieLens 32M via Wikidata, and produces `data/movie_plots.parquet`.

**Architecture:** Stream-parse a Wikipedia XML dump to extract Plot sections, use Wikidata SPARQL to bridge Wikipedia titles to IMDb IDs, join with MovieLens, async-backfill gaps via Wikipedia API, consolidate into a single Parquet file with quality reports at every stage.

**Tech Stack:** Python 3.11, Polars, mwxml, mwparserfromhell, aiohttp, tqdm

---

### Task 1: Add pipeline dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add a `data` optional dependency group**

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
data = [
    "polars>=1.0.0",
    "mwxml>=0.3.3",
    "mwparserfromhell>=0.6.5",
    "aiohttp>=3.9.0",
    "tqdm>=4.65.0",
]
```

**Step 2: Install the new dependencies**

Run: `uv sync --extra data`
Expected: all packages install successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add data pipeline dependencies (polars, mwxml, mwparserfromhell, aiohttp)"
```

---

### Task 2: Create directory structure and download script

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/download.py`

**Step 1: Create the pipeline package**

Create `pipeline/__init__.py` (empty file).

**Step 2: Write the download script**

Create `pipeline/download.py`:

```python
"""Download Wikipedia dump and MovieLens 32M dataset."""

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

WIKI_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = DATA_DIR / "reports"


class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path) -> Path:
    """Download a file with progress bar. Skips if file exists."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    return dest


def download_wiki_dump() -> Path:
    """Download the latest Wikipedia dump (~24GB)."""
    dest = RAW_DIR / "enwiki-latest-pages-articles-multistream.xml.bz2"
    return download_file(WIKI_DUMP_URL, dest)


def download_movielens() -> Path:
    """Download and extract MovieLens 32M dataset."""
    zip_path = RAW_DIR / "ml-32m.zip"
    extracted_dir = RAW_DIR / "ml-32m"
    if extracted_dir.exists():
        print(f"  Already extracted: {extracted_dir}")
        return extracted_dir
    download_file(MOVIELENS_URL, zip_path)
    print("  Extracting MovieLens 32M...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)
    return extracted_dir


def ensure_dirs():
    """Create all required directories."""
    for d in [RAW_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()
    print("Downloading Wikipedia dump...")
    wiki_path = download_wiki_dump()
    print(f"  Done: {wiki_path} ({wiki_path.stat().st_size / 1e9:.1f} GB)")
    print("Downloading MovieLens 32M...")
    ml_path = download_movielens()
    print(f"  Done: {ml_path}")


if __name__ == "__main__":
    main()
```

**Step 3: Test the download script starts correctly**

Run: `python -c "from pipeline.download import ensure_dirs, DATA_DIR; ensure_dirs(); print(DATA_DIR)"`
Expected: prints `data` and creates `data/raw/` and `data/reports/`

**Step 4: Commit**

```bash
git add pipeline/
git commit -m "feat: add download script for Wikipedia dump and MovieLens 32M"
```

---

### Task 3: Stage 1 — Wikipedia dump Plot extractor

**Files:**
- Create: `pipeline/extract_plots.py`
- Test: `tests/pipeline/test_extract_plots.py`

**Step 1: Write tests for wikitext cleaning and Plot section extraction**

Create `tests/__init__.py` (empty) and `tests/pipeline/__init__.py` (empty).

Create `tests/pipeline/test_extract_plots.py`:

```python
"""Tests for Wikipedia dump plot extraction."""

from pipeline.extract_plots import clean_wikitext, extract_plot_section


class TestCleanWikitext:
    def test_strips_links(self):
        assert clean_wikitext("[[John Smith|John]]") == "John"

    def test_strips_bare_links(self):
        assert clean_wikitext("[[London]]") == "London"

    def test_strips_templates(self):
        assert clean_wikitext("Hello {{citation needed}} world") == "Hello  world"

    def test_strips_references(self):
        assert clean_wikitext('Text<ref name="x">cite</ref> more') == "Text more"

    def test_strips_self_closing_refs(self):
        assert clean_wikitext('Text<ref name="x" /> more') == "Text more"

    def test_preserves_plain_text(self):
        assert clean_wikitext("A simple sentence.") == "A simple sentence."

    def test_strips_bold_italic(self):
        assert clean_wikitext("'''bold''' and ''italic''") == "bold and italic"


class TestExtractPlotSection:
    def test_extracts_plot(self):
        wikitext = """== Cast ==
Some cast info.

== Plot ==
The movie begins with a chase scene.
The hero escapes.

== Production ==
Filming took place in London.
"""
        result = extract_plot_section(wikitext)
        assert result is not None
        assert "chase scene" in result
        assert "hero escapes" in result
        assert "Filming" not in result

    def test_returns_none_when_no_plot(self):
        wikitext = """== Cast ==
Some cast info.

== Production ==
Filming took place.
"""
        assert extract_plot_section(wikitext) is None

    def test_handles_plot_summary_heading(self):
        wikitext = """== Plot summary ==
A brief plot.

== Cast ==
Actors.
"""
        result = extract_plot_section(wikitext)
        assert result is not None
        assert "brief plot" in result

    def test_handles_synopsis_heading(self):
        wikitext = """== Synopsis ==
A synopsis of the film.

== Cast ==
Actors.
"""
        result = extract_plot_section(wikitext)
        assert result is not None
        assert "synopsis" in result
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_extract_plots.py -v`
Expected: FAIL — `ImportError: cannot import name 'clean_wikitext' from 'pipeline.extract_plots'`

**Step 3: Write the extraction module**

Create `pipeline/extract_plots.py`:

```python
"""Extract Plot sections from a Wikipedia XML dump.

Streams through the dump using mwxml, parses wikitext with
mwparserfromhell, and extracts clean text from Plot sections.
"""

import bz2
import json
import re
from pathlib import Path

import mwparserfromhell
import mwxml
import polars as pl
from tqdm import tqdm

from pipeline.download import DATA_DIR, REPORTS_DIR

PLOT_HEADINGS = {"plot", "plot summary", "synopsis"}

# Regex patterns for cleanup
REF_PATTERN = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
REF_SELF_CLOSE = re.compile(r"<ref[^/]*/\s*>")
HTML_TAG = re.compile(r"<[^>]+>")


def clean_wikitext(text: str) -> str:
    """Clean wikitext markup to plain text."""
    # Remove references
    text = REF_PATTERN.sub("", text)
    text = REF_SELF_CLOSE.sub("", text)
    # Parse remaining wikitext
    parsed = mwparserfromhell.parse(text)
    # Strip templates
    for template in parsed.filter_templates():
        try:
            parsed.remove(template)
        except ValueError:
            pass
    # Convert wikilinks to plain text
    for link in parsed.filter_wikilinks():
        display = link.text if link.text else link.title
        try:
            parsed.replace(link, str(display))
        except ValueError:
            pass
    result = str(parsed)
    # Remove remaining HTML tags
    result = HTML_TAG.sub("", result)
    # Remove bold/italic markup
    result = result.replace("'''", "").replace("''", "")
    return result


def extract_plot_section(wikitext: str) -> str | None:
    """Extract the Plot section from article wikitext.

    Looks for sections headed 'Plot', 'Plot summary', or 'Synopsis'.
    Returns cleaned plain text, or None if no plot section found.
    """
    parsed = mwparserfromhell.parse(wikitext)
    sections = parsed.get_sections(levels=[2])
    for section in sections:
        headings = section.filter_headings()
        if not headings:
            continue
        heading_text = headings[0].title.strip().lower()
        if heading_text in PLOT_HEADINGS:
            # Remove the heading itself
            content = str(section)
            # Strip the == Heading == line
            lines = content.split("\n")
            body_lines = [l for l in lines[1:] if not l.strip().startswith("==")]
            body = "\n".join(body_lines).strip()
            if not body:
                return None
            return clean_wikitext(body)
    return None


def extract_plots_from_dump(dump_path: Path) -> pl.DataFrame:
    """Stream-parse Wikipedia dump and extract all Plot sections.

    Args:
        dump_path: Path to enwiki-*-pages-articles-multistream.xml.bz2

    Returns:
        DataFrame with columns: wiki_title (Utf8), plot_text (Utf8)
    """
    titles = []
    plots = []
    articles_scanned = 0

    dump = mwxml.Dump.from_file(bz2.open(dump_path, "rb"))

    for page in tqdm(dump, desc="Scanning Wikipedia dump", unit=" pages"):
        # Skip non-article namespaces
        if page.namespace != 0:
            continue
        for revision in page:
            articles_scanned += 1
            if revision.text is None:
                continue
            plot = extract_plot_section(revision.text)
            if plot and len(plot.split()) >= 20:
                titles.append(page.title)
                plots.append(plot)
            break  # Only process latest revision

    df = pl.DataFrame({
        "wiki_title": titles,
        "plot_text": plots,
    })

    # Write report
    report = {
        "articles_scanned": articles_scanned,
        "plots_extracted": len(plots),
        "extraction_rate": f"{len(plots) / max(articles_scanned, 1) * 100:.2f}%",
    }
    report_path = REPORTS_DIR / "stage1_extract_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Stage 1 report: {report}")

    return df


def main():
    from pipeline.download import RAW_DIR
    dump_path = RAW_DIR / "enwiki-latest-pages-articles-multistream.xml.bz2"
    if not dump_path.exists():
        print(f"Wikipedia dump not found at {dump_path}")
        print("Run: python -m pipeline.download")
        return

    df = extract_plots_from_dump(dump_path)
    out_path = DATA_DIR / "wiki_plots_raw.parquet"
    df.write_parquet(out_path)
    print(f"Wrote {len(df)} plots to {out_path}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_extract_plots.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add pipeline/extract_plots.py tests/
git commit -m "feat: add Wikipedia dump Plot section extractor with tests"
```

---

### Task 4: Stage 2 — Wikidata bridge and MovieLens join

**Files:**
- Create: `pipeline/wikidata_bridge.py`
- Create: `pipeline/join_movielens.py`
- Test: `tests/pipeline/test_join_movielens.py`

**Step 1: Write tests for IMDb ID normalization and the join logic**

Create `tests/pipeline/test_join_movielens.py`:

```python
"""Tests for MovieLens joining logic."""

import polars as pl
from pipeline.join_movielens import normalize_imdb_id, join_with_movielens


class TestNormalizeImdbId:
    def test_already_normalized(self):
        assert normalize_imdb_id("tt0114709") == "tt0114709"

    def test_adds_tt_prefix(self):
        assert normalize_imdb_id("0114709") == "tt0114709"

    def test_zero_pads(self):
        assert normalize_imdb_id("tt114709") == "tt0114709"

    def test_bare_number(self):
        assert normalize_imdb_id("114709") == "tt0114709"

    def test_short_number(self):
        assert normalize_imdb_id("12345") == "tt0012345"


class TestJoinWithMovielens:
    def test_inner_join_on_imdb_id(self):
        wikiplots = pl.DataFrame({
            "wiki_title": ["Toy Story", "Unknown Film"],
            "imdbId": ["tt0114709", "tt9999999"],
            "plot_text": ["Woody is a toy.", "Unknown plot."],
        })
        links = pl.DataFrame({
            "movieId": [1, 2],
            "imdbId": ["tt0114709", "tt0113497"],
            "tmdbId": [862, 680],
        })
        movies = pl.DataFrame({
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Jumanji (1995)"],
            "genres": ["Animation|Children|Comedy", "Adventure|Children|Fantasy"],
        })
        result = join_with_movielens(wikiplots, links, movies)
        assert len(result) == 1
        row = result.row(0, named=True)
        assert row["movieId"] == 1
        assert row["imdbId"] == "tt0114709"
        assert row["title"] == "Toy Story"
        assert row["year"] == 1995
        assert "Woody" in row["plot_text"]
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_join_movielens.py -v`
Expected: FAIL — ImportError

**Step 3: Write the Wikidata bridge module**

Create `pipeline/wikidata_bridge.py`:

```python
"""Query Wikidata SPARQL to map Wikipedia article titles to IMDb IDs."""

import json
import time
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# This query fetches ALL English Wikipedia articles that have an IMDb ID.
# Returns ~500k rows. Run once and cache.
BULK_QUERY = """
SELECT ?articleTitle ?imdbId WHERE {
  ?item wdt:P345 ?imdbId .
  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?articleTitle .
}
"""


def fetch_wikidata_mapping(cache_path: Path | None = None) -> pl.DataFrame:
    """Fetch the full Wikipedia title → IMDb ID mapping from Wikidata.

    This is a single large SPARQL query. Results are cached to disk.

    Args:
        cache_path: Path to cache the results as parquet.

    Returns:
        DataFrame with columns: wiki_title (Utf8), imdbId (Utf8)
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached Wikidata mapping from {cache_path}")
        return pl.read_parquet(cache_path)

    print("  Querying Wikidata SPARQL for Wikipedia → IMDb mapping...")
    print("  (This may take 1-3 minutes for ~500k results)")

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "Mamba4RecFusion/1.0 (https://github.com/markriedl/WikiPlots; research)",
    }
    params = {"query": BULK_QUERY}

    retries = 3
    for attempt in range(retries):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL, params=params, headers=headers, timeout=300
            )
            resp.raise_for_status()
            break
        except (requests.RequestException, requests.Timeout) as e:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise

    data = resp.json()
    bindings = data["results"]["bindings"]

    titles = [b["articleTitle"]["value"] for b in bindings]
    imdb_ids = [b["imdbId"]["value"] for b in bindings]

    df = pl.DataFrame({"wiki_title": titles, "imdbId": imdb_ids})

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        print(f"  Cached {len(df)} mappings to {cache_path}")

    return df
```

**Step 4: Write the MovieLens join module**

Create `pipeline/join_movielens.py`:

```python
"""Join WikiPlots data with MovieLens 32M via Wikidata bridge."""

import json
import re
from pathlib import Path

import polars as pl

from pipeline.download import DATA_DIR, REPORTS_DIR


def normalize_imdb_id(imdb_id: str) -> str:
    """Normalize an IMDb ID to 'tt' + 7-digit zero-padded format."""
    # Strip 'tt' prefix if present
    numeric = imdb_id.lstrip("t")
    # Zero-pad to 7 digits and add prefix
    return f"tt{numeric.zfill(7)}"


def parse_title_year(title: str) -> tuple[str, int | None]:
    """Parse MovieLens title format 'Movie Name (1995)' into (name, year)."""
    match = re.match(r"^(.+?)\s*\((\d{4})\)\s*$", title)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title.strip(), None


def join_with_movielens(
    wikiplots_with_imdb: pl.DataFrame,
    links_df: pl.DataFrame,
    movies_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join WikiPlots (with imdbId) against MovieLens links and movies.

    Args:
        wikiplots_with_imdb: DataFrame with wiki_title, imdbId, plot_text
        links_df: MovieLens links.csv as DataFrame (movieId, imdbId, tmdbId)
        movies_df: MovieLens movies.csv as DataFrame (movieId, title, genres)

    Returns:
        Joined DataFrame with movieId, imdbId, title, year, plot_text, genres
    """
    # Normalize IMDb IDs in both datasets
    wikiplots_norm = wikiplots_with_imdb.with_columns(
        pl.col("imdbId").map_elements(normalize_imdb_id, return_dtype=pl.Utf8).alias("imdbId")
    )
    links_norm = links_df.with_columns(
        pl.col("imdbId").cast(pl.Utf8).map_elements(normalize_imdb_id, return_dtype=pl.Utf8).alias("imdbId")
    )

    # Inner join on imdbId
    joined = wikiplots_norm.join(links_norm.select(["movieId", "imdbId"]), on="imdbId", how="inner")

    # Parse title and year from movies.csv
    movies_parsed = movies_df.with_columns([
        pl.col("title").map_elements(lambda t: parse_title_year(t)[0], return_dtype=pl.Utf8).alias("parsed_title"),
        pl.col("title").map_elements(lambda t: parse_title_year(t)[1], return_dtype=pl.UInt16).alias("year"),
    ])

    # Enrich with title, year, genres
    result = joined.join(
        movies_parsed.select(["movieId", "parsed_title", "year", "genres"]),
        on="movieId",
        how="left",
    ).rename({"parsed_title": "title"}).select([
        "movieId", "imdbId", "title", "year", "plot_text", "genres",
    ])

    return result


def run_stage2(
    wiki_plots_raw: pl.DataFrame,
    wikidata_mapping: pl.DataFrame,
    movielens_dir: Path,
) -> pl.DataFrame:
    """Execute Stage 2: bridge WikiPlots to MovieLens.

    Args:
        wiki_plots_raw: Output of Stage 1 (wiki_title, plot_text)
        wikidata_mapping: Wikidata bridge (wiki_title, imdbId)
        movielens_dir: Path to extracted MovieLens 32M directory

    Returns:
        Matched DataFrame ready for Stage 4
    """
    # Join WikiPlots with Wikidata mapping
    wikiplots_with_imdb = wiki_plots_raw.join(wikidata_mapping, on="wiki_title", how="inner")

    # Load MovieLens files
    links_df = pl.read_csv(movielens_dir / "links.csv")
    movies_df = pl.read_csv(movielens_dir / "movies.csv")

    # Join with MovieLens
    matched = join_with_movielens(wikiplots_with_imdb, links_df, movies_df)

    # Reports
    wikidata_matched = len(wikiplots_with_imdb)
    movielens_matched = len(matched)
    total_wikiplots = len(wiki_plots_raw)
    total_movielens = len(links_df)

    report = {
        "total_wikiplots_extracted": total_wikiplots,
        "wikiplots_with_wikidata_match": wikidata_matched,
        "wikidata_match_rate": f"{wikidata_matched / max(total_wikiplots, 1) * 100:.1f}%",
        "total_movielens_movies": total_movielens,
        "wikiplots_in_movielens": movielens_matched,
        "movielens_match_rate": f"{movielens_matched / max(total_movielens, 1) * 100:.1f}%",
    }
    report_path = REPORTS_DIR / "stage2_bridge_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Stage 2 report: {report}")

    # Save unmatched WikiPlots
    unmatched_wikiplots = wiki_plots_raw.join(wikidata_mapping, on="wiki_title", how="anti")
    unmatched_path = REPORTS_DIR / "stage2_unmatched_wikiplots.parquet"
    unmatched_wikiplots.write_parquet(unmatched_path)

    # Save matched
    matched_path = REPORTS_DIR / "stage2_matched_movies.parquet"
    matched.write_parquet(matched_path)

    return matched
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_join_movielens.py -v`
Expected: all tests PASS

**Step 6: Commit**

```bash
git add pipeline/wikidata_bridge.py pipeline/join_movielens.py tests/pipeline/test_join_movielens.py
git commit -m "feat: add Wikidata bridge and MovieLens join (Stage 2)"
```

---

### Task 5: Stage 3 — Async Wikipedia API backfill

**Files:**
- Create: `pipeline/backfill.py`
- Test: `tests/pipeline/test_backfill.py`

**Step 1: Write tests for wikitext cleaning in the backfill context**

Create `tests/pipeline/test_backfill.py`:

```python
"""Tests for Wikipedia API backfill."""

from pipeline.backfill import parse_plot_from_api_response


class TestParsePlotFromApiResponse:
    def test_extracts_plot_from_wikitext(self):
        wikitext = (
            "== Plot ==\n"
            "The hero goes on an adventure.\n"
            "They find a treasure.\n\n"
            "== Cast ==\n"
            "Actor One as Hero\n"
        )
        result = parse_plot_from_api_response(wikitext)
        assert result is not None
        assert "adventure" in result
        assert "treasure" in result
        assert "Actor" not in result

    def test_returns_none_for_no_plot(self):
        wikitext = "== Cast ==\nActor One\n"
        assert parse_plot_from_api_response(wikitext) is None

    def test_handles_empty_input(self):
        assert parse_plot_from_api_response("") is None
        assert parse_plot_from_api_response(None) is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_backfill.py -v`
Expected: FAIL — ImportError

**Step 3: Write the backfill module**

Create `pipeline/backfill.py`:

```python
"""Async Wikipedia API backfill for movies missing plot data.

For MovieLens movies not matched in Stage 2, this module:
1. Queries Wikidata SPARQL to find Wikipedia article titles by imdbId
2. Fetches Plot sections from the Wikipedia API
3. Checkpoints progress every 5000 movies
"""

import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp
import polars as pl
from tqdm import tqdm

from pipeline.download import DATA_DIR, REPORTS_DIR
from pipeline.extract_plots import clean_wikitext, extract_plot_section

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"


def parse_plot_from_api_response(wikitext: str | None) -> str | None:
    """Parse Plot section from Wikipedia API wikitext response."""
    if not wikitext:
        return None
    return extract_plot_section(wikitext)


async def fetch_plot_for_title(
    session: aiohttp.ClientSession,
    title: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str | None, str | None]:
    """Fetch plot text for a single Wikipedia article.

    Returns: (title, plot_text or None, error_reason or None)
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
        "redirects": "1",
    }
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(WIKIPEDIA_API_URL, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        return (title, None, f"http_{resp.status}")
                    data = await resp.json()
                    if "error" in data:
                        return (title, None, data["error"].get("code", "api_error"))
                    wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
                    plot = parse_plot_from_api_response(wikitext)
                    if plot:
                        return (title, plot, None)
                    return (title, None, "no_plot_section")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return (title, None, f"network_error: {type(e).__name__}")
    return (title, None, "max_retries")


def resolve_imdb_to_wiki_titles(imdb_ids: list[str]) -> dict[str, str]:
    """Batch-query Wikidata to resolve imdbId → Wikipedia article title.

    Args:
        imdb_ids: List of IMDb IDs (e.g., ["tt0114709", ...])

    Returns:
        Dict mapping imdbId → Wikipedia article title
    """
    import requests

    result = {}
    chunk_size = 2000
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "Mamba4RecFusion/1.0 (research)",
    }

    for i in tqdm(range(0, len(imdb_ids), chunk_size), desc="Wikidata SPARQL lookups"):
        chunk = imdb_ids[i : i + chunk_size]
        values = " ".join(f'"{iid}"' for iid in chunk)
        query = f"""
        SELECT ?imdbId ?articleTitle WHERE {{
          VALUES ?imdbId {{ {values} }}
          ?item wdt:P345 ?imdbId .
          ?article schema:about ?item ;
                   schema:isPartOf <https://en.wikipedia.org/> ;
                   schema:name ?articleTitle .
        }}
        """
        for attempt in range(3):
            try:
                resp = requests.get(
                    WIKIDATA_SPARQL_URL,
                    params={"query": query},
                    headers=headers,
                    timeout=120,
                )
                resp.raise_for_status()
                bindings = resp.json()["results"]["bindings"]
                for b in bindings:
                    result[b["imdbId"]["value"]] = b["articleTitle"]["value"]
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"  Failed chunk starting at {i}: {e}")

    return result


async def backfill_plots(
    unmatched_df: pl.DataFrame,
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    """Async backfill plots for unmatched MovieLens movies.

    Args:
        unmatched_df: DataFrame with movieId, imdbId columns
        checkpoint_dir: Directory for checkpoint files (default: REPORTS_DIR)

    Returns:
        DataFrame with movieId, imdbId, wiki_title, plot_text for successful fetches
    """
    if checkpoint_dir is None:
        checkpoint_dir = REPORTS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Resolve imdbIds to Wikipedia titles
    imdb_ids = unmatched_df["imdbId"].to_list()
    imdb_to_wiki = resolve_imdb_to_wiki_titles(imdb_ids)
    print(f"  Wikidata resolved {len(imdb_to_wiki)}/{len(imdb_ids)} to Wikipedia titles")

    # Step 2: Fetch plots from Wikipedia API
    semaphore = asyncio.Semaphore(50)
    results = []
    failures = []

    # Load checkpoint if exists
    checkpoint_path = checkpoint_dir / "stage3_checkpoint.parquet"
    completed_titles = set()
    if checkpoint_path.exists():
        checkpoint_df = pl.read_parquet(checkpoint_path)
        completed_titles = set(checkpoint_df["wiki_title"].to_list())
        results = checkpoint_df.to_dicts()
        print(f"  Resuming from checkpoint: {len(completed_titles)} already done")

    titles_to_fetch = [
        (imdb_id, title)
        for imdb_id, title in imdb_to_wiki.items()
        if title not in completed_titles
    ]

    headers = {"User-Agent": "Mamba4RecFusion/1.0 (research; polite)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        imdb_for_title = {}
        for imdb_id, title in titles_to_fetch:
            imdb_for_title[title] = imdb_id
            tasks.append(fetch_plot_for_title(session, title, semaphore))

        for i, coro in enumerate(tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Fetching plots from Wikipedia",
        )):
            title, plot, error = await coro
            imdb_id = imdb_for_title.get(title, "")
            if plot:
                results.append({
                    "wiki_title": title,
                    "imdbId": imdb_id,
                    "plot_text": plot,
                })
            else:
                failures.append({
                    "wiki_title": title,
                    "imdbId": imdb_id,
                    "failure_reason": error or "unknown",
                })

            # Checkpoint every 5000
            if (i + 1) % 5000 == 0:
                _save_checkpoint(results, checkpoint_path)

    _save_checkpoint(results, checkpoint_path)

    # Join back with movieId
    if not results:
        return pl.DataFrame(schema={"movieId": pl.UInt32, "imdbId": pl.Utf8, "wiki_title": pl.Utf8, "plot_text": pl.Utf8})

    results_df = pl.DataFrame(results)
    backfilled = results_df.join(
        unmatched_df.select(["movieId", "imdbId"]),
        on="imdbId",
        how="inner",
    )

    # Reports
    _write_backfill_reports(imdb_ids, imdb_to_wiki, results, failures)

    return backfilled


def _save_checkpoint(results: list[dict], path: Path):
    if results:
        pl.DataFrame(results).write_parquet(path)


def _write_backfill_reports(
    all_imdb_ids: list[str],
    imdb_to_wiki: dict[str, str],
    results: list[dict],
    failures: list[dict],
):
    report = {
        "total_unmatched": len(all_imdb_ids),
        "wikidata_resolved": len(imdb_to_wiki),
        "plots_fetched": len(results),
        "fetch_failures": len(failures),
        "failure_breakdown": {},
    }
    for f in failures:
        reason = f["failure_reason"]
        report["failure_breakdown"][reason] = report["failure_breakdown"].get(reason, 0) + 1

    report_path = REPORTS_DIR / "stage3_backfill_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Stage 3 report: {report}")

    if failures:
        pl.DataFrame(failures).write_parquet(REPORTS_DIR / "stage3_failed_lookups.parquet")

    # Short plots
    short = [r for r in results if len(r.get("plot_text", "").split()) < 50]
    if short:
        pl.DataFrame(short).write_parquet(REPORTS_DIR / "stage3_short_plots.parquet")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_backfill.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add pipeline/backfill.py tests/pipeline/test_backfill.py
git commit -m "feat: add async Wikipedia API backfill with checkpointing (Stage 3)"
```

---

### Task 6: Stage 4 — Consolidation and final reporting

**Files:**
- Create: `pipeline/consolidate.py`
- Test: `tests/pipeline/test_consolidate.py`

**Step 1: Write tests for consolidation logic**

Create `tests/pipeline/test_consolidate.py`:

```python
"""Tests for Stage 4 consolidation."""

import polars as pl
from pipeline.consolidate import consolidate


class TestConsolidate:
    def test_combines_two_sources(self):
        dump_df = pl.DataFrame({
            "movieId": pl.Series([1, 2], dtype=pl.UInt32),
            "imdbId": ["tt0000001", "tt0000002"],
            "title": ["Movie A", "Movie B"],
            "year": pl.Series([2000, 2001], dtype=pl.UInt16),
            "plot_text": ["A long enough plot for movie A that has many words in it.", "Another plot for movie B with enough words."],
            "genres": ["Action|Drama", "Comedy"],
        })
        backfill_df = pl.DataFrame({
            "movieId": pl.Series([3], dtype=pl.UInt32),
            "imdbId": ["tt0000003"],
            "title": ["Movie C"],
            "year": pl.Series([2002], dtype=pl.UInt16),
            "plot_text": ["A backfilled plot for movie C with enough words."],
            "genres": ["Horror"],
        })
        result = consolidate(dump_df, backfill_df)
        assert len(result) == 3
        assert set(result["plot_source"].to_list()) == {"wiki_dump", "wikipedia_api"}
        assert "plot_length" in result.columns

    def test_deduplicates_on_movie_id(self):
        df1 = pl.DataFrame({
            "movieId": pl.Series([1], dtype=pl.UInt32),
            "imdbId": ["tt0000001"],
            "title": ["Movie A"],
            "year": pl.Series([2000], dtype=pl.UInt16),
            "plot_text": ["Dump version of the plot."],
            "genres": ["Drama"],
        })
        df2 = pl.DataFrame({
            "movieId": pl.Series([1], dtype=pl.UInt32),
            "imdbId": ["tt0000001"],
            "title": ["Movie A"],
            "year": pl.Series([2000], dtype=pl.UInt16),
            "plot_text": ["API version of the plot."],
            "genres": ["Drama"],
        })
        result = consolidate(df1, df2)
        assert len(result) == 1
        # Dump version should take priority
        assert result["plot_source"][0] == "wiki_dump"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_consolidate.py -v`
Expected: FAIL — ImportError

**Step 3: Write the consolidation module**

Create `pipeline/consolidate.py`:

```python
"""Stage 4: Consolidate matched and backfilled plots into final Parquet."""

import json
from pathlib import Path

import polars as pl

from pipeline.download import DATA_DIR, REPORTS_DIR


def consolidate(
    dump_matched: pl.DataFrame,
    backfill_df: pl.DataFrame,
) -> pl.DataFrame:
    """Combine dump-matched and API-backfilled plots.

    Deduplicates on movieId, preferring dump_matched (wiki_dump source).

    Args:
        dump_matched: Stage 2 output (movieId, imdbId, title, year, plot_text, genres)
        backfill_df: Stage 3 output (same schema)

    Returns:
        Consolidated DataFrame with plot_source and plot_length columns added
    """
    # Add source column
    dump_tagged = dump_matched.with_columns(pl.lit("wiki_dump").alias("plot_source"))
    backfill_tagged = backfill_df.with_columns(pl.lit("wikipedia_api").alias("plot_source"))

    # Ensure matching schemas
    common_cols = ["movieId", "imdbId", "title", "year", "plot_text", "genres", "plot_source"]
    dump_tagged = dump_tagged.select([c for c in common_cols if c in dump_tagged.columns])
    backfill_tagged = backfill_tagged.select([c for c in common_cols if c in backfill_tagged.columns])

    # Concatenate
    combined = pl.concat([dump_tagged, backfill_tagged], how="diagonal_relaxed")

    # Deduplicate: keep wiki_dump over wikipedia_api
    combined = combined.sort("plot_source")  # wiki_dump sorts before wikipedia_api
    combined = combined.unique(subset=["movieId"], keep="first")

    # Add plot_length
    combined = combined.with_columns(
        pl.col("plot_text").str.split(" ").list.len().alias("plot_length").cast(pl.UInt32)
    )

    return combined.sort("movieId")


def write_final_reports(
    final_df: pl.DataFrame,
    links_df: pl.DataFrame,
    movies_df: pl.DataFrame,
):
    """Write Stage 4 quality reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    total_movielens = len(links_df)
    total_with_plots = len(final_df)
    from_dump = final_df.filter(pl.col("plot_source") == "wiki_dump").height
    from_api = final_df.filter(pl.col("plot_source") == "wikipedia_api").height
    missing = total_movielens - total_with_plots

    # Plot length stats
    lengths = final_df["plot_length"]
    length_stats = {
        "min": int(lengths.min()) if lengths.len() > 0 else 0,
        "p25": int(lengths.quantile(0.25)) if lengths.len() > 0 else 0,
        "median": int(lengths.median()) if lengths.len() > 0 else 0,
        "p75": int(lengths.quantile(0.75)) if lengths.len() > 0 else 0,
        "max": int(lengths.max()) if lengths.len() > 0 else 0,
    }

    report = {
        "total_movielens_movies": total_movielens,
        "movies_with_plot": total_with_plots,
        "coverage_pct": f"{total_with_plots / max(total_movielens, 1) * 100:.1f}%",
        "from_wiki_dump": from_dump,
        "from_wikipedia_api": from_api,
        "still_missing": missing,
        "missing_pct": f"{missing / max(total_movielens, 1) * 100:.1f}%",
        "plot_length_distribution": length_stats,
    }

    # Per-genre match rate
    if "genres" in final_df.columns and "genres" in movies_df.columns:
        all_genres = set()
        for g in movies_df["genres"].drop_nulls().to_list():
            all_genres.update(g.split("|"))
        genre_rates = {}
        for genre in sorted(all_genres - {"(no genres listed)"}):
            total_genre = movies_df.filter(pl.col("genres").str.contains(genre)).height
            matched_genre = final_df.filter(pl.col("genres").str.contains(genre)).height
            if total_genre > 0:
                genre_rates[genre] = f"{matched_genre}/{total_genre} ({matched_genre / total_genre * 100:.0f}%)"
        report["per_genre_match_rate"] = genre_rates

    report_path = REPORTS_DIR / "stage4_final_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Stage 4 report: {json.dumps(report, indent=2)}")

    # No-plot movies list
    matched_ids = set(final_df["imdbId"].to_list())
    import re

    def _parse_year(title):
        m = re.match(r"^.+?\((\d{4})\)\s*$", title)
        return int(m.group(1)) if m else None

    def _parse_title(title):
        m = re.match(r"^(.+?)\s*\(\d{4}\)\s*$", title)
        return m.group(1).strip() if m else title.strip()

    no_plot = (
        links_df.join(movies_df, on="movieId", how="left")
        .with_columns(
            pl.col("imdbId").cast(pl.Utf8),
        )
        .filter(~pl.col("imdbId").is_in(list(matched_ids)))
        .with_columns([
            pl.col("title").map_elements(_parse_title, return_dtype=pl.Utf8).alias("clean_title"),
            pl.col("title").map_elements(_parse_year, return_dtype=pl.UInt16).alias("year"),
        ])
        .select(["imdbId", "clean_title", "year", "genres"])
        .rename({"clean_title": "title"})
        .sort("year", descending=True)
    )

    no_plot.write_parquet(REPORTS_DIR / "stage4_no_plot_movies.parquet")
    no_plot.write_csv(REPORTS_DIR / "stage4_no_plot_movies.csv")
    print(f"  Wrote {len(no_plot)} unmatched movies to stage4_no_plot_movies.csv")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_consolidate.py -v`
Expected: all tests PASS

**Step 5: Commit**

```bash
git add pipeline/consolidate.py tests/pipeline/test_consolidate.py
git commit -m "feat: add consolidation and final reporting (Stage 4)"
```

---

### Task 7: Main pipeline runner

**Files:**
- Create: `pipeline/run_pipeline.py`

**Step 1: Write the pipeline orchestrator**

Create `pipeline/run_pipeline.py`:

```python
"""Main pipeline runner — orchestrates all 4 stages."""

import asyncio
import sys
from pathlib import Path

import polars as pl

from pipeline.download import DATA_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs


def main():
    ensure_dirs()

    # --- Stage 1: Extract plots from Wikipedia dump ---
    print("=" * 60)
    print("STAGE 1: Extract plots from Wikipedia dump")
    print("=" * 60)

    wiki_plots_path = DATA_DIR / "wiki_plots_raw.parquet"
    dump_path = RAW_DIR / "enwiki-latest-pages-articles-multistream.xml.bz2"

    if wiki_plots_path.exists():
        print(f"  Loading cached: {wiki_plots_path}")
        wiki_plots_raw = pl.read_parquet(wiki_plots_path)
    else:
        if not dump_path.exists():
            print(f"  ERROR: Wikipedia dump not found at {dump_path}")
            print(f"  Run: python -m pipeline.download")
            sys.exit(1)
        from pipeline.extract_plots import extract_plots_from_dump
        wiki_plots_raw = extract_plots_from_dump(dump_path)
        wiki_plots_raw.write_parquet(wiki_plots_path)

    print(f"  Extracted {len(wiki_plots_raw)} plots")

    # --- Stage 2: Wikidata bridge + MovieLens join ---
    print("=" * 60)
    print("STAGE 2: Wikidata bridge + MovieLens join")
    print("=" * 60)

    matched_path = REPORTS_DIR / "stage2_matched_movies.parquet"
    movielens_dir = RAW_DIR / "ml-32m"

    if not movielens_dir.exists():
        print(f"  ERROR: MovieLens not found at {movielens_dir}")
        print(f"  Run: python -m pipeline.download")
        sys.exit(1)

    if matched_path.exists():
        print(f"  Loading cached: {matched_path}")
        matched_df = pl.read_parquet(matched_path)
    else:
        from pipeline.wikidata_bridge import fetch_wikidata_mapping
        from pipeline.join_movielens import run_stage2

        wikidata_cache = DATA_DIR / "wikidata_mapping.parquet"
        wikidata_mapping = fetch_wikidata_mapping(cache_path=wikidata_cache)
        print(f"  Wikidata mapping: {len(wikidata_mapping)} entries")

        matched_df = run_stage2(wiki_plots_raw, wikidata_mapping, movielens_dir)

    print(f"  Matched {len(matched_df)} movies")

    # --- Stage 3: Gap identification + backfill ---
    print("=" * 60)
    print("STAGE 3: Gap identification + Wikipedia API backfill")
    print("=" * 60)

    links_df = pl.read_csv(movielens_dir / "links.csv")
    from pipeline.join_movielens import normalize_imdb_id

    links_norm = links_df.with_columns(
        pl.col("imdbId").cast(pl.Utf8).map_elements(normalize_imdb_id, return_dtype=pl.Utf8).alias("imdbId")
    )

    matched_ids = set(matched_df["imdbId"].to_list())
    unmatched = links_norm.filter(~pl.col("imdbId").is_in(list(matched_ids)))
    print(f"  Unmatched MovieLens movies: {len(unmatched)}")

    backfill_path = REPORTS_DIR / "stage3_checkpoint.parquet"
    if len(unmatched) > 0:
        from pipeline.backfill import backfill_plots

        # Enrich unmatched with title/year/genres for reporting
        movies_df = pl.read_csv(movielens_dir / "movies.csv")
        unmatched_enriched = unmatched.join(movies_df, on="movieId", how="left")

        backfill_df = asyncio.run(backfill_plots(unmatched_enriched))

        # Add title/year/genres from movies
        if len(backfill_df) > 0 and "title" not in backfill_df.columns:
            backfill_df = backfill_df.join(
                movies_df, on="movieId", how="left"
            )
        print(f"  Backfilled {len(backfill_df)} additional plots")
    else:
        backfill_df = pl.DataFrame(schema={
            "movieId": pl.UInt32, "imdbId": pl.Utf8, "title": pl.Utf8,
            "year": pl.UInt16, "plot_text": pl.Utf8, "genres": pl.Utf8,
        })

    # --- Stage 4: Consolidate ---
    print("=" * 60)
    print("STAGE 4: Consolidate and store")
    print("=" * 60)

    from pipeline.consolidate import consolidate, write_final_reports

    final_df = consolidate(matched_df, backfill_df)
    final_path = DATA_DIR / "movie_plots.parquet"
    final_df.write_parquet(final_path)
    print(f"  Wrote {len(final_df)} movies to {final_path}")

    movies_df = pl.read_csv(movielens_dir / "movies.csv")
    write_final_reports(final_df, links_norm, movies_df)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Output: {final_path}")
    print(f"  Reports: {REPORTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it imports cleanly**

Run: `python -c "from pipeline.run_pipeline import main; print('OK')"`
Expected: prints `OK`

**Step 3: Commit**

```bash
git add pipeline/run_pipeline.py
git commit -m "feat: add main pipeline runner orchestrating all 4 stages"
```

---

### Task 8: Run all tests and verify

**Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS

**Step 2: Verify linting passes**

Run: `ruff check pipeline/ tests/`
Expected: no errors (fix any issues)

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: address linting issues in pipeline"
```

---

### Task 9: Download data and run pipeline

**Step 1: Download MovieLens 32M**

Run: `python -m pipeline.download`

Note: The Wikipedia dump is ~24GB and takes a while. You can start with just MovieLens and test the pipeline structure.

**Step 2: Run the full pipeline**

Run: `python -m pipeline.run_pipeline`

**Step 3: Inspect reports**

Check each report file in `data/reports/` for data quality. Key files:
- `stage1_extract_report.json` — how many plots extracted
- `stage2_bridge_report.json` — Wikidata and MovieLens match rates
- `stage3_backfill_report.json` — backfill success/failure breakdown
- `stage4_final_report.json` — overall coverage stats
- `stage4_no_plot_movies.csv` — the list of movies with no plot (imdbId, title, year, genres)

**Step 4: Commit the pipeline outputs to .gitignore**

Add to `.gitignore`:
```
data/raw/
data/*.parquet
data/reports/*.parquet
```

```bash
git add .gitignore
git commit -m "chore: gitignore large data files and parquet outputs"
```
