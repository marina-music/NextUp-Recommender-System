"""Query Wikidata SPARQL to map Wikipedia article titles to IMDb IDs."""

import json
import time
from pathlib import Path

import polars as pl
import requests
from tqdm import tqdm

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

BULK_QUERY = """
SELECT ?articleTitle ?imdbId WHERE {
  ?item wdt:P345 ?imdbId .
  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?articleTitle .
}
"""


def fetch_wikidata_mapping(cache_path: Path | None = None) -> pl.DataFrame:
    """Fetch the full Wikipedia title -> IMDb ID mapping from Wikidata.

    This is a single large SPARQL query. Results are cached to disk.

    Args:
        cache_path: Path to cache the results as parquet.

    Returns:
        DataFrame with columns: wiki_title (Utf8), imdbId (Utf8)
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached Wikidata mapping from {cache_path}")
        return pl.read_parquet(cache_path)

    print("  Querying Wikidata SPARQL for Wikipedia -> IMDb mapping...")
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
        except (requests.RequestException, requests.Timeout) as exc:
            if attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} after {wait}s: {exc}")
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
