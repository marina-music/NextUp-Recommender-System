"""Filter wiki_plots_raw.parquet to movies/TV only via Wikidata SPARQL."""
import json
import time
from pathlib import Path
from typing import Set

import polars as pl
import requests

from pipeline.download import DATA_DIR, REPORTS_DIR

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Instance-of types for movies and TV
SPARQL_QUERY = """
SELECT DISTINCT ?articleTitle WHERE {{
  ?article schema:isPartOf <https://en.wikipedia.org/> ;
           schema:name ?articleTitle ;
           schema:about ?item .
  ?item wdt:P31 ?instance .
  VALUES ?instance {{
    wd:Q11424      wd:Q24856      wd:Q5398426
    wd:Q21191270   wd:Q24862      wd:Q506240
    wd:Q1261214    wd:Q63952888   wd:Q220898
  }}
}}
"""


def fetch_movie_tv_titles() -> Set[str]:
    """Fetch all Wikipedia article titles that are movies or TV from Wikidata."""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "Mamba4RecFusion/1.0 (research; polite)",
    }

    for attempt in range(5):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": SPARQL_QUERY},
                headers=headers,
                timeout=300,
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            titles = set()
            for binding in data["results"]["bindings"]:
                titles.add(binding["articleTitle"]["value"])
            return titles
        except Exception as e:
            if attempt < 4:
                print(f"  Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(10 * (attempt + 1))
                continue
            raise

    return set()


def filter_plots_to_movies_tv(
    raw_plots: pl.DataFrame, movie_tv_titles: Set[str]
) -> pl.DataFrame:
    """Filter raw plots DataFrame to only movies/TV titles."""
    return raw_plots.filter(pl.col("wiki_title").is_in(list(movie_tv_titles)))


def main():
    """Run the full filter pipeline."""
    raw_path = DATA_DIR / "wiki_plots_raw.parquet"
    output_path = DATA_DIR / "wiki_plots_movies_tv.parquet"

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run the extraction pipeline first.")
        return

    raw = pl.read_parquet(raw_path)
    print(f"Raw plots: {len(raw)} articles")

    print("Fetching movie/TV titles from Wikidata...")
    movie_tv_titles = fetch_movie_tv_titles()
    print(f"Wikidata returned {len(movie_tv_titles)} movie/TV titles")

    filtered = filter_plots_to_movies_tv(raw, movie_tv_titles)
    print(f"After filtering: {len(filtered)} movie/TV articles")

    filtered.write_parquet(output_path)
    print(f"Wrote {output_path}")

    # Report
    report = {
        "raw_count": len(raw),
        "wikidata_titles": len(movie_tv_titles),
        "filtered_count": len(filtered),
        "removed_count": len(raw) - len(filtered),
    }
    report_path = REPORTS_DIR / "filter_plots_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
