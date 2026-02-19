"""Async Wikipedia API backfill for movies missing plot data.

For MovieLens movies not matched in Stage 2, this module:
1. Queries Wikidata SPARQL to find Wikipedia article titles by imdbId
2. Fetches Plot sections from the Wikipedia API
3. Checkpoints progress every 5000 movies
"""

import asyncio
import json
import time
from pathlib import Path

import aiohttp
import polars as pl
from tqdm import tqdm

from pipeline.download import REPORTS_DIR
from pipeline.extract_plots import extract_plot_section

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
                async with session.get(
                    WIKIPEDIA_API_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        return (title, None, f"http_{resp.status}")
                    data = await resp.json()
                    if "error" in data:
                        return (title, None, data["error"].get("code", "api_error"))
                    wikitext = (
                        data.get("parse", {}).get("wikitext", {}).get("*", "")
                    )
                    plot = parse_plot_from_api_response(wikitext)
                    if plot:
                        return (title, plot, None)
                    return (title, None, "no_plot_section")
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                return (title, None, f"network_error: {type(exc).__name__}")
    return (title, None, "max_retries")


def resolve_imdb_to_wiki_titles(imdb_ids: list[str]) -> dict[str, str]:
    """Batch-query Wikidata to resolve imdbId -> Wikipedia article title.

    Args:
        imdb_ids: List of IMDb IDs (e.g., ["tt0114709", ...])

    Returns:
        Dict mapping imdbId -> Wikipedia article title
    """
    import requests

    result = {}
    chunk_size = 2000
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "Mamba4RecFusion/1.0 (research)",
    }

    for i in tqdm(
        range(0, len(imdb_ids), chunk_size), desc="Wikidata SPARQL lookups"
    ):
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
            except Exception as exc:
                if attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    print(f"  Failed chunk starting at {i}: {exc}")

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

    imdb_ids = unmatched_df["imdbId"].to_list()
    imdb_to_wiki = resolve_imdb_to_wiki_titles(imdb_ids)
    print(
        f"  Wikidata resolved {len(imdb_to_wiki)}/{len(imdb_ids)} to Wikipedia titles"
    )

    semaphore = asyncio.Semaphore(50)
    results = []
    failures = []

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

        for i, coro in enumerate(
            tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Fetching plots from Wikipedia",
            )
        ):
            title, plot, error = await coro
            imdb_id = imdb_for_title.get(title, "")
            if plot:
                results.append(
                    {
                        "wiki_title": title,
                        "imdbId": imdb_id,
                        "plot_text": plot,
                    }
                )
            else:
                failures.append(
                    {
                        "wiki_title": title,
                        "imdbId": imdb_id,
                        "failure_reason": error or "unknown",
                    }
                )

            if (i + 1) % 5000 == 0:
                _save_checkpoint(results, checkpoint_path)

    _save_checkpoint(results, checkpoint_path)

    if not results:
        return pl.DataFrame(
            schema={
                "movieId": pl.UInt32,
                "imdbId": pl.Utf8,
                "wiki_title": pl.Utf8,
                "plot_text": pl.Utf8,
            }
        )

    results_df = pl.DataFrame(results)
    backfilled = results_df.join(
        unmatched_df.select(["movieId", "imdbId"]),
        on="imdbId",
        how="inner",
    )

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
    for failure in failures:
        reason = failure["failure_reason"]
        report["failure_breakdown"][reason] = (
            report["failure_breakdown"].get(reason, 0) + 1
        )

    report_path = REPORTS_DIR / "stage3_backfill_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Stage 3 report: {report}")

    if failures:
        pl.DataFrame(failures).write_parquet(
            REPORTS_DIR / "stage3_failed_lookups.parquet"
        )

    short = [r for r in results if len(r.get("plot_text", "").split()) < 50]
    if short:
        pl.DataFrame(short).write_parquet(
            REPORTS_DIR / "stage3_short_plots.parquet"
        )
