"""Main pipeline runner — orchestrates all 4 stages."""

import asyncio
import sys

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
            print("  Run: python -m pipeline.download")
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
        print("  Run: python -m pipeline.download")
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

    if len(unmatched) > 0:
        from pipeline.backfill import backfill_plots

        movies_df = pl.read_csv(movielens_dir / "movies.csv")
        unmatched_enriched = unmatched.join(movies_df, on="movieId", how="left")

        backfill_df = asyncio.run(backfill_plots(unmatched_enriched))

        if len(backfill_df) > 0 and "title" not in backfill_df.columns:
            backfill_df = backfill_df.join(movies_df, on="movieId", how="left")
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
