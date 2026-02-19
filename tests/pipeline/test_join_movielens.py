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
