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
