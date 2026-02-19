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
