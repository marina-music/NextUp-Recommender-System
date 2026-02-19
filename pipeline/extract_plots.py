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
            body_lines = [line for line in lines[1:] if not line.strip().startswith("==")]
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
