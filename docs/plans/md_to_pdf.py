"""Convert a Markdown file to PDF using pandoc (via weasyprint engine)."""
import os
import sys
from pathlib import Path

import pypandoc

# Ensure venv binaries (weasyprint) are on PATH for pandoc to find
VENV_BIN = Path(sys.executable).parent
os.environ["PATH"] = str(VENV_BIN) + os.pathsep + os.environ.get("PATH", "")

STYLE = Path(__file__).parent / "style.css"


def convert(md_path: str, pdf_path: str | None = None):
    md_path = Path(md_path)
    pdf_path = Path(pdf_path) if pdf_path else md_path.with_suffix(".pdf")

    pypandoc.convert_file(
        str(md_path),
        "pdf",
        outputfile=str(pdf_path),
        extra_args=[
            "--pdf-engine=weasyprint",
            "--css", str(STYLE),
        ],
    )
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf.py <input.md> [output.pdf]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
