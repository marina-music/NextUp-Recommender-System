"""Download Wikipedia dump and MovieLens 32M dataset."""

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
