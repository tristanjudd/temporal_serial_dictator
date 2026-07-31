"""Download the real-world .tsoi datasets and convert them to JSONL.

The real-world datasets are too large to check into the repository
itself (the raw collection is ~270MB, ~630MB once converted to JSONL).
This is how anyone cloning the repo gets them in one step: download the
same "sudema" tsoi dataset collection Martin Lackner's perpetual-voting
codebase uses (see lackner_perpetual_voting/README.md for the original
pointer to this same URL), extract it into real_data/, then convert
every dataset to JSONL via convert_lackner_datasets.convert_all() --
exactly the same conversion step used when the raw data is already
present locally, so nothing about the conversion logic is duplicated
here.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from rich.progress import Progress

from .convert_lackner_datasets import REAL_DATA_DIR, convert_all

DATA_URL = "https://www.dbai.tuwien.ac.at/proj/sudema/data/data.zip"

# The archive extracts to a single top-level "data/" folder containing
# exactly these three directories, which become real_data/<name>.
ARCHIVE_TOP_LEVEL_DIRS = ["eurovision_song_contest_tsoi", "i_phone", "spotify"]

DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _download(url: str, dest: Path) -> None:
    """Download url to dest, showing a progress bar (byte-based, since
    this is one large file rather than many small items).
    """
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0)) or None
        with Progress() as progress, dest.open("wb") as f:
            task = progress.add_task("Downloading dataset archive...", total=total)
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                progress.update(task, advance=len(chunk))


def _extract_and_place(archive_path: Path) -> None:
    """Extract archive_path (a copy of data.zip) and move its top-level
    dataset directories into REAL_DATA_DIR.

    A destination that already exists is left untouched (rather than
    overwritten) rather than re-copied, so a partially-populated
    real_data/ isn't clobbered by a retry.
    """
    with tempfile.TemporaryDirectory() as extract_tmp:
        extract_path = Path(extract_tmp)
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_path)
        except (zipfile.BadZipFile, OSError) as e:
            print(f"Error extracting '{archive_path}': {e}", file=sys.stderr)
            return

        extracted_data_dir = extract_path / "data"
        REAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name in ARCHIVE_TOP_LEVEL_DIRS:
            source = extracted_data_dir / name
            dest = REAL_DATA_DIR / name
            if dest.exists():
                print(f"'{dest}' already exists, skipping.")
                continue
            if not source.exists():
                print(
                    f"Warning: expected '{source}' in the archive but it's missing.",
                    file=sys.stderr,
                )
                continue
            shutil.move(str(source), str(dest))
            print(f"Moved {name} -> {dest}")


def download_and_convert_data() -> None:
    """Download the real-world .tsoi dataset collection (unless every
    top-level dataset directory is already present under real_data/),
    extract it, and convert every dataset to JSONL via
    convert_lackner_datasets.convert_all().
    """
    missing_dirs = [name for name in ARCHIVE_TOP_LEVEL_DIRS if not (REAL_DATA_DIR / name).exists()]

    if not missing_dirs:
        print(f"Raw datasets already present under '{REAL_DATA_DIR}', skipping download.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "data.zip"
            print(f"Downloading {DATA_URL} ...")
            try:
                _download(DATA_URL, archive_path)
            except OSError as e:
                print(f"Error downloading '{DATA_URL}': {e}", file=sys.stderr)
                return

            print(f"Extracting {archive_path} ...")
            _extract_and_place(archive_path)

    print("Converting datasets to JSONL...")
    convert_all()


if __name__ == "__main__":
    download_and_convert_data()
