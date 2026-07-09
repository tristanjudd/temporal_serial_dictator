"""Batch-convert all of Lackner's real .tsoi datasets to JSONL.

Runs tsoi_dir_to_json on every known real_data .tsoi dataset directory,
saving each result to real_data/json_datasets/<name>, where <name> is the
directory's path relative to real_data/ with '/' replaced by '-' (e.g.
spotify/weekly_tsoi -> spotify-weekly_tsoi).
"""

from __future__ import annotations

from pathlib import Path

from rich.progress import track

from src.data_transformation.tsoi_to_json import tsoi_dir_to_json

REAL_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "real_data"
OUTPUT_DIR = REAL_DATA_DIR / "json_datasets"

DATASET_DIRS = [
    "eurovision_song_contest_tsoi",
    "i_phone/games/free_games_tsoi",
    "i_phone/games/gross_games_tsoi",
    "i_phone/games/paid_games_tsoi",
    "i_phone/news/free_news_tsoi",
    "i_phone/news/gross_news_tsoi",
    "i_phone/news/paid_news_tsoi",
    "spotify/daily_tsoi",
    "spotify/viral_daily_tsoi",
    "spotify/viral_weekly_tsoi",
    "spotify/weekly_tsoi",
]


def convert_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for relative_dir in track(DATASET_DIRS, description="Converting datasets..."):
        name = relative_dir.replace("/", "-")
        print(f"Converting {relative_dir} -> json_datasets/{name}")
        tsoi_dir_to_json(REAL_DATA_DIR / relative_dir, OUTPUT_DIR / name)


if __name__ == "__main__":
    convert_all()
