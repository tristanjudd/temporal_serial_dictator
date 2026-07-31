import zipfile
from pathlib import Path

from src.data_transformation import download_and_convert_data as dl_module
from src.data_transformation.download_and_convert_data import (
    ARCHIVE_TOP_LEVEL_DIRS,
    _extract_and_place,
    download_and_convert_data,
)


def _build_fake_archive(path: Path) -> None:
    """A minimal zip matching the real archive's shape: a top-level
    "data/" folder containing the three known dataset directories, each
    with one tiny file.
    """
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data/eurovision_song_contest_tsoi/round_1.tsoi", "eurovision content")
        zf.writestr("data/i_phone/games/free_games_tsoi/round_1.tsoi", "i_phone content")
        zf.writestr("data/spotify/daily_tsoi/round_1.tsoi", "spotify content")


def test_extract_and_place_moves_all_top_level_dirs(tmp_path: Path, monkeypatch) -> None:
    real_data_dir = tmp_path / "real_data"
    monkeypatch.setattr(dl_module, "REAL_DATA_DIR", real_data_dir)

    archive_path = tmp_path / "data.zip"
    _build_fake_archive(archive_path)

    _extract_and_place(archive_path)

    assert (real_data_dir / "eurovision_song_contest_tsoi" / "round_1.tsoi").read_text() == (
        "eurovision content"
    )
    assert (real_data_dir / "i_phone" / "games" / "free_games_tsoi" / "round_1.tsoi").exists()
    assert (real_data_dir / "spotify" / "daily_tsoi" / "round_1.tsoi").exists()


def test_extract_and_place_does_not_overwrite_existing_destination(
    tmp_path: Path, monkeypatch
) -> None:
    real_data_dir = tmp_path / "real_data"
    monkeypatch.setattr(dl_module, "REAL_DATA_DIR", real_data_dir)

    existing = real_data_dir / "eurovision_song_contest_tsoi"
    existing.mkdir(parents=True)
    (existing / "already_here.tsoi").write_text("pre-existing content")

    archive_path = tmp_path / "data.zip"
    _build_fake_archive(archive_path)
    _extract_and_place(archive_path)

    # untouched: the archive's version was not moved in over it.
    assert (existing / "already_here.tsoi").exists()
    assert not (existing / "round_1.tsoi").exists()
    # the other two dirs, which didn't already exist, are still placed.
    assert (real_data_dir / "i_phone" / "games" / "free_games_tsoi" / "round_1.tsoi").exists()


def test_extract_and_place_handles_corrupt_archive_gracefully(tmp_path: Path, monkeypatch) -> None:
    real_data_dir = tmp_path / "real_data"
    monkeypatch.setattr(dl_module, "REAL_DATA_DIR", real_data_dir)

    bad_archive = tmp_path / "data.zip"
    bad_archive.write_text("not a zip file")

    _extract_and_place(bad_archive)  # should not raise

    assert not real_data_dir.exists() or list(real_data_dir.iterdir()) == []


def test_download_is_skipped_when_all_dirs_already_present(tmp_path: Path, monkeypatch) -> None:
    real_data_dir = tmp_path / "real_data"
    for name in ARCHIVE_TOP_LEVEL_DIRS:
        (real_data_dir / name).mkdir(parents=True)
    monkeypatch.setattr(dl_module, "REAL_DATA_DIR", real_data_dir)

    download_calls = []
    monkeypatch.setattr(dl_module, "_download", lambda url, dest: download_calls.append(url))
    convert_calls = []
    monkeypatch.setattr(dl_module, "convert_all", lambda: convert_calls.append(True))

    download_and_convert_data()

    assert download_calls == []
    assert convert_calls == [True]


def test_download_is_triggered_when_a_dir_is_missing(tmp_path: Path, monkeypatch) -> None:
    real_data_dir = tmp_path / "real_data"
    monkeypatch.setattr(dl_module, "REAL_DATA_DIR", real_data_dir)

    def fake_download(url: str, dest: Path) -> None:
        _build_fake_archive(dest)

    monkeypatch.setattr(dl_module, "_download", fake_download)
    convert_calls = []
    monkeypatch.setattr(dl_module, "convert_all", lambda: convert_calls.append(True))

    download_and_convert_data()

    assert (real_data_dir / "eurovision_song_contest_tsoi" / "round_1.tsoi").exists()
    assert convert_calls == [True]
