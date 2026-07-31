import json
from pathlib import Path

from src.data_transformation import convert_lackner_datasets as convert_module
from src.data_transformation.convert_lackner_datasets import convert_all

ROUND_1 = """2
101,Candidate A
102,Candidate B
2,2,2
alice:1,101[10]
bob:1,102[9]
"""


def _write_source_dataset(real_data_dir: Path, relative_dir: str) -> None:
    tsoi_dir = real_data_dir / relative_dir
    tsoi_dir.mkdir(parents=True, exist_ok=True)
    (tsoi_dir / "round_1.tsoi").write_text(ROUND_1)


def test_convert_all_converts_every_dataset_dir(tmp_path: Path, monkeypatch) -> None:
    real_data_dir = tmp_path / "real_data"
    output_dir = real_data_dir / "json_datasets"
    monkeypatch.setattr(convert_module, "REAL_DATA_DIR", real_data_dir)
    monkeypatch.setattr(convert_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(convert_module, "DATASET_DIRS", ["simple_tsoi", "nested/dataset_tsoi"])

    _write_source_dataset(real_data_dir, "simple_tsoi")
    _write_source_dataset(real_data_dir, "nested/dataset_tsoi")

    convert_all()

    assert (output_dir / "simple_tsoi").exists()
    assert (output_dir / "nested-dataset_tsoi").exists()

    metadata = json.loads((output_dir / "simple_tsoi").read_text().splitlines()[0])
    assert metadata["T"] == 1
    assert sorted(metadata["voters"]) == ["alice", "bob"]


def test_convert_all_creates_output_dir_if_missing(tmp_path: Path, monkeypatch) -> None:
    real_data_dir = tmp_path / "real_data"
    output_dir = real_data_dir / "json_datasets"
    monkeypatch.setattr(convert_module, "REAL_DATA_DIR", real_data_dir)
    monkeypatch.setattr(convert_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(convert_module, "DATASET_DIRS", ["simple_tsoi"])
    _write_source_dataset(real_data_dir, "simple_tsoi")

    assert not output_dir.exists()
    convert_all()
    assert output_dir.is_dir()


def test_convert_all_skips_a_dataset_whose_source_files_are_corrupted(
    tmp_path: Path, monkeypatch
) -> None:
    real_data_dir = tmp_path / "real_data"
    output_dir = real_data_dir / "json_datasets"
    monkeypatch.setattr(convert_module, "REAL_DATA_DIR", real_data_dir)
    monkeypatch.setattr(convert_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(convert_module, "DATASET_DIRS", ["good_tsoi", "corrupted_tsoi"])

    _write_source_dataset(real_data_dir, "good_tsoi")
    corrupted_dir = real_data_dir / "corrupted_tsoi"
    corrupted_dir.mkdir(parents=True)
    (corrupted_dir / "round_1.tsoi").write_text("not a valid tsoi file at all\n")

    convert_all()

    assert (output_dir / "good_tsoi").exists()
    assert not (output_dir / "corrupted_tsoi").exists()
