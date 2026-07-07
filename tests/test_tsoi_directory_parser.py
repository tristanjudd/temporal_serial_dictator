from pathlib import Path

from src.real_data_tools.tsoi_parser import parse_tsoi_directory
from src.synthetic_data_tools.profiles import ApprovalProfile


def test_tsoi_directory_parser():
    # Get a tsoi directory from the real_data directory
    tsoi_directory = (
        Path(__file__).parent.parent / "real_data" / "eurovision_song_contest_tsoi"
    )

    parsed_profiles = parse_tsoi_directory(tsoi_directory)

    assert isinstance(parsed_profiles, list)
    assert len(parsed_profiles) > 0
    assert all(isinstance(profile, ApprovalProfile) for profile in parsed_profiles)
