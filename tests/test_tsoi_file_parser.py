from pathlib import Path

from src.real_data_tools.tsoi_parser import parse_tsoi
from src.synthetic_data_tools.profiles import ApprovalProfile


def test_parse_tsoi():
    # Get a tsoi file from the real_data directory
    tsoi_file = (
        Path(__file__).parent.parent
        / "real_data"
        / "spotify"
        / "daily_tsoi"
        / "top200_20170101.tsoi"
    )

    parsed_profile = parse_tsoi(tsoi_file)

    assert isinstance(parsed_profile, ApprovalProfile)
