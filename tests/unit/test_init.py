from importlib.metadata import version

import realtime_agentic_site_builder


def test_package_defines_version() -> None:
    assert realtime_agentic_site_builder.__version__


def test_version_matches_distribution_metadata() -> None:
    expected = version("realtime-agentic-site-builder")
    assert realtime_agentic_site_builder.__version__ == expected
