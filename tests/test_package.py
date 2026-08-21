from rag_quality_lab import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.2.0"
