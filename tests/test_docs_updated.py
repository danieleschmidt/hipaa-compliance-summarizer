from pathlib import Path


def test_usage_example_present():
    readme = Path("README.md").read_text()
    assert "DocumentType" in readme and "HIPAAProcessor" in readme
