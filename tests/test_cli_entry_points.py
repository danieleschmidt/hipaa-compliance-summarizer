import tomllib


def test_cli_scripts_defined():
    with open("pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("hipaa-summarize")
    assert scripts.get("hipaa-batch-process")
    assert scripts.get("hipaa-compliance-report")
