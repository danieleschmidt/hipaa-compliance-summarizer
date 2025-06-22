from hipaa_summarizer import parse_requirements


def test_pytest_listed_in_requirements():
    reqs = parse_requirements()
    assert any(r.startswith("pytest") for r in reqs)


def test_edge_case_null_input():
    reqs = parse_requirements(None)
    assert isinstance(reqs, list)
