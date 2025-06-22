import importlib


def test_tests_package_exists():
    spec = importlib.util.find_spec("tests")
    assert spec is not None
