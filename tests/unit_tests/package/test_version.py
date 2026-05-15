from pathlib import Path

import tomllib

import upsonic


def test_package_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert upsonic.__version__ == data["project"]["version"]
