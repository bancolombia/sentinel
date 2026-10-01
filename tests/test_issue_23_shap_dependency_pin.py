from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shap_optional_dependency_is_pinned_to_a_compatible_version():
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"shap>=0.44.0,<0.52.0"' in pyproject_text
