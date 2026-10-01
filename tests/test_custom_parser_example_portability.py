import builtins
import importlib.util
import tempfile
from pathlib import Path


def _load_example_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "custom_parser_example.py"
    spec = importlib.util.spec_from_file_location("custom_parser_example", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_parser_example_uses_os_temp_dir(monkeypatch):
    module = _load_example_module()
    real_open = builtins.open
    opened_paths = []

    def fake_open(path, mode="r", *args, **kwargs):
        opened_paths.append(str(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    module.main()

    assert opened_paths
    assert all(Path(path).parent == Path(tempfile.gettempdir()) for path in opened_paths)
    assert any(Path(path).name == "sample_log.csv" for path in opened_paths)
