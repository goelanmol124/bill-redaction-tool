import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_synthetic", ROOT / "examples" / "make_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def synthetic():
    return _load_generator()


@pytest.fixture(scope="session")
def bundle(tmp_path_factory, synthetic):
    return synthetic.build_bundle(tmp_path_factory.mktemp("data") / "bundle.pdf")


@pytest.fixture(scope="session")
def example_config():
    from bill_redact import load_config
    return load_config(ROOT / "examples" / "config.yaml")
