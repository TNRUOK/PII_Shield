"""
tests/conftest.py
-----------------
Shared pytest fixtures.
"""
import sys
from pathlib import Path

# Ensure pii_shield root is on PYTHONPATH for all tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import yaml


@pytest.fixture(scope="session")
def config():
    """Load the real config.yaml once per test session."""
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def synthetic_dir():
    return ROOT / "synthetic"


@pytest.fixture(scope="session")
def gt_dir():
    return ROOT / "synthetic" / "ground_truth"
