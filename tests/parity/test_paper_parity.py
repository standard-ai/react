"""The library reproduces the paper's per-event results (opt-in, about 5 minutes).

Runs tests/parity/run_parity.sh, which needs local data and the reproducibility repository;
see that script for its inputs. Enable with REACT_PARITY=1 (REACT_PARITY_TRACKS sets the
subset size, at most 20).
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("REACT_PARITY") != "1", reason="paper parity check: set REACT_PARITY=1"
)


def test_matches_paper_run():
    script = Path(__file__).with_name("run_parity.sh")
    n = os.environ.get("REACT_PARITY_TRACKS", "5")
    proc = subprocess.run([str(script), n], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout
