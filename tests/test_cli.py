"""The command line runs a small scan and a two-replica ensemble."""

import subprocess
import sys

import numpy as np
import pandas as pd

from spad.ensemble import ensemble_job, load_record


def test_help():
    completed = subprocess.run(
        [sys.executable, "-m", "spad.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "qscan" in completed.stdout
    assert "not a best-q rule" in completed.stdout


def test_ensemble_job_is_seed_stable(tmp_path):
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "lat": 51.0 + rng.normal(0, 0.05, 24),
            "lon": 157.0 + rng.normal(0, 0.05, 24),
            "depth": 25.0 + rng.normal(0, 1.0, 24),
            "Aftershock": False,
        }
    )
    path = tmp_path / "tiny.csv"
    frame.to_csv(path, index=False)
    spec = {
        "kind": "shift50",
        "i": 0,
        "seed": 1000,
        "catalog": str(path),
        "depth": "catalog",
        "q_values": np.array([-1.5, -1.0]),
        "real_r": [5.0, 8.0],
    }
    first = ensemble_job({**spec, "path": str(tmp_path / "a.pkl")})
    second = ensemble_job({**spec, "path": str(tmp_path / "b.pkl")})
    left = load_record(first)
    right = load_record(second)
    assert np.array_equal(left["lat"], right["lat"])
    assert np.array_equal(left["depth"], right["depth"])
    assert left["solutions"][0]["n_old"] == right["solutions"][0]["n_old"]
    assert left["solutions"][0]["r"] == right["solutions"][0]["r"]
    assert "solutions_fixed_r" in left
    assert left["solutions_fixed_r"][0]["r"] == 5.0
    assert left["solutions_fixed_r"][1]["r"] == 8.0
    boot = {
        "kind": "boot",
        "i": 0,
        "seed": 4000,
        "catalog": str(path),
        "depth": "catalog",
        "q_values": np.array([-1.0]),
        "path": str(tmp_path / "boot.pkl"),
    }
    record = load_record(ensemble_job(boot))
    assert record["idx"] is not None
    assert len(record["idx"]) == round(0.8 * 24)
    assert "solutions_fixed_r" not in record
