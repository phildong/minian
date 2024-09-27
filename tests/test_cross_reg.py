import os
import subprocess

import pandas as pd


def test_cross_reg_notebook():
    os.makedirs("artifact", exist_ok=True)
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--output",
        "artifact/cross-registration.ipynb",
        "--execute",
        "cross-registration.ipynb",
    ]
    subprocess.run(args, check=True)
    assert os.path.exists("./demo_data/shiftds.nc")
    assert os.path.exists("./demo_data/cents.pkl")
    assert os.path.exists("./demo_data/mappings.pkl")
    cents = pd.read_pickle("./demo_data/cents.pkl")
    mappings = pd.read_pickle("./demo_data/mappings.pkl")
    assert len(cents) == 508
    assert int(cents["height"].sum()) == 99091
    assert int(cents["width"].sum()) == 213627
    assert len(mappings) == 431
    assert mappings[("group", "group")].value_counts().to_dict() == {
        ("session2",): 182,
        ("session1",): 172,
        ("session1", "session2"): 77,
    }
