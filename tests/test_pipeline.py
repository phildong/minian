import os
import subprocess

import numpy as np

# import pytest
import yaml
from distributed import Client, LocalCluster

from minian.pipeline import minian_process
from minian.utilities import TaskAnnotation, open_minian


# @pytest.mark.flaky(reruns=3)
def test_pipeline_notebook():
    os.makedirs("artifact", exist_ok=True)
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--output",
        "artifact/pipeline.ipynb",
        "--execute",
        "pipeline.ipynb",
    ]
    subprocess.run(args, check=True)
    minian_ds = open_minian("./demo_movies/minian")
    assert minian_ds.sizes["frame"] == 2000
    assert minian_ds.sizes["height"] == 480
    assert minian_ds.sizes["width"] == 752
    assert minian_ds.sizes["unit_id"] == 282
    assert (
        minian_ds["motion"].sum("frame").values.astype(int) == np.array([423, -239])
    ).all()
    assert int(minian_ds["max_proj"].sum().compute()) == 1501505
    assert int(minian_ds["C"].sum().compute()) == 478444
    assert int(minian_ds["S"].sum().compute()) == 3943
    assert int(minian_ds["A"].sum().compute()) == 41755
    assert os.path.exists("./demo_movies/minian_mc.mp4")
    assert os.path.exists("./demo_movies/minian.mp4")


# @pytest.mark.flaky(reruns=3)
def test_pipeline():
    # dask cluster init
    cluster = LocalCluster(
        n_workers=8,
        memory_limit="5GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
        local_directory=".",
    )
    # annt_plugin = TaskAnnotation()
    # cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    # load parameters
    with open("./default_params.yaml") as yf:
        param = yaml.full_load(yf)
    minian_ds, plots = minian_process(
        dpath="./demo_movies",
        intpath="./minian_intermediate",
        param=param,
        video_path="./demo_movies/minian.mp4",
    )
    assert minian_ds.sizes["frame"] == 2000
    assert minian_ds.sizes["height"] == 480
    assert minian_ds.sizes["width"] == 752


if __name__ == "__main__":
    test_pipeline()
