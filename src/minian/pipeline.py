import io
import os
import shutil
import traceback
from contextlib import redirect_stdout
from datetime import datetime

import dask as da
import dask.array as darr
import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from bokeh.plotting import save as bk_savefig
from bokeh.resources import CDN
from distributed import LocalCluster
from holoviews import opts
from scipy.interpolate import interp1d

from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_background,
    update_spatial,
    update_temporal,
)
from minian.initialization import initA, initC, pnr_refine, seeds_init, seeds_merge
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.stripe_correction import label_good_frames, ripple_correction
from minian.utilities import get_optimal_chk, load_videos, save_minian
from minian.visualization import (
    generate_videos,
    plotA_contour,
    visualize_motion,
    visualize_seeds,
)


def minian_process(
    dpath,
    intpath,
    param,
    return_stage=None,
    varr=None,
    flip=False,
    video_path=None,
    motion=None,
):
    # setup
    dpath = os.path.abspath(os.path.expanduser(dpath))
    intpath = os.path.abspath(os.path.expanduser(intpath))
    shutil.rmtree(intpath, ignore_errors=True)
    # plotting options
    plots = dict()
    opts_crv = opts.Curve(**{"frame_width": 700, "aspect": 2})
    opts_tr = opts.Image(**{"frame_width": 700, "aspect": 2, "cmap": "viridis"})
    opts_im = opts.Image(
        **{
            "frame_width": 400,
            "aspect": 1,
            "cmap": "gray",
            "colorbar": True,
        }
    )
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    if varr is None:
        varr = load_videos(dpath, **param["load_videos"])
    else:
        del varr.encoding["chunks"]
    chk, _ = get_optimal_chk(varr, dtype=float)
    if flip:
        varr = xr.apply_ufunc(
            darr.flip,
            varr,
            input_core_dims=[["frame", "height", "width"]],
            output_core_dims=[["frame", "height", "width"]],
            kwargs={"axis": 2},
            dask="allowed",
        )
    subset = param.get("subset")
    if subset is not None:
        for d, sub in subset.items():
            if isinstance(sub, tuple):
                subset[d] = slice(*sub)
    varr = varr.sel(subset)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    if param["stripe_corr"] is not None:
        good_frame = label_good_frames(varr, **param["stripe_corr"])
        varr = varr.sel(frame=good_frame)
    if param["ripple_corr"] is not None:
        varr = ripple_correction(varr, **param["ripple_corr"])
    varr_ref = varr
    # example frame
    smp_fm = np.random.choice(varr_ref.coords["frame"], 1000)
    exp_fm = varr_ref.sel(frame=smp_fm).max("frame").rename("exp_fm").compute()
    # preprocessing
    if param["glow_rm"] == "min":
        varr_min = varr_ref.min("frame").compute()
        varr_ref = varr_ref - varr_min
    else:
        varr_ref = (
            remove_background(varr_ref.astype(float), **param["glow_rm"])
            .clip(0, 255)
            .astype(np.uint8)
        )
    varr_ref = denoise(varr_ref, **param["denoise"])
    varr_ref = remove_background(varr_ref, **param["background_removal"])
    varr_ref = save_minian(
        varr_ref.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename(
            "varr_ref"
        ),
        dpath=intpath,
        overwrite=True,
    )
    if return_stage == "pre-processing":
        return varr_ref, plots
    # motion-correction
    if motion is None:
        motion = estimate_motion(varr_ref, **param["estimate_motion"])
    else:
        motion = xr.DataArray(
            resample_motion(
                motion.values,
                f_org=motion.coords["frame"].values,
                f_new=varr_ref.coords["frame"].values,
            ),
            dims=motion.dims,
            coords={
                "frame": varr_ref.coords["frame"].values,
                "shift_dim": motion.coords["shift_dim"].values,
            },
        ).sel(frame=varr_ref.coords["frame"].values)
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), intpath, overwrite=True
    )
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )
    plots["motion"] = (
        hv.Image(
            varr_ref.max("frame").compute().astype(np.float32),
            ["width", "height"],
            label="before_mc",
        ).opts(opts_im)
        + hv.Image(
            Y_hw_chk.max("frame").compute().astype(np.float32),
            ["width", "height"],
            label="after_mc",
        ).opts(opts_im)
        + visualize_motion(motion.compute()).opts(opts_crv)
    ).cols(2)
    if return_stage == "motion-correction":
        return xr.merge([motion, Y_fm_chk, Y_hw_chk]), plots
    # initilization
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), intpath, overwrite=True
    ).compute()
    seeds = seeds_init(Y_fm_chk, **param["seeds_init"])
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param["pnr_refine"])
    seeds_final = seeds[seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param["seeds_merge"])
    A_init = initA(
        Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param["initialize"]
    )
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    try:
        A, C = unit_merge(A_init, C_init, **param["init_merge"])
    except (KeyError, ValueError):
        A, C = A_init, C_init
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)
    plots["init"] = (
        plotA_contour(A, max_proj).opts(opts_im).relabel("Initial Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).rename("C").compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Initial Temporal Components")
        + hv.Image(
            b.rename("b").compute().astype(np.float32), kdims=["width", "height"]
        )
        .opts(opts_im)
        .relabel("Initial Background Sptial")
        + hv.Curve(f.rename("f").compute(), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Initial Background Temporal")
        + visualize_seeds(max_proj, seeds_final, "mask_mrg")
        .opts(opts_im)
        .relabel("Final Seeds")
    ).cols(2)
    if return_stage == "initialization":
        return xr.merge([A, C, b, f]), plots
    # cnmf
    sn_spatial = get_noise_fft(Y_hw_chk, **param["get_noise"])
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
    ## first iteration
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, C, sn_spatial, **param["first_spatial"]
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    plots["first-spatial"] = (
        plotA_contour(A, max_proj).opts(opts_im).relabel("Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Temporal Trace")
        + hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"])
        .opts(opts_im)
        .relabel("Background Spatial")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Background Temporal")
    ).cols(2)
    if return_stage == "first-spatial":
        return xr.merge([A, C, b, f]), plots
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param["first_temporal"]
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)
    plots["first-temporal"] = (
        plotA_contour(A, max_proj).opts(opts_im).relabel("Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Temporal Trace")
        + hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"])
        .opts(opts_im)
        .relabel("Background Spatial")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Background Temporal")
    ).cols(2)
    if return_stage == "first-temporal":
        return xr.merge([A, C, S, b, f]), plots
    ## merge
    try:
        A_mrg, C_mrg, [sig_mrg] = unit_merge(
            A, C, [C + b0 + c0], **param["first_merge"]
        )
    except KeyError:
        A_mrg, C_mrg, sig_mrg = A, C, C + b0 + c0
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)
    plots["first-merge"] = (
        plotA_contour(A, max_proj).opts(opts_im).relabel("Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Temporal Trace")
        + hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"])
        .opts(opts_im)
        .relabel("Background Spatial")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Background Temporal")
    ).cols(2)
    ## second iteration
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, sig, sn_spatial, **param["second_spatial"]
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    plots["second-spatial"] = (
        plotA_contour(A, max_proj).opts(opts_im).relabel("Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Temporal Trace")
        + hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"])
        .opts(opts_im)
        .relabel("Background Spatial")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Background Temporal")
    ).cols(2)
    if return_stage == "second-spatial":
        return xr.merge([A, C, S, b, f]), plots
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param["second_temporal"]
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)
    plots["final"] = (
        plotA_contour(A, max_proj).opts(opts_im).relabel("Spatial Footprints")
        + hv.Image(
            C.sel(frame=slice(0, None, 10)).compute().astype(np.float32),
            kdims=["frame", "unit_id"],
        )
        .opts(opts_tr)
        .relabel("Temporal Trace")
        + hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"])
        .opts(opts_im)
        .relabel("Background Spatial")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(opts_crv)
        .relabel("Background Temporal")
    ).cols(2)
    # generate video
    if video_path is not None:
        vpath, vname = os.path.split(video_path)
        generate_videos(varr, Y_fm_chk, A=A, C=C_chk, vpath=vpath, vname=vname)
    result_ds = xr.merge(
        [
            A.rename("A"),
            C.rename("C"),
            YrA.rename("YrA"),
            S.rename("S"),
            c0.rename("c0"),
            b0.rename("b0"),
            b.rename("b"),
            f.rename("f"),
            motion,
            max_proj,
            exp_fm,
        ]
    )
    return result_ds, plots


def resample_motion(motion, nsmp=None, f_org=None, f_new=None):
    if f_org is None:
        f_org = np.linspace(0, 1, motion.shape[0], endpoint=True)
    if f_new is None:
        f_new = np.linspace(0, 1, nsmp, endpoint=True)
    motion_ret = np.zeros((len(f_new), 2))
    for i in range(2):
        motion_ret[:, i] = interp1d(
            f_org, motion[:, i], bounds_error=False, fill_value="extrapolate"
        )(f_new)
    return motion_ret


def minian_process_batch(
    ss_csv: str,
    id_cols: list = ["animal", "session"],
    dat_col: str = "data",
    param_col: str = None,
    dat_path: str = "./data",
    param_path: str = "./process/params",
    out_path: str = "./intermediate/processed",
    fig_path: str = "./figs/processed",
    err_path: str = "./process/errs",
    int_path: str = "./process/minian_int",
    worker_path: str = "./process/dask-worker-space",
    skip_existing: bool = True,
    raise_err: bool = False,
    cluster_kws: dict = dict(),
):
    # book-keeping
    hv.extension("bokeh")
    hv.config.image_rtol = 100
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = int_path
    np.seterr(all="ignore")
    da.config.set(
        **{
            "distributed.comm.timeouts.connect": "60s",
            "distributed.comm.timeouts.tcp": "60s",
        }
    )
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(err_path, exist_ok=True)
    os.makedirs(int_path, exist_ok=True)
    # read session map
    ssmap = pd.read_csv(ss_csv)
    # process loop
    for id_vars, ss_row in ssmap.set_index(id_cols).iterrows():
        dp = ss_row[dat_col]
        fname = "-".join(id_vars)
        opath = os.path.join(out_path, "{}.nc".format(fname))
        if skip_existing and os.path.exists(opath):
            print("skipping {}".format(dp))
            continue
        else:
            print("processing {}".format(dp))
        # determine parameters
        pfiles = ["generic.yaml"]
        if param_col is not None and pd.notnull(ss_row[param_col]):
            pfiles.extend(ss_row[param_col].split(";"))
        param = dict()
        for pf in pfiles:
            with open(os.path.join(param_path, pf), "r") as yf:
                p = yaml.full_load(yf)
            param.update(p)
        print("using params: {}".format(pfiles))
        # start cluster
        started = False
        while not started:
            try:
                clst_kws = {
                    "n_workers": 8,
                    "memory_limit": "5GB",
                    "resources": {"MEM": 1},
                    "threads_per_worker": 2,
                    "dashboard_address": "0.0.0.0:12345",
                    "local_directory": worker_path,
                }
                clst_kws.update(cluster_kws)
                cluster = LocalCluster(**clst_kws)
                client = cluster.get_client()
                started = True
            except:
                cluster.close()
        # start process
        shutil.rmtree(int_path, ignore_errors=True)
        try:
            tstart = datetime.now()
            with redirect_stdout(io.StringIO()):
                result_ds, plots = minian_process(
                    dpath=os.path.join(dat_path, dp),
                    intpath=int_path,
                    param=param,
                    video_path=os.path.join(out_path, "{}.mp4".format(fname)),
                )
            tend = datetime.now()
            print("minian success: {}".format(dp))
            print("time: {}".format(tend - tstart))
        except Exception as err:
            print("minian failed: {}".format(dp))
            with open(os.path.join(err_path, "{}.log".format(fname)), "w") as txtf:
                traceback.print_exception(None, err, err.__traceback__, file=txtf)
            if raise_err:
                raise err
            client.close()
            cluster.close()
            continue
        result_ds = (
            result_ds.assign_coords({k: v for k, v in zip(id_cols, id_vars)})
            .expand_dims(id_cols)
            .compute()
        )
        result_ds.to_netcdf(
            os.path.join(out_path, "{}.nc".format(fname)), format="NETCDF4"
        )
        for plt_name, plt in plots.items():
            plt_name = "-".join([fname, plt_name])
            bk_savefig(
                hv.render(plt),
                os.path.join(fig_path, "{}.html".format(plt_name)),
                title=plt_name,
                resources=CDN,
            )
        client.close()
        cluster.close()
