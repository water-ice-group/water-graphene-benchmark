#!/usr/bin/env python3
"""
All-vs-all functional benchmarking across every property in preprocess-all.ipynb.

The notebook scores every functional against ONE reference (rpa_name="RPA-QZ",
with a commented-out "revPBE-D3-BJ" alternative). This script keeps the metrics
bit-for-bit but sweeps the reference over *every* functional, so the output is a
reference x test matrix per property instead of a single column.

Properties (same definitions, same defaults as the notebook cells):

  density      profile d per species (O, H), the species-average "combined",
               the contact-layer ratio r = rho2/rho1 d-score, and their
               "density_and_ratio" fusion.
  friction     plateau lambda (scalar), the Green-Kubo running integral
               lambda(tau) (curve), and their 0.5*(s+c) combination.
  orientation  count-weighted similarity of P(cos theta) over a slab-width
               window (default z <= 11 A, weights = reference N(z)).
  rdf          O-O / H-O / H-H, interfacial (graphene-water) + bulk (pure
               water), plus per-condition and overall combinations.
  vdos         O / H, interfacial + bulk, same combination structure as rdf.
  sfg          three physical descriptors (peak positions, intensity ratio,
               H-bond width) per mode, averaged into a per-mode d and a
               cross-mode "combined".

Every d-score is in [0, 1] (1 = identical to the reference). For each
reference, kappa = (1 - d) / sigma_d is computed across the functionals scored
against THAT reference, exactly as the notebook does -- so kappa is always
relative to its own column of the matrix and is not comparable across
references without care. Raw scalars (lambda, rho2/rho1) additionally get the
directly interpretable kappa_abs = |Delta| / sigma(value).

Data is loaded and reduced ONCE per functional (the raw orientation pkls are
~31 MB each, so the notebook's habit of re-reading the reference for every test
functional is not viable at N^2). The reduced form is cached to a pickle.

Usage
-----
    python benchmark_all_references.py                      # everything
    python benchmark_all_references.py -p density friction  # subset
    python benchmark_all_references.py --references RPA-QZ  # notebook behaviour
    python benchmark_all_references.py --refresh-cache      # re-read raw data

Outputs (under --out-dir, default ./benchmark_all_references):
    pairwise_scores.csv       tidy: reference, test, property, metric, d, kappa, rank
    pairwise_values.csv       raw scalars (lambda, rho2/rho1) + signed errors + kappa_abs
    reference_consensus.csv   per functional: mean/std rank and d across all references
    matrices/<prop>__<metric>.csv   N x N d-score matrix (rows = reference)
    scores_experimental.csv  sfg only: every functional scored against the fixed
                              experimental (Wang et al. 2024) HD-SFG reference,
                              same descriptor machinery as the functional-vs-
                              functional sfg scoring (skipped if the digitized
                              experimental data file is not found)
    summary.pkl               everything above, unrounded
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict

import numpy as np


# ===========================================================================
# Defaults (copied from the notebook's __main__ blocks)
# ===========================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INTERF_ROOT = os.path.join(HERE, "..", "..", "17_FINAL_PROD_out")
DEFAULT_BULK_ROOT = os.path.join(HERE, "..", "..", "12_analyse_final_prods")

# Digitized experimental HD-SFG spectra (Wang et al., Angew. Chem. Int. Ed. 63,
# e202319503 (2024), Fig. 1D), as reduced/shared under 03_sfg/.
DEFAULT_SFG_EXPT_PATH = os.path.join(
    HERE, "..", "03_sfg", "data-yongkang-himself",
    "Air_water_and_graphene_water_Figure_1.txt")

ALL_PROPERTIES = ("density", "friction", "orientation", "rdf", "vdos", "sfg")

CFG = {
    "system": "19A",
    "n_tag": "n100",

    # -- density ----------------------------------------------------------
    "density_species": ("O", "H"),
    "density_pkl_glob": "density_*.pkl",
    "ratio_species": "O",
    "starting_c": 7.5,
    "x1": (2.5, 4.0),
    "x2": (4.0, 7.5),

    # -- friction ---------------------------------------------------------
    "n_agg": "n100_aggregate",
    "scalar_subdir": "flexible",
    "curve_subdir": "flexible",
    "curve_datafile": "lambda_fft_vs_tau_average.dat",
    "curve_window": (0.0, None),

    # -- orientation ------------------------------------------------------
    "orientation_pkl_glob": "orientation_*.pkl",
    "weight_source": "ref",          # "ref" | "both" | "none"
    "z_min": None,
    "z_max": 11.0,

    # -- rdf --------------------------------------------------------------
    "rdf_species": ("O-O", "H-O", "H-H"),
    "rdf_pkl_glob": "rdfs_*.pkl",

    # -- vdos -------------------------------------------------------------
    "vdos_species": ("O", "H"),
    "vdos_pkl_glob": "vdos_*.pkl",

    # -- sfg --------------------------------------------------------------
    "sfg_modes": ("mode2", "mode3"),
    "sfg_window": (2500, 4100),
    "sfg_width_method": "fwhm",      # "equivalent" | "fwhm" | "integral"
    "sfg_width_window": (2800, None),
    "sfg_strict_components": False,
    "sfg_renormalise_ref": True,
    "sfg_expt_path": DEFAULT_SFG_EXPT_PATH,
    # column index within the experimental csv (after the "xaxis" column) for
    # each mode: 1 = "Air/water", 2 = "Air/graphene/water"
    "sfg_expt_columns": {"mode2": 1, "mode3": 2},
}

CACHE_VERSION = 3


# ===========================================================================
# Generic helpers
# ===========================================================================

def _summary_stats(vals):
    """Mean +/- SEM over runs, NaNs dropped. Same convention as the notebook."""
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"mean": np.nan, "sem": np.nan, "n_runs": 0}
    mean = float(np.mean(a))
    sem = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {"mean": mean, "sem": sem, "n_runs": int(len(a))}


def _nanmean_rows(rows):
    """Column-wise nanmean of a list of equal-length sequences."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(rows, axis=0), axis=0)


def bray_curtis(ref, test):
    """d = 1 - sum|ref - test| / sum(|ref| + |test|), test interpolated onto ref."""
    x_ref, y_ref = np.asarray(ref[0], float), np.asarray(ref[1], float)
    x_test, y_test = np.asarray(test[0], float), np.asarray(test[1], float)
    y_i = np.interp(x_ref, x_test, y_test)
    num = np.sum(np.abs(y_ref - y_i))
    den = np.sum(np.abs(y_ref) + np.abs(y_i))
    if den == 0:
        return np.nan
    return 1.0 - num / den


def scalar_similarity(a, b):
    num = abs(a - b)
    den = abs(a) + abs(b)
    if den == 0:
        return np.nan
    return 1.0 - num / den


def scalar_similarity_sem(a, b, sa, sb):
    den = abs(a) + abs(b)
    if den == 0:
        return np.nan
    return float(np.sqrt(sa ** 2 + sb ** 2) / den)


def average_curves(curves):
    """(x, mean, sem, n) over a list of (x, y), all interpolated onto curves[0][0]."""
    x_ref = np.asarray(curves[0][0], dtype=float)
    ys = []
    for x, y in curves:
        ys.append(np.interp(x_ref, np.asarray(x, float), np.asarray(y, float)))
    ys = np.stack(ys, axis=0)
    mean = ys.mean(axis=0)
    n = ys.shape[0]
    sem = (ys.std(axis=0, ddof=1) / np.sqrt(n) if n > 1
           else np.zeros_like(mean))
    return x_ref, mean, sem, n


def record(mean, sem, n_runs, **extra):
    rec = {"d": float(mean) if np.isfinite(mean) else np.nan,
           "d_sem": float(sem) if np.isfinite(sem) else np.nan,
           "n_runs": int(n_runs)}
    rec.update(extra)
    return rec


def _unwrap(data, keys):
    """Unwrap a {'test_x': {...}, 'ref_x': {...}} pkl to the inner dict."""
    if isinstance(data, dict):
        for k in keys:
            if k in data and isinstance(data[k], dict):
                return data[k]
    return data


def _interf_glob(root, fn, system, n_tag, pkl_glob):
    return sorted(glob.glob(os.path.join(
        root, fn, "gra-runs", "frozen", system, "run-*", n_tag, pkl_glob)))


def _bulk_pkl(bulk_root, fn, n_tag, stem):
    """<bulk_root>/<fn>/bulk-runs/<n_tag>/<stem>_<n_tag>.pkl, with a loose fallback."""
    d = os.path.join(bulk_root, fn, "bulk-runs", n_tag)
    hits = sorted(glob.glob(os.path.join(d, f"{stem}_{n_tag}.pkl")))
    if not hits:
        hits = sorted(glob.glob(os.path.join(d, f"{stem}_*.pkl")))
    return hits[0] if hits else None


# ===========================================================================
# DENSITY  (notebook cell 0)
# ===========================================================================

def load_density(interf_root, bulk_root, fn, cfg):
    paths = _interf_glob(interf_root, fn, cfg["system"], cfg["n_tag"],
                         cfg["density_pkl_glob"])
    runs = []
    for p in paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        data = _unwrap(data, ("test_density", "ref_density", "density"))
        runs.append({sp: (np.asarray(v[0], float), np.asarray(v[1], float))
                     for sp, v in data.items()})
    return {"runs": runs} if runs else None


def _ratio_from_mean_profile(z, rho_mean, rho_sem, x1, x2):
    """Peak heights of the averaged profile in two z windows, and their ratio."""
    m1 = (z >= x1[0]) & (z <= x1[1])
    m2 = (z >= x2[0]) & (z <= x2[1])
    if not np.any(m1) or not np.any(m2):
        return None
    i1 = int(np.argmax(rho_mean[m1]))
    i2 = int(np.argmax(rho_mean[m2]))
    r1 = float(rho_mean[m1][i1])
    r2 = float(rho_mean[m2][i2])
    if r1 == 0:
        return None
    s1 = float(rho_sem[m1][i1]) if rho_sem is not None else 0.0
    s2 = float(rho_sem[m2][i2]) if rho_sem is not None else 0.0
    s1 = s1 if np.isfinite(s1) else 0.0
    s2 = s2 if np.isfinite(s2) else 0.0
    ratio = r2 / r1
    if r2 != 0:
        ratio_se = abs(ratio) * np.sqrt((s2 / r2) ** 2 + (s1 / r1) ** 2)
    else:
        ratio_se = np.nan
    return {"rho1": r1, "rho1_sem": s1, "rho2": r2, "rho2_sem": s2,
            "ratio": ratio, "ratio_sem": ratio_se}


def _density_ratio(data, cfg):
    sp = cfg["ratio_species"]
    curves = [r[sp] for r in data["runs"] if sp in r]
    if not curves:
        return None
    z, mean, sem, _ = average_curves(curves)
    return _ratio_from_mean_profile(z - cfg["starting_c"], mean, sem,
                                    cfg["x1"], cfg["x2"])


def ref_density(data, cfg):
    species = cfg["density_species"]
    ref_profiles = {}
    for sp in species:
        curves = [r[sp] for r in data["runs"] if sp in r]
        if not curves:
            continue
        z, mean, _, _ = average_curves(curves)
        ref_profiles[sp] = (z, mean)
    ratio = _density_ratio(data, cfg)
    if not ref_profiles or ratio is None or not np.isfinite(ratio["ratio"]):
        return None
    return {"profiles": ref_profiles, "ratio": ratio}


def score_density(ctx, data, cfg):
    species = cfg["density_species"]
    out = {}

    per_species = {sp: [] for sp in species}
    for run in data["runs"]:
        for sp in species:
            if sp not in run or sp not in ctx["profiles"]:
                per_species[sp].append(np.nan)
            else:
                per_species[sp].append(bray_curtis(ctx["profiles"][sp], run[sp]))

    for sp in species:
        s = _summary_stats(per_species[sp])
        out[sp] = record(s["mean"], s["sem"], s["n_runs"])

    combined = _nanmean_rows([per_species[sp] for sp in species])
    s = _summary_stats(combined.tolist())
    out["combined"] = record(s["mean"], s["sem"], s["n_runs"])

    # -- contact-layer ratio ------------------------------------------------
    ref_ratio = ctx["ratio"]
    test_ratio = _density_ratio(data, cfg)
    n_runs = len(data["runs"])
    if test_ratio is None or not np.isfinite(test_ratio["ratio"]):
        out["ratio"] = record(np.nan, np.nan, n_runs)
    else:
        r, rs = test_ratio["ratio"], test_ratio["ratio_sem"]
        d_r = scalar_similarity(ref_ratio["ratio"], r)
        d_r_sem = scalar_similarity_sem(ref_ratio["ratio"], r,
                                        ref_ratio["ratio_sem"], rs)
        signed = float(r - ref_ratio["ratio"])
        out["ratio"] = record(
            d_r, d_r_sem if np.isfinite(d_r_sem) else 0.0, n_runs,
            value=r, value_sem=rs, ref_value=ref_ratio["ratio"],
            signed_err=signed, abs_err=abs(signed),
            rel_err=(abs(signed) / abs(ref_ratio["ratio"])
                     if ref_ratio["ratio"] != 0 else np.nan),
            rho1=test_ratio["rho1"], rho2=test_ratio["rho2"],
        )

    # -- profile x ratio fusion --------------------------------------------
    d_p, d_p_sem = out["combined"]["d"], out["combined"]["d_sem"]
    d_r, d_r_sem = out["ratio"]["d"], out["ratio"]["d_sem"]
    if np.isfinite(d_p) and np.isfinite(d_r):
        out["density_and_ratio"] = record(
            0.5 * (d_p + d_r),
            0.5 * float(np.sqrt(d_p_sem ** 2 + d_r_sem ** 2)),
            out["combined"]["n_runs"])
    else:
        out["density_and_ratio"] = record(np.nan, np.nan,
                                          out["combined"]["n_runs"])
    return out


# ===========================================================================
# FRICTION  (notebook cell 1)
# ===========================================================================

def load_friction(interf_root, bulk_root, fn, cfg):
    def agg(subdir):
        return os.path.join(interf_root, fn, "gra-runs", subdir,
                            cfg["system"], cfg["n_agg"])

    sdir = agg(cfg["scalar_subdir"])
    cdir = agg(cfg["curve_subdir"])
    if not os.path.isdir(sdir) or not os.path.isdir(cdir):
        return None

    try:
        mean = float(np.asarray(np.load(
            os.path.join(sdir, "lambda_static_mean.npy"))).item())
        sem = float(np.asarray(np.load(
            os.path.join(sdir, "lambda_static_sem.npy"))).item())
    except (FileNotFoundError, ValueError):
        return None

    n_blocks = 0
    try:
        n_blocks = int(np.asarray(
            np.load(os.path.join(sdir, "lambda_static_blocks.npy"))).size)
    except Exception:
        pass

    cpath = os.path.join(cdir, cfg["curve_datafile"])
    if not os.path.isfile(cpath):
        return None
    arr = np.loadtxt(cpath, skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] < 2 or arr.shape[1] < 3:
        return None

    return {
        "scalar": {"mean": mean, "sem": sem, "n_blocks": n_blocks},
        "curve": {"tau": arr[:, 0].astype(float),
                  "lam": arr[:, 1].astype(float),
                  "sem": arr[:, 2].astype(float)},
    }


def _curve_window_mask(tau, window):
    if window is None:
        return np.ones_like(tau, dtype=bool)
    lo, hi = window
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    mask = (tau >= lo) & (tau <= hi)
    return mask if mask.any() else np.ones_like(tau, dtype=bool)


def _lambda_curve_similarity(ref_curve, test_curve, window):
    """Bray-Curtis on lambda(tau) with linearised SEM propagation."""
    tau_ref, lam_ref, sem_ref = (ref_curve["tau"], ref_curve["lam"],
                                 ref_curve["sem"])
    lam_i = np.interp(tau_ref, test_curve["tau"], test_curve["lam"])
    sem_i = np.interp(tau_ref, test_curve["tau"], test_curve["sem"])

    mask = _curve_window_mask(tau_ref, window)
    a, b = lam_ref[mask], lam_i[mask]
    diff = a - b
    N = float(np.sum(np.abs(diff)))
    D = float(np.sum(np.abs(a)) + np.sum(np.abs(b)))
    if D == 0:
        return np.nan, np.nan
    d = 1.0 - N / D

    s_diff = np.sign(diff)
    ga = -(s_diff / D - N * np.sign(a) / (D * D))
    gb = -(-s_diff / D - N * np.sign(b) / (D * D))
    var = (float(np.sum((ga * sem_ref[mask]) ** 2))
           + float(np.sum((gb * sem_i[mask]) ** 2)))
    return d, float(np.sqrt(var))


def ref_friction(data, cfg):
    return {"scalar": data["scalar"], "curve": data["curve"]}


def score_friction(ctx, data, cfg):
    lam_ref = ctx["scalar"]["mean"]
    sem_ref = ctx["scalar"]["sem"]
    lam = data["scalar"]["mean"]
    sem = data["scalar"]["sem"]
    n_blocks = data["scalar"]["n_blocks"]

    signed = lam - lam_ref
    d_s = scalar_similarity(lam_ref, lam)
    den = abs(lam_ref) + abs(lam)
    d_s_sem = float(np.sqrt(sem ** 2 + sem_ref ** 2) / den) if den > 0 else 0.0

    d_c, d_c_sem = _lambda_curve_similarity(ctx["curve"], data["curve"],
                                            cfg["curve_window"])

    out = {
        "scalar": record(d_s, d_s_sem, n_blocks,
                         value=lam, value_sem=sem, ref_value=lam_ref,
                         signed_err=float(signed), abs_err=abs(float(signed)),
                         rel_err=(abs(signed) / abs(lam_ref)
                                  if lam_ref != 0 else np.nan)),
        "curve": record(d_c, d_c_sem, n_blocks),
    }
    if np.isfinite(d_s) and np.isfinite(d_c):
        out["combined"] = record(0.5 * (d_s + d_c),
                                 0.5 * float(np.sqrt(d_s_sem ** 2
                                                     + d_c_sem ** 2)),
                                 n_blocks)
    else:
        out["combined"] = record(np.nan, np.nan, n_blocks)
    return out


# ===========================================================================
# ORIENTATION  (notebook cell 2)
# ===========================================================================

ORIENTATION_KEY = "water"


def _edges_from_centres(x):
    x = np.asarray(x, dtype=float)
    h = float(np.median(np.diff(x)))
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - h / 2
    edges[-1] = x[-1] + h / 2
    return edges


def load_orientation(interf_root, bulk_root, fn, cfg):
    paths = _interf_glob(interf_root, fn, cfg["system"], cfg["n_tag"],
                         cfg["orientation_pkl_glob"])
    runs = []
    for p in paths:
        with open(p, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and ORIENTATION_KEY in data:
            x, y = data[ORIENTATION_KEY]
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            raw_d = data.get("_raw_distances")
            weights = None
            if raw_d is not None:
                arr = np.asarray(raw_d, dtype=float).ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    counts, _ = np.histogram(arr, bins=_edges_from_centres(x))
                    weights = counts.astype(float)
        elif isinstance(data, tuple) and len(data) == 2:
            x = np.asarray(data[0], float)
            y = np.asarray(data[1], float)
            weights = None
        else:
            continue

        runs.append({"x": x, "y": y, "w": weights})
        # The raw (n_frames, n_waters) arrays are ~30 MB each; drop them now.
        del data
    return {"runs": runs} if runs else None


def _cutoff_mask(x, z_min, z_max):
    m = np.ones(np.shape(x), dtype=bool)
    if z_min is not None:
        m &= x >= float(z_min)
    if z_max is not None:
        m &= x <= float(z_max)
    return m


def _average_orientation(runs):
    x_ref = runs[0]["x"]
    ys, ws = [], []
    for r in runs:
        ys.append(np.interp(x_ref, r["x"], r["y"]))
        if r["w"] is not None:
            ws.append(np.interp(x_ref, r["x"], r["w"]))
    y_mean = np.mean(np.stack(ys, axis=0), axis=0)
    w_mean = np.mean(np.stack(ws, axis=0), axis=0) if ws else None
    return x_ref, y_mean, w_mean


def ref_orientation(data, cfg):
    x_ref, y_ref, w_ref = _average_orientation(data["runs"])
    mask = _cutoff_mask(x_ref, cfg["z_min"], cfg["z_max"])
    if not mask.any():
        return None
    return {"profile": (x_ref, y_ref), "w": w_ref, "mask": mask}


def _orientation_similarity(ref, test, weights, z_min, z_max):
    x_ref, y_ref = np.asarray(ref[0], float), np.asarray(ref[1], float)
    y_i = np.interp(x_ref, np.asarray(test[0], float),
                    np.asarray(test[1], float))

    if weights is None:
        w = np.ones_like(y_ref)
    else:
        w = np.asarray(weights, float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    w = np.where(_cutoff_mask(x_ref, z_min, z_max), w, 0.0)
    if not np.any(w > 0):
        return np.nan

    num = np.sum(w * np.abs(y_ref - y_i))
    den = np.sum(w * (np.abs(y_ref) + np.abs(y_i)))
    if den == 0:
        return np.nan
    return 1.0 - num / den


def score_orientation(ctx, data, cfg):
    x_ref = ctx["profile"][0]
    w_ref = ctx["w"]
    source = cfg["weight_source"]

    scores = []
    for run in data["runs"]:
        if source == "none" or w_ref is None:
            w = None
        elif source == "ref":
            w = w_ref
        elif source == "both":
            if run["w"] is None:
                w = w_ref
            else:
                w_i = np.interp(x_ref, run["x"], run["w"])
                w = np.sqrt(np.clip(w_ref, 0, None) * np.clip(w_i, 0, None))
        else:
            raise ValueError(f"Unknown weight_source: {source}")
        scores.append(_orientation_similarity(
            ctx["profile"], (run["x"], run["y"]), w,
            cfg["z_min"], cfg["z_max"]))

    s = _summary_stats(scores)
    return {ORIENTATION_KEY: record(s["mean"], s["sem"], s["n_runs"])}


# ===========================================================================
# RDF and VDOS  (notebook cells 3 and 5)
#
# Same shape: per-run interfacial curves + one bulk curve per functional,
# combined per species / per condition / overall.
# ===========================================================================

def _get_species(d, sp):
    """Species lookup tolerating 'A-B' vs 'B-A'."""
    if not isinstance(d, dict):
        return None
    if sp in d:
        return d[sp]
    if "-" in sp:
        a, b = sp.split("-")
        alt = f"{b}-{a}"
        if alt in d:
            return d[alt]
    return None


def _load_curve_property(interf_root, bulk_root, fn, cfg, pkl_glob,
                         bulk_stem, unwrap_keys, species):
    paths = _interf_glob(interf_root, fn, cfg["system"], cfg["n_tag"], pkl_glob)
    runs = []
    for p in paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        data = _unwrap(data, unwrap_keys)
        run = {}
        for sp in species:
            xy = _get_species(data, sp)
            if xy is not None:
                run[sp] = (np.asarray(xy[0], float), np.asarray(xy[1], float))
        runs.append(run)

    bulk = None
    bpath = _bulk_pkl(bulk_root, fn, cfg["n_tag"], bulk_stem)
    if bpath is not None:
        with open(bpath, "rb") as f:
            data = pickle.load(f)
        data = _unwrap(data, unwrap_keys)
        bulk = {}
        for sp in species:
            xy = _get_species(data, sp)
            if xy is not None:
                bulk[sp] = (np.asarray(xy[0], float), np.asarray(xy[1], float))

    if not runs and not bulk:
        return None
    return {"runs": runs, "bulk": bulk}


def load_rdf(interf_root, bulk_root, fn, cfg):
    return _load_curve_property(
        interf_root, bulk_root, fn, cfg, cfg["rdf_pkl_glob"], "rdfs",
        ("test_rdf", "ref_rdf", "rdf"), cfg["rdf_species"])


def load_vdos(interf_root, bulk_root, fn, cfg):
    return _load_curve_property(
        interf_root, bulk_root, fn, cfg, cfg["vdos_pkl_glob"], "vdos",
        ("test_vdos", "ref_vdos", "vdos", "test_rdf", "ref_rdf", "rdf"),
        cfg["vdos_species"])


def _ref_curve_property(data, species):
    ref = {"interfacial": {}, "bulk": data.get("bulk")}
    for sp in species:
        curves = [r[sp] for r in data["runs"] if sp in r]
        if curves:
            x, mean, _, _ = average_curves(curves)
            ref["interfacial"][sp] = (x, mean)
    if not ref["interfacial"] and not ref["bulk"]:
        return None
    return ref


def ref_rdf(data, cfg):
    return _ref_curve_property(data, cfg["rdf_species"])


def ref_vdos(data, cfg):
    return _ref_curve_property(data, cfg["vdos_species"])


def _combine_blocks(interf, bulk, species):
    """Per-species / per-condition / overall combination (notebook _combine_results)."""
    out = {}

    for sp in species:
        i = interf.get(sp)
        b = bulk.get(sp)
        if i is not None:
            out[f"interfacial:{sp}"] = i
        if b is not None:
            out[f"bulk:{sp}"] = b

        vals, sems, ns = [], [], []
        for rec in (i, b):
            if rec is not None and np.isfinite(rec["d"]):
                vals.append(rec["d"])
                sems.append(rec["d_sem"])
                ns.append(rec["n_runs"])
        if vals:
            sem = (float(np.sqrt(np.sum(np.square(sems))) / len(vals))
                   if len(vals) > 1 else float(sems[0]))
            out[f"overall:{sp}"] = record(float(np.mean(vals)), sem,
                                          int(np.sum(ns)))
        else:
            out[f"overall:{sp}"] = record(np.nan, np.nan, 0)

    def _cond_combined(prefix):
        recs = [out[f"{prefix}:{sp}"] for sp in species
                if f"{prefix}:{sp}" in out
                and np.isfinite(out[f"{prefix}:{sp}"]["d"])]
        if not recs:
            return record(np.nan, np.nan, 0)
        means = [r["d"] for r in recs]
        sems = [r["d_sem"] for r in recs]
        sem = (float(np.sqrt(np.sum(np.square(sems))) / len(sems))
               if len(sems) > 1 else float(sems[0]))
        return record(float(np.mean(means)), sem,
                      int(np.sum([r["n_runs"] for r in recs])))

    out["interfacial:combined"] = _cond_combined("interfacial")
    out["bulk:combined"] = _cond_combined("bulk")

    cvals, csems, cns = [], [], []
    for cond in ("interfacial", "bulk"):
        c = out[f"{cond}:combined"]
        if np.isfinite(c["d"]):
            cvals.append(c["d"])
            csems.append(c["d_sem"])
            cns.append(c["n_runs"])
    if cvals:
        sem = (float(np.sqrt(np.sum(np.square(csems))) / len(csems))
               if len(csems) > 1 else float(csems[0]))
        out["overall:combined"] = record(float(np.mean(cvals)), sem,
                                         int(np.sum(cns)))
    else:
        out["overall:combined"] = record(np.nan, np.nan, 0)
    return out


def _score_curve_property(ctx, data, species):
    interf = {}
    for sp in species:
        if sp not in ctx["interfacial"]:
            continue
        scores = [bray_curtis(ctx["interfacial"][sp], r[sp])
                  for r in data["runs"] if sp in r]
        if not scores:
            continue
        s = _summary_stats(scores)
        interf[sp] = record(s["mean"], s["sem"], s["n_runs"])

    bulk = {}
    if ctx.get("bulk") and data.get("bulk"):
        for sp in species:
            ref = ctx["bulk"].get(sp)
            test = data["bulk"].get(sp)
            if ref is None or test is None:
                continue
            # One aggregated bulk pkl per functional: n_runs = 1, sem = 0.
            bulk[sp] = record(bray_curtis(ref, test), 0.0, 1)

    return _combine_blocks(interf, bulk, species)


def score_rdf(ctx, data, cfg):
    return _score_curve_property(ctx, data, cfg["rdf_species"])


def score_vdos(ctx, data, cfg):
    return _score_curve_property(ctx, data, cfg["vdos_species"])


# ===========================================================================
# SFG  (notebook cell 4)
# ===========================================================================

def load_sfg(interf_root, bulk_root, fn, cfg):
    out = {}
    for m in cfg["sfg_modes"]:
        paths = _interf_glob(interf_root, fn, cfg["system"], cfg["n_tag"],
                             f"FT_{m}.dat")
        runs = []
        for p in paths:
            arr = np.loadtxt(p)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            runs.append((arr[:, 0].astype(float), arr[:, 1].astype(float)))
        if not runs:
            return None
        out[m] = runs
    return {"raw": out}


def _noncondon(wn):
    """Auer-Skinner mu*alpha non-Condon factor."""
    wn = np.asarray(wn, dtype=float)
    mu = 1.377 + (53.03 * (3737.0 - wn) / 6932.2)
    alpha = 1.271 + (6.287 * (3737.0 - wn) / 6932.2)
    return mu * alpha


def _window_mask(freq, window):
    freq = np.asarray(freq, float)
    if window is None:
        return np.ones_like(freq, dtype=bool)
    m = (freq >= window[0]) & (freq <= window[1])
    return m if m.any() else np.ones_like(freq, dtype=bool)


def _normalise_max(freq, y, window):
    mask = _window_mask(freq, window)
    scale = np.nanmax(np.asarray(y, float)[mask])
    if not np.isfinite(scale) or scale == 0:
        return y
    return y / scale


def _preprocess_simulated(freq, y, window):
    return freq, _normalise_max(freq, np.asarray(y, float) * _noncondon(freq),
                                window)


def _hbond_peak(freq, y, window):
    m = _window_mask(freq, window)
    f, yy = np.asarray(freq, float)[m], np.asarray(y, float)[m]
    i = int(np.argmin(yy))
    return float(f[i]), float(yy[i])


def _freeoh_peak(freq, y, window):
    m = _window_mask(freq, window)
    f, yy = np.asarray(freq, float)[m], np.asarray(y, float)[m]
    i = int(np.argmax(yy))
    return float(f[i]), float(yy[i])


def _hbond_zero_crossing(freq, y, window):
    """Last negative->positive crossing between the trough and the free-OH max."""
    m = _window_mask(freq, window)
    f, yy = np.asarray(freq, float)[m], np.asarray(y, float)[m]
    order = np.argsort(f)
    f, yy = f[order], yy[order]

    i_tr = int(np.argmin(yy))
    i_mx = int(np.argmax(yy))
    if i_mx <= i_tr:
        return None

    x_cross = None
    for j in range(i_tr, i_mx):
        y0, y1 = yy[j], yy[j + 1]
        if y0 < 0.0 <= y1 and y1 != y0:
            t = (0.0 - y0) / (y1 - y0)
            x_cross = f[j] + t * (f[j + 1] - f[j])
    return None if x_cross is None else float(x_cross)


def _resolve_width_bounds(freq, y, window, width_window):
    freq = np.asarray(freq, float)
    m = _window_mask(freq, window)
    f_lo, f_hi = float(np.min(freq[m])), float(np.max(freq[m]))
    if width_window is None:
        return f_lo, f_hi
    lo_req, hi_req = width_window
    lo = f_lo if lo_req is None else float(lo_req)
    if hi_req is not None:
        return lo, float(hi_req)
    x_cross = _hbond_zero_crossing(freq, y, window)
    return lo, (f_hi if x_cross is None else x_cross)


def _trapz(y, x):
    fn = getattr(np, "trapezoid", getattr(np, "trapz"))
    return float(fn(y, x))


def _hbond_width_integral(freq, y, window, width_window):
    freq = np.asarray(freq, float)
    y = np.asarray(y, float)
    lo, hi = _resolve_width_bounds(freq, y, window, width_window)
    m = (freq >= lo) & (freq <= hi)
    if not m.any():
        return np.nan
    f, yy = freq[m], y[m]
    order = np.argsort(f)
    f, yy = f[order], yy[order]
    return _trapz(np.where(yy < 0, -yy, 0.0), f)


def _hbond_width_equivalent(freq, y, window, width_window):
    area = _hbond_width_integral(freq, y, window, width_window)
    _, y_min = _hbond_peak(freq, y, window)
    if not np.isfinite(area) or not np.isfinite(y_min) or y_min >= 0:
        return np.nan
    return float(area / abs(y_min))


def _hbond_width_fwhm(freq, y, window):
    m = _window_mask(freq, window)
    f, yy = np.asarray(freq, float)[m], np.asarray(y, float)[m]
    order = np.argsort(f)
    f, yy = f[order], yy[order]

    i_peak = int(np.argmin(yy))
    y_min = yy[i_peak]
    if y_min >= 0:
        return np.nan
    half = y_min / 2.0

    x_left = None
    for j in range(i_peak, 0, -1):
        y0, y1 = yy[j], yy[j - 1]
        if (y0 - half) * (y1 - half) <= 0 and y1 != y0:
            x_left = f[j] + (half - y0) / (y1 - y0) * (f[j - 1] - f[j])
            break

    x_right = None
    for j in range(i_peak, len(yy) - 1):
        y0, y1 = yy[j], yy[j + 1]
        if (y0 - half) * (y1 - half) <= 0 and y1 != y0:
            x_right = f[j] + (half - y0) / (y1 - y0) * (f[j + 1] - f[j])
            break

    if x_left is None or x_right is None:
        return np.nan
    return float(abs(x_right - x_left))


def _extract_descriptors(freq, y, window, width_method, width_window):
    f_hb, y_hb = _hbond_peak(freq, y, window)
    f_oh, y_oh = _freeoh_peak(freq, y, window)
    ratio = np.nan if y_hb == 0 else float(y_oh / y_hb)

    if width_method == "equivalent":
        width = _hbond_width_equivalent(freq, y, window, width_window)
    elif width_method == "fwhm":
        width = _hbond_width_fwhm(freq, y, window)
    elif width_method == "integral":
        width = _hbond_width_integral(freq, y, window, width_window)
    else:
        raise ValueError(f"Unknown width method {width_method!r}")

    return {"hbond_pos": f_hb, "freeoh_pos": f_oh, "ratio": ratio,
            "width": width}


def _agreement(q_test, q_ref):
    """Symmetric relative agreement in [0, 1].

        a = 1 - |q_test - q_ref| / (|q_ref| + |q_test|)

    Same normalisation as the friction scalar score and as the
    function-based norms used for the RDFs, VDOS, density and orientation
    profiles, so every observable in the framework is on one scale.

    Bounded in [0, 1] by construction: when q_test and q_ref share a sign
    the numerator is at most their sum, and when they differ in sign
    |q_test - q_ref| = |q_ref| + |q_test| exactly, giving 0. The clamp is
    therefore redundant but kept as a guard against float drift.
    """
    if q_ref is None or not np.isfinite(q_ref):
        return np.nan
    if q_test is None or not np.isfinite(q_test):
        return np.nan
    den = abs(q_ref) + abs(q_test)
    if den == 0:
        return np.nan
    a = 1.0 - abs(q_test - q_ref) / den
    return float(min(1.0, max(0.0, a)))


def _score_descriptors(test_desc, ref_desc, strict):
    a_hb = _agreement(test_desc["hbond_pos"], ref_desc["hbond_pos"])
    a_oh = _agreement(test_desc["freeoh_pos"], ref_desc["freeoh_pos"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a_pos = float(np.nanmean([a_hb, a_oh]))

    comps = {"position": a_pos,
             "ratio": _agreement(test_desc["ratio"], ref_desc["ratio"]),
             "width": _agreement(test_desc["width"], ref_desc["width"])}
    missing = [k for k, v in comps.items() if not np.isfinite(v)]

    if strict and missing:
        dscore = np.nan
    else:
        finite = [v for v in comps.values() if np.isfinite(v)]
        dscore = float(np.mean(finite)) if finite else np.nan

    comps["dscore"] = dscore
    return comps


def _sfg_processed(data, cfg):
    return {m: [_preprocess_simulated(f, y, cfg["sfg_window"])
                for f, y in runs]
            for m, runs in data["raw"].items()}


def ref_sfg(data, cfg):
    proc = _sfg_processed(data, cfg)
    ref_desc = {}
    for m in cfg["sfg_modes"]:
        f_mean, y_mean, _, _ = average_curves(proc[m])
        if cfg["sfg_renormalise_ref"]:
            # Averaging max-normalised runs pulls the mean max below 1 whenever
            # the free-OH peak shifts between runs; every test run has max == 1
            # exactly, so renormalise to avoid biasing the `ratio` descriptor.
            y_mean = _normalise_max(f_mean, y_mean, cfg["sfg_window"])
        ref_desc[m] = _extract_descriptors(
            f_mean, y_mean, cfg["sfg_window"], cfg["sfg_width_method"],
            cfg["sfg_width_window"])
    return {"desc": ref_desc}


def score_sfg(ctx, data, cfg):
    modes = cfg["sfg_modes"]
    proc = _sfg_processed(data, cfg)
    n_runs = min(len(proc[m]) for m in modes)

    per_mode = {m: {c: [] for c in ("position", "ratio", "width", "dscore")}
                for m in modes}
    for i in range(n_runs):
        for m in modes:
            f, y = proc[m][i]
            td = _extract_descriptors(f, y, cfg["sfg_window"],
                                      cfg["sfg_width_method"],
                                      cfg["sfg_width_window"])
            sc = _score_descriptors(td, ctx["desc"][m],
                                    cfg["sfg_strict_components"])
            for c in per_mode[m]:
                per_mode[m][c].append(sc[c])

    out = {}
    for m in modes:
        for c in ("position", "ratio", "width"):
            s = _summary_stats(per_mode[m][c])
            out[f"{m}:{c}"] = record(s["mean"], s["sem"], s["n_runs"])
        s = _summary_stats(per_mode[m]["dscore"])
        out[m] = record(s["mean"], s["sem"], s["n_runs"])

    combined = _nanmean_rows([per_mode[m]["dscore"] for m in modes])
    s = _summary_stats(combined.tolist())
    out["combined"] = record(s["mean"], s["sem"], s["n_runs"])
    return out


# ===========================================================================
# SFG vs the fixed experimental reference (Wang et al. 2024 HD-SFG)
#
# A single digitized spectrum per mode, not a per-functional simulation
# directory, so it does not go through load_sfg/discover_functionals -- it is
# scored once, on demand, against whatever functionals were already loaded
# for the ordinary sfg property.
# ===========================================================================

def load_sfg_experimental(cfg):
    """{mode: (freq, intensity)} from the digitized experimental csv, or None."""
    path = cfg["sfg_expt_path"]
    if not path or not os.path.isfile(path):
        return None
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    freq = arr[:, 0].astype(float)

    out = {}
    for m in cfg["sfg_modes"]:
        col = cfg["sfg_expt_columns"].get(m)
        if col is None or col >= arr.shape[1]:
            continue
        out[m] = (freq, arr[:, col].astype(float))
    return out or None


def ref_sfg_experimental(cfg):
    """Descriptor context built straight from the (already normalised)
    experimental spectrum -- no non-Condon correction, unlike the simulated
    reference (see _preprocess_simulated vs. this)."""
    raw = load_sfg_experimental(cfg)
    if raw is None:
        return None
    ref_desc = {}
    for m in cfg["sfg_modes"]:
        if m not in raw:
            continue
        f, y = raw[m]
        y = _normalise_max(f, y, cfg["sfg_window"])
        ref_desc[m] = _extract_descriptors(
            f, y, cfg["sfg_window"], cfg["sfg_width_method"],
            cfg["sfg_width_window"])
    return {"desc": ref_desc} if ref_desc else None


def run_experimental_sfg(data, tests, cfg, verbose=True):
    """{test: {metric: record}} scoring every test functional's sfg data
    against the fixed experimental reference. None if the experimental
    reference data is unavailable."""
    ctx = ref_sfg_experimental(cfg)
    if ctx is None:
        if verbose:
            print(f"\n[warn] experimental SFG reference not found "
                  f"({cfg['sfg_expt_path']}); skipping scores_experimental.")
        return None

    pdata = data.get("sfg", {})
    per_ref = {}
    for test in tests:
        if test not in pdata:
            continue
        try:
            per_ref[test] = score_sfg(ctx, pdata[test], cfg)
        except Exception as e:                          # noqa: BLE001
            if verbose:
                print(f"  [warn] experimental vs {test}: "
                      f"{type(e).__name__}: {e}")
    if not per_ref:
        return None
    _normalise_kappa(per_ref)
    if verbose:
        primary = PROPERTY_SPEC["sfg"][3]
        best = _rank_line(per_ref, primary)
        print(f"\n=== sfg vs experimental (Wang et al. 2024): "
              f"n={len(per_ref)} ===")
        print(f"  best: {best}")
    return per_ref


# ===========================================================================
# Property registry
# ===========================================================================

PROPERTY_SPEC = {
    "density":     (load_density, ref_density, score_density, "density_and_ratio"),
    "friction":    (load_friction, ref_friction, score_friction, "combined"),
    "orientation": (load_orientation, ref_orientation, score_orientation,
                    ORIENTATION_KEY),
    "rdf":         (load_rdf, ref_rdf, score_rdf, "overall:combined"),
    "vdos":        (load_vdos, ref_vdos, score_vdos, "overall:combined"),
    "sfg":         (load_sfg, ref_sfg, score_sfg, "combined"),
}


# ===========================================================================
# Discovery and caching
# ===========================================================================

def discover_functionals(interf_root, system, exclude=()):
    names = []
    for name in sorted(os.listdir(interf_root)):
        full = os.path.join(interf_root, name)
        if not os.path.isdir(full) or name in exclude:
            continue
        if not os.path.isdir(os.path.join(full, "gra-runs", "frozen", system)):
            continue
        names.append(name)
    return names


def _cache_key(cfg, properties):
    return {"version": CACHE_VERSION,
            "properties": sorted(properties),
            "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in cfg.items()}}


def load_all(interf_root, bulk_root, functionals, properties, cfg,
             cache_path=None, refresh=False, verbose=True):
    """{property: {functional: reduced_data}} with an on-disk cache."""
    key = _cache_key(cfg, properties)

    if cache_path and os.path.isfile(cache_path) and not refresh:
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if (blob.get("key") == key
                and set(functionals).issubset(set(blob.get("functionals", [])))):
            if verbose:
                print(f"Loaded reduced data from cache: {cache_path}")
            return blob["data"]
        if verbose:
            print("Cache present but stale (config or functional set changed); "
                  "re-reading raw data.")

    data = {p: {} for p in properties}
    for i, fn in enumerate(functionals, 1):
        t0 = time.time()
        got = []
        for p in properties:
            loader = PROPERTY_SPEC[p][0]
            try:
                d = loader(interf_root, bulk_root, fn, cfg)
            except Exception as e:                      # noqa: BLE001
                if verbose:
                    print(f"  [warn] {fn}/{p}: {type(e).__name__}: {e}")
                d = None
            if d is not None:
                data[p][fn] = d
                got.append(p)
        if verbose:
            print(f"[{i:2d}/{len(functionals)}] {fn:16s} "
                  f"{', '.join(got) if got else '(no data)':55s} "
                  f"{time.time() - t0:5.1f}s")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".",
                    exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"key": key, "functionals": list(functionals),
                         "data": data}, f, protocol=4)
        if verbose:
            print(f"Cached reduced data -> {cache_path}")
    return data


# ===========================================================================
# Driver: all-vs-all scoring
# ===========================================================================

def run_all_vs_all(data, references, tests, properties, cfg,
                   include_self=False, verbose=True):
    """results[property][reference][test][metric] = record dict."""
    results = {}
    for prop in properties:
        _, ref_fn, score_fn, _ = PROPERTY_SPEC[prop]
        pdata = data.get(prop, {})
        refs = [r for r in references if r in pdata]
        if verbose:
            print(f"\n=== {prop}: {len(refs)} references x "
                  f"{len([t for t in tests if t in pdata])} tests ===")

        results[prop] = {}
        for ref in refs:
            try:
                ctx = ref_fn(pdata[ref], cfg)
            except Exception as e:                      # noqa: BLE001
                if verbose:
                    print(f"  [warn] reference {ref}: {type(e).__name__}: {e}")
                continue
            if ctx is None:
                if verbose:
                    print(f"  [warn] reference {ref}: unusable reference data")
                continue

            per_ref = {}
            for test in tests:
                if test not in pdata:
                    continue
                if test == ref and not include_self:
                    continue
                try:
                    per_ref[test] = score_fn(ctx, pdata[test], cfg)
                except Exception as e:                  # noqa: BLE001
                    if verbose:
                        print(f"  [warn] {ref} vs {test}: "
                              f"{type(e).__name__}: {e}")
            _normalise_kappa(per_ref)
            results[prop][ref] = per_ref
            if verbose:
                primary = PROPERTY_SPEC[prop][3]
                best = _rank_line(per_ref, primary)
                print(f"  ref={ref:16s} n={len(per_ref):2d}   best: {best}")
    return results


def _normalise_kappa(per_ref):
    """kappa = (1 - d) / sigma_d, sigma over the functionals in this column.

    Where a metric also carries a raw scalar (lambda, rho2/rho1), add the
    directly interpretable kappa_abs = |Delta| / sigma(value).
    """
    metrics = sorted({m for rec in per_ref.values() for m in rec})

    for metric in metrics:
        ds = np.array([per_ref[t][metric]["d"] for t in per_ref
                       if metric in per_ref[t]], dtype=float)
        valid = ds[np.isfinite(ds)]
        sigma = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

        vals = np.array([per_ref[t][metric].get("value", np.nan)
                         for t in per_ref if metric in per_ref[t]],
                        dtype=float)
        vvalid = vals[np.isfinite(vals)]
        sigma_v = float(np.std(vvalid, ddof=1)) if len(vvalid) > 1 else 0.0

        for t in per_ref:
            rec = per_ref[t].get(metric)
            if rec is None:
                continue
            rec["sigma_d"] = sigma
            if sigma > 0 and np.isfinite(rec["d"]):
                rec["kappa"] = (1.0 - rec["d"]) / sigma
                rec["kappa_sem"] = (rec["d_sem"] or 0.0) / sigma
            else:
                rec["kappa"] = np.nan
                rec["kappa_sem"] = np.nan

            if sigma_v > 0 and np.isfinite(rec.get("signed_err", np.nan)):
                rec["sigma_value"] = sigma_v
                rec["kappa_abs"] = abs(rec["signed_err"]) / sigma_v
                rec["kappa_abs_sem"] = (rec.get("value_sem", 0.0) or 0.0) / sigma_v

    # rank 1 = best (highest d) within each (reference, metric) column
    for metric in metrics:
        scored = [(t, per_ref[t][metric]["d"]) for t in per_ref
                  if metric in per_ref[t]
                  and np.isfinite(per_ref[t][metric]["d"])]
        scored.sort(key=lambda kv: -kv[1])
        for i, (t, _) in enumerate(scored, 1):
            per_ref[t][metric]["rank"] = i


def _rank_line(per_ref, metric, n=3):
    scored = [(t, per_ref[t][metric]["d"]) for t in per_ref
              if metric in per_ref[t] and np.isfinite(per_ref[t][metric]["d"])]
    if not scored:
        return "(no finite scores)"
    scored.sort(key=lambda kv: -kv[1])
    return "  ".join(f"{t} ({d:.4f})" for t, d in scored[:n])


# ===========================================================================
# Output
# ===========================================================================

def _fmt(x, p=6):
    try:
        return f"{float(x):.{p}f}" if np.isfinite(x) else "nan"
    except (TypeError, ValueError):
        return "nan"


def write_experimental_scores(exp_results, out_dir, verbose=True):
    """scores_experimental.csv: every functional vs. the fixed sfg
    experimental reference, same columns as pairwise_scores.csv."""
    path = os.path.join(out_dir, "scores_experimental.csv")
    primary = PROPERTY_SPEC["sfg"][3]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reference", "test", "property", "metric", "is_primary",
                    "n_runs", "d", "d_sem", "kappa", "kappa_sem",
                    "sigma_d", "rank"])
        for test in sorted(exp_results):
            for metric in sorted(exp_results[test]):
                r = exp_results[test][metric]
                w.writerow([
                    "Experimental", test, "sfg", metric,
                    int(metric == primary), r["n_runs"],
                    _fmt(r["d"]), _fmt(r["d_sem"]),
                    _fmt(r.get("kappa", np.nan), 4),
                    _fmt(r.get("kappa_sem", np.nan), 4),
                    _fmt(r.get("sigma_d", np.nan), 6),
                    r.get("rank", ""),
                ])
    if verbose:
        print(f"  {path}")
    return path


def write_outputs(results, out_dir, properties, cfg, verbose=True,
                  exp_results=None):
    os.makedirs(out_dir, exist_ok=True)
    mat_dir = os.path.join(out_dir, "matrices")
    os.makedirs(mat_dir, exist_ok=True)

    # ---- tidy scores ------------------------------------------------------
    scores_path = os.path.join(out_dir, "pairwise_scores.csv")
    with open(scores_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reference", "test", "property", "metric", "is_primary",
                    "n_runs", "d", "d_sem", "kappa", "kappa_sem",
                    "sigma_d", "rank"])
        for prop in properties:
            primary = PROPERTY_SPEC[prop][3]
            for ref in sorted(results.get(prop, {})):
                per_ref = results[prop][ref]
                for test in sorted(per_ref):
                    for metric in sorted(per_ref[test]):
                        r = per_ref[test][metric]
                        w.writerow([
                            ref, test, prop, metric,
                            int(metric == primary), r["n_runs"],
                            _fmt(r["d"]), _fmt(r["d_sem"]),
                            _fmt(r.get("kappa", np.nan), 4),
                            _fmt(r.get("kappa_sem", np.nan), 4),
                            _fmt(r.get("sigma_d", np.nan), 6),
                            r.get("rank", ""),
                        ])

    # ---- raw scalar values + signed errors --------------------------------
    values_path = os.path.join(out_dir, "pairwise_values.csv")
    with open(values_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["reference", "test", "property", "metric",
                    "ref_value", "value", "value_sem",
                    "signed_err", "abs_err", "rel_err",
                    "kappa_abs", "kappa_abs_sem", "sigma_value"])
        for prop in properties:
            for ref in sorted(results.get(prop, {})):
                per_ref = results[prop][ref]
                for test in sorted(per_ref):
                    for metric in sorted(per_ref[test]):
                        r = per_ref[test][metric]
                        if "value" not in r:
                            continue
                        w.writerow([
                            ref, test, prop, metric,
                            f"{r['ref_value']:.6g}", f"{r['value']:.6g}",
                            f"{r.get('value_sem', float('nan')):.6g}",
                            f"{r['signed_err']:.6g}", f"{r['abs_err']:.6g}",
                            _fmt(r.get("rel_err", np.nan)),
                            _fmt(r.get("kappa_abs", np.nan), 4),
                            _fmt(r.get("kappa_abs_sem", np.nan), 4),
                            _fmt(r.get("sigma_value", np.nan), 6),
                        ])

    # ---- N x N d matrices, one file per (property, metric) ----------------
    n_matrices = 0
    for prop in properties:
        by_ref = results.get(prop, {})
        if not by_ref:
            continue
        metrics = sorted({m for per_ref in by_ref.values()
                          for rec in per_ref.values() for m in rec})
        tests = sorted({t for per_ref in by_ref.values() for t in per_ref})
        refs = sorted(by_ref)
        for metric in metrics:
            safe = metric.replace(":", "-").replace("/", "-")
            path = os.path.join(mat_dir, f"{prop}__{safe}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["reference \\ test"] + tests)
                for ref in refs:
                    row = [ref]
                    for t in tests:
                        rec = by_ref[ref].get(t, {}).get(metric)
                        row.append(_fmt(rec["d"]) if rec else "")
                    w.writerow(row)
            n_matrices += 1

    # ---- consensus across references --------------------------------------
    consensus_path = os.path.join(out_dir, "reference_consensus.csv")
    with open(consensus_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["property", "metric", "functional", "n_references",
                    "mean_d", "std_d", "mean_rank", "std_rank",
                    "best_rank", "worst_rank"])
        for prop in properties:
            by_ref = results.get(prop, {})
            if not by_ref:
                continue
            metrics = sorted({m for per_ref in by_ref.values()
                              for rec in per_ref.values() for m in rec})
            for metric in metrics:
                agg = defaultdict(lambda: {"d": [], "rank": []})
                for per_ref in by_ref.values():
                    for test, recs in per_ref.items():
                        rec = recs.get(metric)
                        if rec is None or not np.isfinite(rec["d"]):
                            continue
                        agg[test]["d"].append(rec["d"])
                        if "rank" in rec:
                            agg[test]["rank"].append(rec["rank"])
                rows = []
                for test, vals in agg.items():
                    d = np.asarray(vals["d"], float)
                    rk = np.asarray(vals["rank"], float)
                    rows.append((
                        test, len(d), float(np.mean(d)),
                        float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                        float(np.mean(rk)) if rk.size else np.nan,
                        float(np.std(rk, ddof=1)) if rk.size > 1 else 0.0,
                        int(np.min(rk)) if rk.size else "",
                        int(np.max(rk)) if rk.size else "",
                    ))
                rows.sort(key=lambda r: (np.inf if not np.isfinite(r[4])
                                         else r[4]))
                for r in rows:
                    w.writerow([prop, metric, r[0], r[1], _fmt(r[2]),
                                _fmt(r[3]), _fmt(r[4], 3), _fmt(r[5], 3),
                                r[6], r[7]])

    # ---- sfg vs. fixed experimental reference ------------------------------
    exp_path = None
    if exp_results:
        exp_path = write_experimental_scores(exp_results, out_dir,
                                             verbose=False)

    # ---- everything, unrounded --------------------------------------------
    pkl_path = os.path.join(out_dir, "summary.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump({"config": cfg, "properties": list(properties),
                     "results": results,
                     "results_experimental": exp_results}, fh, protocol=4)

    if verbose:
        print(f"\nSaved:")
        print(f"  {scores_path}")
        print(f"  {values_path}")
        print(f"  {consensus_path}")
        print(f"  {mat_dir}/  ({n_matrices} matrices)")
        if exp_path:
            print(f"  {exp_path}")
        print(f"  {pkl_path}")


def print_consensus(results, properties, top=8):
    """Short human-readable digest: mean rank over all references."""
    print("\n" + "=" * 74)
    print("Consensus ranking on each property's primary metric")
    print("(mean rank over every reference functional; 1 = best)")
    print("=" * 74)
    for prop in properties:
        by_ref = results.get(prop, {})
        primary = PROPERTY_SPEC[prop][3]
        if not by_ref:
            continue
        agg = defaultdict(list)
        for per_ref in by_ref.values():
            for test, recs in per_ref.items():
                rec = recs.get(primary)
                if rec and "rank" in rec:
                    agg[test].append(rec["rank"])
        if not agg:
            continue
        rows = sorted(((t, float(np.mean(v)), float(np.std(v)), len(v))
                       for t, v in agg.items()), key=lambda r: r[1])
        print(f"\n{prop}  [{primary}]  ({len(by_ref)} references)")
        for t, mu, sd, n in rows[:top]:
            print(f"   {t:18s} mean rank {mu:5.2f} +/- {sd:4.2f}  (n={n})")
        if len(rows) > top:
            print(f"   ... {len(rows) - top} more in reference_consensus.csv")


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Benchmark every functional against every other "
                    "functional on all properties.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--interf-root", default=DEFAULT_INTERF_ROOT,
                    help="root holding <functional>/gra-runs/...")
    ap.add_argument("--bulk-root", default=DEFAULT_BULK_ROOT,
                    help="root holding <functional>/bulk-runs/... (rdf, vdos)")
    ap.add_argument("-o", "--out-dir", default="./benchmark_all_references")
    ap.add_argument("-p", "--properties", nargs="+", default=list(ALL_PROPERTIES),
                    choices=list(ALL_PROPERTIES),
                    help="properties to score")
    ap.add_argument("--references", nargs="+", default=None,
                    help="reference functionals (default: all discovered)")
    ap.add_argument("--tests", nargs="+", default=None,
                    help="test functionals (default: all discovered)")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="functionals to drop entirely")
    ap.add_argument("--include-self", action="store_true",
                    help="also score each reference against itself (d == 1)")
    ap.add_argument("--system", default=CFG["system"])
    ap.add_argument("--n-tag", default=CFG["n_tag"])
    ap.add_argument("--z-max", type=float, default=CFG["z_max"],
                    help="orientation slab-width cutoff in A (<=0 for none)")
    ap.add_argument("--sfg-width-method", default=CFG["sfg_width_method"],
                    choices=("equivalent", "fwhm", "integral"))
    ap.add_argument("--sfg-expt-path", default=CFG["sfg_expt_path"],
                    help="digitized experimental SFG csv (xaxis, Air/water, "
                        "Air/graphene/water); set to '' to disable")
    ap.add_argument("--cache", default="./.benchmark_cache.pkl",
                    help="reduced-data cache (set to '' to disable)")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    cfg = dict(CFG)
    cfg["system"] = args.system
    cfg["n_tag"] = args.n_tag
    cfg["z_max"] = None if args.z_max is not None and args.z_max <= 0 else args.z_max
    cfg["sfg_width_method"] = args.sfg_width_method
    cfg["sfg_expt_path"] = args.sfg_expt_path or None

    interf_root = os.path.abspath(args.interf_root)
    bulk_root = os.path.abspath(args.bulk_root)
    if not os.path.isdir(interf_root):
        ap.error(f"interfacial root not found: {interf_root}")

    functionals = discover_functionals(interf_root, cfg["system"],
                                       exclude=set(args.exclude))
    if not functionals:
        ap.error(f"no functionals with gra-runs/frozen/{cfg['system']} "
                 f"under {interf_root}")

    references = args.references or functionals
    tests = args.tests or functionals
    unknown = [f for f in set(references) | set(tests) if f not in functionals]
    if unknown:
        ap.error(f"unknown functional(s): {', '.join(sorted(unknown))}")

    if verbose:
        print(f"Interfacial root : {interf_root}")
        print(f"Bulk root        : {bulk_root}")
        print(f"System / n_tag   : {cfg['system']} / {cfg['n_tag']}")
        print(f"Properties       : {', '.join(args.properties)}")
        print(f"Functionals      : {len(functionals)} discovered")
        print(f"References       : {len(references)}")
        print(f"Tests            : {len(tests)}")
        print()

    needed = sorted(set(references) | set(tests))
    data = load_all(interf_root, bulk_root, needed, args.properties, cfg,
                    cache_path=(args.cache or None),
                    refresh=args.refresh_cache, verbose=verbose)

    results = run_all_vs_all(data, references, tests, args.properties, cfg,
                             include_self=args.include_self, verbose=verbose)

    exp_results = None
    if "sfg" in args.properties:
        exp_results = run_experimental_sfg(data, tests, cfg, verbose=verbose)

    write_outputs(results, args.out_dir, args.properties, cfg, verbose=verbose,
                  exp_results=exp_results)
    if verbose:
        print_consensus(results, args.properties)
    return 0


if __name__ == "__main__":
    sys.exit(main())
