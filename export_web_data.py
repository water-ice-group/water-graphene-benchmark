#!/usr/bin/env python3
"""
Export the interactive scorecard's data file.

Run this where the raw analysis lives (the machine that holds
``17_FINAL_PROD_out`` and ``12_analyse_final_prods``).  For each reference you
ask for, it re-runs the six preprocess notebooks and writes one JSON that
``index.html`` picks up from its reference list (or through the "Load a
scores.json" control, for a file that is not sitting in ``data/``).

Why a re-run is unavoidable
---------------------------
Each individual score is computed directly from the simulated observables,
using the norm of Eq. S11,

    d = 1 - sum|a - b| / sum(|a| + |b|)

so the score of an XC approximation against RPA/QZ carries no information
about its score against revPBE-D3(BJ). Scoring against a different reference
means recomputing from the analysis pickles. Nothing here approximates that.

Any level of theory in the study can play the reference, and so can
experiment: ``--refs all`` walks every XC approximation plus ``expt``.  Only
the VSFG spectra are measured, so the ``expt`` payload carries that one
observable and the page scores on it alone.

Usage
-----
    # sitting next to the preprocess-*.ipynb files
    python export_web_data.py --refs RPA-QZ revPBE-D3-BJ RPA

    # every reference the page offers: all 24 XC approximations plus experiment
    python export_web_data.py --refs all

    # reuse scoring directories you have already produced, no re-run
    python export_web_data.py --refs RPA-QZ --reuse

    # only rebuild two metrics (handy while iterating)
    python export_web_data.py --refs RPA-QZ --metrics density friction

Output
------
    data/scores_<REF>.json      one file per reference, which index.html fetches
    data/scores.json            the first reference, as the default drop-in
    data/manifest.json          which references exist, so the page can say so

Copy that directory next to ``index.html`` and the reference list in the
console becomes live.

Each file carries, per XC approximation: the six individual scores with their
per-observable standard errors, every sub-score the tidy CSVs expose (so the
page can offer a sub-observable breakdown), and the kappa values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import inspect
import os
import re
import sys
import traceback
from datetime import date

# ---------------------------------------------------------------------------
# Where things are.  Override any of these on the command line.
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    notebook_dir=".",
    interf_root="../../17_FINAL_PROD_out",
    bulk_root="../../12_analyse_final_prods",
    system="19A",
    n_tag="n100",
    out_dir="data",
    scoring_root=".",
    expt_dir="./experimental-data-sfg",
)

# Experiment is a reference like any other, except that only the VSFG spectra
# are measured. The id is the label benchmark_all_references.py writes into
# scores_experimental.csv, so the page, the file name and the source agree.
EXPT_REF = "Experimental"
EXPT_METRICS = ("sfg",)
EXPT_DISPLAY = "Experiment (VSFG)"
# Just the provenance: the page already says a one-observable reference scores
# on one observable, and saying it twice reads like a stutter.
EXPT_NOTE = ("Digitised HD-VSFG spectra from Wang et al., Angew. Chem. Int. "
             "Ed. 63, e202319503 (2024), Fig. 1D.")

# Orientation weighting, mirroring CFG in benchmark_all_references.py, which is
# what produced the published matrix.
#
# Each bin of the orientation profile enters the score weighted by its
# population N(z), taken from the reference's own runs, so the sparse regions
# cannot outvote the water that is actually there, and the comparison stops
# `x_max` angstrom from the sheet -- past that the film is answering to the
# air-water interface rather than to graphene.  `min_frac` (a floor on the
# population, relative to its peak) and `weight_power` are the extra knobs
# preprocess-orientation.ipynb offers; the production numbers use neither.
# Overridable from the command line, and whatever was applied is recorded in
# each payload's provenance.
ORIENTATION_WEIGHTING = dict(
    weight_source="ref",
    min_frac=0.0,
    weight_power=1.0,
    x_max=11.0,
)

# metric id -> (notebook, scoring function, output-dir template, extra kwargs)
#
# The signatures are taken verbatim from the __main__ blocks of each notebook.
METRIC_SPECS = {
    "rdf": dict(
        notebook="preprocess-rdfs.ipynb",
        func="scoring_structuring_all_functionals_rdf",
        out_tpl="scoring_rdf_{system}",
        roots=("interf_root", "bulk_root"),
        kwargs=dict(normalize=True),
        headline="combined",
    ),
    "vdos": dict(
        notebook="preprocess-vdos.ipynb",
        func="scoring_structuring_all_functionals_vdos",
        out_tpl="scoring_vdos_{system}",
        roots=("interf_root", "bulk_root"),
        kwargs=dict(normalize=True),
        headline="combined",
    ),
    "density": dict(
        notebook="preprocess-density.ipynb",
        func="scoring_structuring_all_functionals",
        out_tpl="scoring_density_{system}",
        roots=("root_dir",),
        kwargs=dict(
            species=("O", "H"), ratio_species="O",
            starting_c=7.5, x1=(2.5, 4.0), x2=(4.0, 7.5),
            normalize=True,
        ),
        headline="density_and_ratio",
    ),
    "orientation": dict(
        notebook="preprocess-orientation.ipynb",
        func="scoring_structuring_all_functionals_orientation",
        out_tpl="scoring_orientation_{system}",
        roots=("root_dir",),
        # filled in from ORIENTATION_WEIGHTING at run time, so the command line
        # can move the cut without the spec and the CLI drifting apart
        kwargs=dict(normalize=True),
        headline="water",
    ),
    "sfg": dict(
        # the plotting notebooks read scoring_sfgphys_*, which is what the
        # REVISIT notebook writes -- not preprocess-sfg.ipynb.
        notebook="preprocess-sfg-REVISIT.ipynb",
        func="scoring_structuring_all_functionals",
        out_tpl="scoring_sfgphys_{system}",
        roots=("root_dir",),
        kwargs=dict(ref_kind="rpa", modes=("mode2", "mode3"),
                    window=(2500, 4100), normalize=True),
        headline="combined",
    ),
    "friction": dict(
        notebook="preprocess-friction.ipynb",
        func="scoring_friction_all_functionals",
        out_tpl="scoring_friction_{system}",
        roots=("root_dir",),
        kwargs=dict(n_agg="n100_aggregate",
                    scalar_subdir="flexible", curve_subdir="flexible",
                    datafile="lambda_fft_vs_tau_average.dat",
                    window=(0.0, None)),
        headline="combined",
    ),
}

STRUCTURE = ["rdf", "density", "orientation"]
DYNAMICS = ["vdos", "sfg", "friction"]

METRIC_META = {
    "rdf": ("RDF", "#9ad83c",
            "Radial distribution functions among the liquid atoms, evaluated "
            "in bulk water and in interfacial water and averaged over the "
            "O-O, O-H and H-H species pairs."),
    "vdos": ("VDOS", "#47c06e",
             "Vibrational density of states, evaluated in bulk water and in "
             "interfacial water."),
    "density": ("Density", "#1ea087",
                "Density profiles of the liquid at the interface, including "
                "the wetting proxy rho2/rho1."),
    "orientation": ("Orientation", "#267f8e",
                    "Orientational profiles of the liquid as a function of "
                    "distance from the surface, {range}, with every bin "
                    "weighted by the water population N(z) sitting in it{floor}"
                    ", so the sparse regions cannot outvote the water that is "
                    "actually there."),
    "sfg": ("VSFG", "#355c8c",
            "VSFG spectra for the graphene-water and air-water interfaces, "
            "scored on the hydrogen-bonded and dangling O-H peak positions, "
            "their intensity ratio and the width of the hydrogen-bonded "
            "band."),
    "friction": ("Friction", "#45347f",
                 "Solid-liquid friction, scored on the plateau value of the "
                 "friction coefficient and on the full Green-Kubo running "
                 "integral."),
}

# functional_rows / row_colors, verbatim from plots-V2.ipynb cell 0.
FUNCTIONAL_ROWS = [
    [("RPA/QZ", "RPA-QZ", "#323232"), ("RPA/TZ", "RPA", "#8a8a8a")],
    [("HSE06-D3(0)", "HSE06-D3-0", "#8c1d18"),
     ("B3LYP-D3(0)", "B3LYP-D3-0", "#b22222"),
     ("revPBE0-D3(0)", "revPBE0-D3-0", "#e34a33")],
    [("M06-L-D3(0)", "M06L-D3-0", "#6a1b9a"),
     ("r2SCAN-D4", "r2SCAN-D4", "#5e60ce"),
     ("r2SCAN-D3(BJ)", "r2SCAN-D3-BJ", "#9fa8da"),
     ("r2SCAN", "r2SCAN", "#c5cae9"),
     ("B97M-rV", "B97M-rV", "#3f007d")],
    [("revPBE-D4", "revPBE-D4", "#08306b"),
     ("revPBE-D3(BJ)", "revPBE-D3-BJ", "#08519c"),
     ("revPBE-D3(0)", "revPBE-D3-0", "#2171b5"),
     ("revPBE-DRSLL", "revPBE-DRSLL", "#6baed6"),
     ("revPBE-rVV10", "revPBE-rVV10", "#c6dbef")],
    [("PBE-D4", "PBE-D4", "#00441b"),
     ("PBE-D3(BJ)", "PBE-D3-BJ", "#006d2c"),
     ("PBE-D3M(BJ)", "PBE-D3m-BJ", "#238b45"),
     ("PBE-D3(0)", "PBE-D3-0", "#41ab5d"),
     ("PBE-rVV10", "PBE-rVV10", "#c7e9c0")],
    [("BLYP-D4", "BLYP-D4", "#7f2704"),
     ("BLYP-D3(BJ)", "BLYP-D3-BJ", "#a63603"),
     ("BLYP-D3(0)", "BLYP-D3-0", "#d94801"),
     ("optB88-vdW", "optB88-DRSLL", "#3182bd")],
]
ROW_TO_FAMILY = [0, 1, 2, 3, 3, 3]
FAMILIES = [("RPA", "star"), ("Hybrids", "diamond"),
            ("Meta-GGAs", "square"), ("GGAs", "circle")]

EXCLUDE = {"PBE-DRSLL", "SCAN"}

# fig 4b's four blocks, from the `group` switch in plots-V2.ipynb cell 3.
PANEL_GROUPS = [
    ("Hybrids", ["HSE06-D3-0", "B3LYP-D3-0", "revPBE0-D3-0"]),
    ("Meta-GGAs", ["M06L-D3-0", "r2SCAN-D4", "r2SCAN-D3-BJ", "r2SCAN"]),
    ("Meta-GGAs", ["B97M-rV"]),
    ("GGAs", ["revPBE-D4", "revPBE-D3-0", "PBE-D3-BJ",
              "PBE-rVV10", "BLYP-D3-BJ", "optB88-DRSLL"]),
]

DISPERSION_HATCHES = {"none": "", "D3-0": "fwd", "D3-BJ": "back",
                      "D3m-BJ": "back", "D4": "fwd-dense",
                      "DRSLL": "horiz", "rVV10": "horiz"}


def metric_detail(metric: str, cfg: dict) -> str:
    """The metric blurb the page shows, with the orientation cut filled in."""
    detail = METRIC_META[metric][2]
    if metric != "orientation":
        return detail
    weighting = cfg["orientation_weighting"]
    x_max, min_frac = weighting.get("x_max"), weighting.get("min_frac") or 0.0
    return detail.format(
        range=(f"compared out to {x_max:g} \u00c5 from the sheet" if x_max
               else "compared over the whole film"),
        floor=(f" and bins below {min_frac:g} of the peak population dropped"
               if min_frac > 0 else ""))


def dispersion_of(fn: str) -> str:
    """get_dispersion() from plots-V2.ipynb, unchanged."""
    if fn in ("RPA", "r2SCAN", "RPA-QZ"):
        return "none"
    if "D3m-BJ" in fn or "D3M(BJ)" in fn:
        return "D3m-BJ"
    if "D3-BJ" in fn or "D3(BJ)" in fn:
        return "D3-BJ"
    if "D3-0" in fn or "D3(0)" in fn:
        return "D3-0"
    if "D4" in fn:
        return "D4"
    if "DRSLL" in fn:
        return "DRSLL"
    if "rVV10" in fn or fn == "B97M-rV":
        return "rVV10"
    return "none"


def build_lookups():
    meta = {}
    for ri, row in enumerate(FUNCTIONAL_ROWS):
        for display, folder, color in row:
            meta[folder] = dict(display=display, color=color,
                                family=FAMILIES[ROW_TO_FAMILY[ri]][0])
    meta[EXPT_REF] = dict(display=EXPT_DISPLAY, color="#323232",
                          family="Experiment")
    return meta


def reference_roster() -> list[tuple[str, str]]:
    """Every reference the page offers: each XC approximation, then experiment.

    Kept in the paper's row order so the console list reads like Figure 4.
    """
    out = [(folder, display)
           for row in FUNCTIONAL_ROWS for display, folder, _ in row]
    out.append((EXPT_REF, EXPT_DISPLAY))
    return out


def metrics_for(ref: str, wanted) -> list[str]:
    """Which observables exist against this reference."""
    if ref == EXPT_REF:
        return [m for m in wanted if m in EXPT_METRICS]
    return list(wanted)


# ---------------------------------------------------------------------------
# Running a notebook's scoring function without touching the notebook
# ---------------------------------------------------------------------------

def load_notebook_namespace(path: str) -> dict:
    """Execute a notebook's code cells and hand back the resulting namespace.

    ``__name__`` is set to the notebook's own name, so the ``if __name__ ==
    "__main__"`` block at the bottom of each notebook is skipped and only the
    definitions land.  Cells are executed in order, so where a notebook defines
    the same function twice the later definition wins -- exactly what happens
    when the notebook is run top to bottom.
    """
    with open(path, "r") as f:
        nb = json.load(f)

    ns = {"__name__": "nb_" + re.sub(r"\W", "_", os.path.basename(path)),
          "__file__": os.path.abspath(path)}
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        # skip IPython magics and shell escapes, which aren't valid Python
        src = "\n".join("" if re.match(r"\s*[!%]", ln) else ln
                        for ln in src.split("\n"))
        try:
            exec(compile(src, f"{path}#cell{i}", "exec"), ns)
        except Exception:
            print(f"  [warn] {os.path.basename(path)} cell {i} raised:",
                  file=sys.stderr)
            traceback.print_exc(limit=3)
    return ns


def filter_kwargs(fn, kwargs: dict, where: str, strict_keys=()) -> dict:
    """Drop kwargs the notebook's function does not take.

    The notebooks move faster than this script, so an argument that has been
    renamed should not take the whole export down.  Anything in `strict_keys`
    is different: those change *what is being computed*, so dropping one
    silently would hand back numbers that are not what was asked for.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    kept, dropped = {}, []
    for k, v in kwargs.items():
        if k in params:
            kept[k] = v
        else:
            dropped.append(k)
    for k in dropped:
        if k in strict_keys:
            raise TypeError(
                f"{where} does not accept `{k}`, which changes the score "
                f"itself. Update the notebook, or pass a value this script "
                f"does not have to forward."
            )
        print(f"  [warn] {where} does not accept `{k}`; ignoring it")
    return kept


def run_metric(metric: str, ref: str, cfg: dict) -> str:
    """Re-score one observable against `ref` and return its output directory."""
    spec = METRIC_SPECS[metric]
    nb_path = os.path.join(cfg["notebook_dir"], spec["notebook"])
    if not os.path.exists(nb_path):
        raise FileNotFoundError(nb_path)

    out_dir = os.path.join(
        cfg["out_dir"], "scoring", ref,
        spec["out_tpl"].format(system=cfg["system"]),
    )
    os.makedirs(out_dir, exist_ok=True)

    ns = load_notebook_namespace(nb_path)
    fn = ns.get(spec["func"])
    if fn is None:
        raise AttributeError(
            f"{spec['notebook']} does not define {spec['func']}()")

    kwargs = dict(spec["kwargs"])
    kwargs.update(rpa_name=ref, system=cfg["system"], n_tag=cfg["n_tag"],
                  plot=False, out_dir=out_dir, verbose=cfg["verbose"])

    strict: tuple = ()
    if metric == "orientation":
        # the weighting *is* the metric here, so these may not be dropped
        kwargs.update(cfg["orientation_weighting"])
        strict = tuple(cfg["orientation_weighting"])
    if ref == EXPT_REF:
        # experiment is not a folder under interf_root: the SFG notebook reads
        # the digitised spectra instead
        kwargs.pop("rpa_name", None)
        kwargs.update(ref_kind="expt", expt_dir=cfg["expt_dir"])

    if spec["roots"] == ("interf_root", "bulk_root"):
        kwargs.update(interf_root=cfg["interf_root"],
                      bulk_root=cfg["bulk_root"])
    else:
        kwargs.update(root_dir=cfg["interf_root"])
    if metric == "friction":
        # friction aggregates instead of using n_tag
        kwargs.pop("n_tag", None)
        kwargs.pop("verbose", None)

    kwargs = filter_kwargs(fn, kwargs,
                           f"{spec['notebook']}:{spec['func']}()",
                           strict_keys=strict)

    print(f"  {metric:12s} -> {out_dir}")
    fn(**kwargs)
    return out_dir


# ---------------------------------------------------------------------------
# Reading the tidy CSVs back
# ---------------------------------------------------------------------------

def read_tidy(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def collect(metric: str, scoring_dir: str) -> dict:
    """functional -> {headline: (d, sem), subs: {...}, kappa: ...}"""
    rows = read_tidy(os.path.join(scoring_dir, "summary_tidy.csv"))
    headline = METRIC_SPECS[metric]["headline"]
    out: dict[str, dict] = {}
    for r in rows:
        fn = r.get("functional")
        if not fn:
            continue
        rec = out.setdefault(fn, {"d": None, "sem": None,
                                  "subs": {}, "kappa": None})
        name = r.get("metric", "")
        # the rdf/vdos tidy files carry an extra `condition` column
        cond = r.get("condition")
        key = f"{cond}:{name}" if cond else name
        kind = r.get("kind")
        val, sem = fnum(r.get("value")), fnum(r.get("sem"))

        if kind == "d":
            rec["subs"][key] = {"value": val, "sem": sem}
            # headline: the `overall`/unconditioned `combined` row
            if name == headline and cond in (None, "", "overall"):
                rec["d"], rec["sem"] = val, sem
        elif kind == "kappa" and name == headline and cond in (None, "", "overall"):
            rec["kappa"] = val
        elif kind in ("position", "ratio", "width"):
            rec["subs"][f"{key}:{kind}"] = {"value": val, "sem": sem}
    return out


# ---------------------------------------------------------------------------
# Reading benchmark_all_references.py's matrix
#
# That script sweeps the reference over every functional in one pass, so its
# pairwise_scores.csv already holds what a reference-by-reference re-run of the
# notebooks would produce. Reading it is the same numbers without the compute.
# ---------------------------------------------------------------------------

BENCHMARK_CSV = "pairwise_scores.csv"
BENCHMARK_EXPT_CSV = "scores_experimental.csv"


def read_benchmark(path: str) -> dict:
    """The benchmark CSVs -> {reference: {metric: {functional: rec}}}.

    Reads pairwise_scores.csv and, when it sits alongside,
    scores_experimental.csv: same columns, one fixed reference, VSFG only.

    Each property's headline is whichever row the script marked `is_primary`,
    so the choice stays with the script that did the scoring. Every other row
    for that property is kept as a sub-metric.
    """
    paths = []
    if os.path.isdir(path):
        paths.append(os.path.join(path, BENCHMARK_CSV))
        expt = os.path.join(path, BENCHMARK_EXPT_CSV)
        if os.path.exists(expt):
            paths.append(expt)
    else:
        paths.append(path)
    if not os.path.exists(paths[0]):
        raise FileNotFoundError(paths[0])

    table: dict = {}
    for one in paths:
        _read_benchmark_csv(one, table)
    return table


def _read_benchmark_csv(path: str, table: dict) -> dict:
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            prop = r.get("property")
            if prop not in METRIC_SPECS:
                continue
            ref, test = r.get("reference"), r.get("test")
            if not ref or not test:
                continue
            rec = (table.setdefault(ref, {})
                        .setdefault(prop, {})
                        .setdefault(test, {"d": None, "sem": None,
                                           "subs": {}, "kappa": None}))
            val, sem = fnum(r.get("d")), fnum(r.get("d_sem"))
            rec["subs"][r.get("metric", "")] = {"value": val, "sem": sem}
            if r.get("is_primary") == "1":
                rec["d"], rec["sem"] = val, sem
                rec["kappa"] = fnum(r.get("kappa"))
    return table


def prov_path(path: str, cfg: dict) -> str:
    """How a filesystem path is recorded in provenance.

    Absolute by default, which is what you want in a working copy: it says
    exactly which tree the numbers came from.  Under ``--relative-paths`` it is
    made relative to the current directory instead, so a payload published to a
    public repository does not carry the author's home directory around.  A
    path outside the tree falls back to its basename.
    """
    if not cfg.get("relative_paths"):
        return os.path.abspath(path)
    try:
        rel = os.path.relpath(os.path.abspath(path))
    except ValueError:                      # different drive, on Windows
        return os.path.basename(path)
    return os.path.basename(path) if rel.startswith(os.pardir + os.sep) else rel


def build_payload(ref: str, scoring_dirs: dict, cfg: dict,
                  per_metric: dict | None = None) -> dict:
    meta = build_lookups()
    if per_metric is None:
        per_metric = {m: collect(m, d) for m, d in scoring_dirs.items()}
    metrics = [m for m in METRIC_SPECS if m in per_metric]

    names = sorted({fn for tbl in per_metric.values() for fn in tbl}
                   - EXCLUDE - {ref})

    functionals = []
    for fn in names:
        d, sem, subs, kappa = {}, {}, {}, {}
        for m in metrics:
            rec = per_metric[m].get(fn)
            if rec is None or rec["d"] is None:
                continue
            d[m] = rec["d"]
            sem[m] = rec["sem"] if rec["sem"] is not None else 0.0
            kappa[m] = rec["kappa"]
            subs[m] = rec["subs"]
        if not d:
            continue
        info = meta.get(fn, {})
        present = [d[m] for m in metrics if m in d]
        functionals.append(dict(
            id=fn,
            display=info.get("display", fn),
            color=info.get("color", "#7f7f7f"),
            family=info.get("family", "GGAs"),
            dispersion=dispersion_of(fn),
            d=d, sem=sem, kappa=kappa, submetrics=subs,
            sem_is_estimated=False,
            overall_equal_weight=(sum(present) / len(present) * 100.0
                                  if len(present) == len(metrics) else None),
        ))

    # The page lists every reference, whether or not it has been exported yet,
    # and says which ones are actually on disk next to this file.
    refs = []
    for rid, rdisp in reference_roster():
        if rid == ref:
            refs.append(dict(id=rid, display=rdisp, status="loaded"))
            continue
        fname = f"scores_{rid}.json"
        entry = dict(
            id=rid, display=rdisp, file=f"data/{fname}",
            status=("available"
                    if os.path.exists(os.path.join(cfg["out_dir"], fname))
                    else "needs-export"),
        )
        if rid == EXPT_REF:
            entry["metrics"] = list(EXPT_METRICS)
        refs.append(entry)

    return {
        "schema": "solliq-scores/1",
        "reference": {"id": ref,
                      "display": meta.get(ref, {}).get("display", ref),
                      "note": (EXPT_NOTE if ref == EXPT_REF else
                               "Reference by construction; pinned at 100.")},
        "provenance": {
            "source": cfg.get("source", "export_web_data.py"),
            "generated": date.today().isoformat(),
            "system": cfg["system"], "n_tag": cfg["n_tag"],
            "graphene": "frozen",
            "reference_kind": "expt" if ref == EXPT_REF else "rpa",
            "orientation_weighting": dict(cfg["orientation_weighting"]),
            "interf_root": prov_path(cfg["interf_root"], cfg),
            "bulk_root": prov_path(cfg["bulk_root"], cfg),
            "scoring_dirs": {m: prov_path(p, cfg)
                             for m, p in scoring_dirs.items()},
            "preferred_metric_per_dir": {m: METRIC_SPECS[m]["headline"]
                                         for m in metrics},
            "sem_is_exact": True,
            "sem_note": ("Standard errors are the per-observable values the "
                         "scoring wrote out, propagated in quadrature under "
                         "the chosen weights. They are per-observable, so "
                         "they no longer share one value the way the "
                         "hand-transcribed baseline did."),
            "excluded": sorted(EXCLUDE),
        },
        "metrics": [
            {"id": m, "label": METRIC_META[m][0], "color": METRIC_META[m][1],
             "group": "structure" if m in STRUCTURE else "dynamics",
             "detail": metric_detail(m, cfg)}
            for m in metrics
        ],
        "families": [{"id": n, "marker": mk} for n, mk in FAMILIES],
        "dispersion_hatches": DISPERSION_HATCHES,
        "panel_groups": [{"label": lbl, "members": mem}
                         for lbl, mem in PANEL_GROUPS],
        "available_references": refs,
        "functionals": functionals,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", default=["RPA-QZ"],
                    help="reference folder names to score against, or `all` "
                         "for every XC approximation plus experiment "
                         "(default: RPA-QZ)")
    ap.add_argument("--metrics", nargs="+", default=list(METRIC_SPECS),
                    choices=list(METRIC_SPECS),
                    help="subset of observables to rebuild")
    ap.add_argument("--notebook-dir", default=DEFAULTS["notebook_dir"])
    ap.add_argument("--interf-root", default=DEFAULTS["interf_root"],
                    help="the 17_FINAL_PROD_out tree")
    ap.add_argument("--bulk-root", default=DEFAULTS["bulk_root"],
                    help="the 12_analyse_final_prods tree (rdf/vdos bulk)")
    ap.add_argument("--system", default=DEFAULTS["system"])
    ap.add_argument("--n-tag", default=DEFAULTS["n_tag"])
    ap.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    ap.add_argument("--reuse", action="store_true",
                    help="do not re-run the notebooks; read the existing "
                         "scoring_*_<system> directories under --scoring-root")
    ap.add_argument("--scoring-root", default=DEFAULTS["scoring_root"],
                    help="where the existing scoring_* dirs live, for --reuse")
    ap.add_argument("--from-benchmark", metavar="DIR_OR_CSV",
                    help="read benchmark_all_references.py's pairwise_scores.csv "
                         "instead of re-running the notebooks: it already holds "
                         "every reference, so this is the same numbers without "
                         "the compute")
    ap.add_argument("--expt-dir", default=DEFAULTS["expt_dir"],
                    help="digitised experimental VSFG spectra, for --refs expt")
    ap.add_argument("--orient-weight-source",
                    default=ORIENTATION_WEIGHTING["weight_source"],
                    choices=("ref", "both", "none"),
                    help="whose population weights the orientation bins")
    ap.add_argument("--orient-min-frac", type=float,
                    default=ORIENTATION_WEIGHTING["min_frac"],
                    help="drop orientation bins below this fraction of the "
                         "peak population (0 disables the cut)")
    ap.add_argument("--orient-weight-power", type=float,
                    default=ORIENTATION_WEIGHTING["weight_power"],
                    help="exponent on the normalised orientation weights")
    ap.add_argument("--orient-x-max", type=float,
                    default=ORIENTATION_WEIGHTING["x_max"],
                    help="score the orientation profile only out to this "
                         "distance from the sheet, in angstrom")
    ap.add_argument("--relative-paths", action="store_true",
                    help="record paths in provenance relative to the current "
                         "directory rather than absolute, so a payload meant "
                         "for a public repository carries no local layout")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    benchmark = read_benchmark(a.from_benchmark) if a.from_benchmark else None

    refs = a.refs
    if len(refs) == 1 and refs[0].lower() == "all":
        refs = [rid for rid, _ in reference_roster()]
        if benchmark is not None:
            # experiment is not in that matrix, and neither is anything the
            # matrix was not asked to sweep
            refs = [r for r in refs if r in benchmark]

    orientation_weighting = dict(
        weight_source=a.orient_weight_source,
        min_frac=a.orient_min_frac,
        weight_power=a.orient_weight_power,
        x_max=a.orient_x_max,
    )

    cfg = dict(notebook_dir=a.notebook_dir, interf_root=a.interf_root,
               bulk_root=a.bulk_root, system=a.system, n_tag=a.n_tag,
               out_dir=a.out_dir, verbose=not a.quiet, all_refs=refs,
               expt_dir=a.expt_dir, relative_paths=a.relative_paths,
               orientation_weighting=orientation_weighting)
    if benchmark is not None:
        cfg["source"] = ("benchmark_all_references.py, read back from "
                         + prov_path(a.from_benchmark, cfg))
    os.makedirs(a.out_dir, exist_ok=True)

    written = []
    for ref in refs:
        print(f"\n=== reference: {ref} ===")
        wanted = metrics_for(ref, a.metrics)
        if ref == EXPT_REF and not wanted:
            print("  experiment only has VSFG, and --metrics excludes it")
            continue
        if ref == EXPT_REF and set(a.metrics) - set(EXPT_METRICS):
            print("  only VSFG is measured, so that is all this payload holds")

        if benchmark is not None:
            per_metric = {m: t for m, t in benchmark.get(ref, {}).items()
                          if m in wanted}
            if not per_metric:
                print(f"  {ref} is not a reference in that matrix")
                continue
            payload = build_payload(ref, {}, cfg, per_metric=per_metric)
            path = os.path.join(a.out_dir, f"scores_{ref}.json")
            with open(path, "w") as f:
                json.dump(payload, f, indent=1)
            written.append(path)
            print(f"  {len(payload['functionals'])} functionals, "
                  f"{len(payload['metrics'])} observables -> {path}")
            continue

        scoring_dirs = {}
        if a.reuse and len(refs) > 1:
            print("  [warn] --reuse reads one set of scoring_* directories, "
                  "which can only belong to one reference")
        for m in wanted:
            tpl = METRIC_SPECS[m]["out_tpl"].format(system=a.system)
            if a.reuse:
                d = os.path.join(a.scoring_root, tpl)
                if not os.path.exists(os.path.join(d, "summary_tidy.csv")):
                    print(f"  [skip] {m}: no summary_tidy.csv in {d}")
                    continue
                print(f"  {m:12s} <- {d} (reused)")
                scoring_dirs[m] = d
            else:
                try:
                    scoring_dirs[m] = run_metric(m, ref, cfg)
                except Exception as e:
                    print(f"  [fail] {m}: {e}")

        if not scoring_dirs:
            print(f"  nothing to export for {ref}")
            continue

        payload = build_payload(ref, scoring_dirs, cfg)
        path = os.path.join(a.out_dir, f"scores_{ref}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)
        written.append(path)
        n = len(payload["functionals"])
        print(f"  wrote {path}  ({n} functionals, "
              f"{len(payload['metrics'])} metrics)")

    if not written:
        print("\nNothing written.")
        return 1

    default = os.path.join(a.out_dir, "scores.json")
    with open(written[0]) as src, open(default, "w") as dst:
        dst.write(src.read())

    # The page reads the manifest on load so it can mark the references that
    # have not been exported yet, instead of finding out by a failed fetch.
    manifest = {
        "generated": date.today().isoformat(),
        "default": os.path.basename(written[0]),
        "references": [],
    }
    for rid, rdisp in reference_roster():
        fname = f"scores_{rid}.json"
        if not os.path.exists(os.path.join(a.out_dir, fname)):
            continue
        entry = {"id": rid, "display": rdisp, "file": f"data/{fname}"}
        if rid == EXPT_REF:
            entry["metrics"] = list(EXPT_METRICS)
        manifest["references"].append(entry)
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"\nDefault drop-in: {default}")
    print(f"Manifest: {os.path.join(a.out_dir, 'manifest.json')} "
          f"({len(manifest['references'])} references on disk)")
    print(f"Copy {a.out_dir}/ next to index.html and the reference list in "
          f"the console goes live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
