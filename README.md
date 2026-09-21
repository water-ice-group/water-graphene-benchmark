# Multi-Observable Scoring Framework

Interactive companion to *How Good Is DFT for Solid-Liquid Interfaces? A Comparison With the Random-Phase Approximation for Water on Graphene.* by Xavier R. Advincula, Yair Litman, Jiuyang Shi, Flaviano Della Pia, Christoph Schran, and Angelos Michaelides.

**[Open the interactive framework →](https://water-ice-group.github.io/water-graphene-benchmark/)**

We assess a broad set of DFT exchange-correlation (XC) approximations against the random-phase approximation (RPA) at the graphene-water interface. Machine-learned potentials trained on both levels of theory give the trajectory lengths needed for interfacial structure, wettability, friction and vibrational sum-frequency generation. Each functional's fidelity to RPA is summarised in a single measure, the average of six observable-level scores: radial distribution functions, vibrational density of states, density profiles, orientational order, VSFG spectra and friction.

This page reproduces Figures 4B, 4C, 4D and S11 of the manuscript interactively. The six observables can be reweighted individually or through a set of preset weightings, and the individual, structure, dynamics and overall scores update accordingly for every XC approximation considered.

RPA/QZ is the reference by default, but it is not the only one. Every score is computed directly from the simulated observables, so a different reference means re-scoring against it, which `benchmark_all_references.py` does for every pair at once. `export_web_data.py --refs all --from-benchmark <its output dir>` turns that matrix into one `data/scores_<name>.json` per reference, and the reference list in the console loads them. All 24 levels of theory are live, and so is experiment. `benchmark_all_references.py` also writes `scores_experimental.csv`, scoring every functional's VSFG against the digitised HD-VSFG spectra of Wang et al., *Angew. Chem. Int. Ed.* **63**, e202319503 (2024). Only the VSFG spectra are measured, so that payload carries one observable and the page collapses to scoring on it alone: one slider, no structure column, and the presets that would zero it are hidden.

## How a score is computed

All six observables share one measure of agreement, the symmetric (Bray-Curtis) norm of Eq. S11. For a profile or a spectrum it runs over the curve,

    d = 1 - sum|a - b| / sum(|a| + |b|)

and for a scalar, such as the friction coefficient or a VSFG peak position, it is the same expression on a single pair of numbers,

    d = 1 - |a - b| / (|a| + |b|)

It is symmetric in its two arguments and bounded by 0 and 1, so no observable is scored on a scale of its own, and swapping which level of theory plays the reference does not change the size of the disagreement.

The VSFG score follows from that norm too. Each spectrum is reduced to three physical descriptors, the hydrogen-bonded and dangling O-H peak positions, their intensity ratio, and the width of the hydrogen-bonded band. The scalar form above turns each descriptor into an agreement, and the three are then averaged. This replaces an earlier, asymmetric definition of the VSFG agreement: the descriptors now use the same normalisation as friction and as the curve-based observables. VSFG scores are uniformly higher under it, so the overall scores on the page sit a few points above what the earlier export gave. `_agreement` in `benchmark_all_references.py` is the definition.

Orientation is scored on the population-weighted profile. Each bin enters weighted by the water population N(z) of the reference, and the comparison stops 11 Å from the sheet, so the air-water side of the film does not dominate a score that is about the solid-liquid interface. Every score on the page, orientation included, comes from that all-vs-all run rather than from the numbers transcribed for the paper's figure, so it does not reproduce the published value exactly and a ranking can shift by a place or two.

## What is in this repository

| | |
|---|---|
| `index.html` | the whole page: markup, styles and logic in one file, with the RPA/QZ payload inlined so it also works opened straight from disk |
| `data/` | one `scores_<reference>.json` per reference, plus `manifest.json`, which is what the reference list in the console fetches |
| `benchmark_all_references.py` | the scoring itself, every ordered pair of levels of theory in one run |
| `benchmark_all_references/` | that run's output, trimmed to the two files the export reads |
| `export_web_data.py` | turns the run's CSVs into `data/` |
| `sync_inline_baseline.py` | keeps the payload inlined in `index.html` in step with `data/` |

Serving the folder is enough to run the page locally:

```sh
python -m http.server
```

Opening `index.html` from the filesystem also works, but only for RPA/QZ. Browsers refuse the `fetch` of `data/` from a `file://` page, so the other references need a server.

## Regenerating the data

The page is a read of `benchmark_all_references.py`'s all-vs-all matrix, so a change to the scoring means re-running it and re-exporting:

```sh
python export_web_data.py --refs all --from-benchmark benchmark_all_references --relative-paths
python sync_inline_baseline.py
```

Those two steps need nothing but the standard library, and they read only `benchmark_all_references/pairwise_scores.csv` and `scores_experimental.csv`, so the whole of `data/` can be rebuilt from what is committed here. `--relative-paths` keeps the recorded paths free of whichever machine did the run. Drop it and provenance records absolute ones instead.

Re-running the scoring itself is the step this repository cannot reproduce on its own:

```sh
python benchmark_all_references.py          # writes benchmark_all_references/
```

It reads the analysed trajectories (`17_FINAL_PROD_out`, `12_analyse_final_prods`), which are far too large to publish. The committed CSVs are its output.

The sync step is not optional. `index.html` carries the RPA/QZ payload inline, as a `const BASELINE = {...}` literal, so the page works opened straight from disk. It seeds its reference cache from that literal and never re-reads `data/scores_RPA-QZ.json`, so exporting without syncing leaves the default view showing the old numbers while every other reference shows the new ones. Run `python sync_inline_baseline.py --check` before publishing: it reports drift and exits non-zero without writing.

`PBE-DRSLL` and `SCAN` stay out of the page, as they are out of Figure 4. `PBE-DRSLL` is in the matrix, so dropping it from `EXCLUDE` in `export_web_data.py` brings it in.
