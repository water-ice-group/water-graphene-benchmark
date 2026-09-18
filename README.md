# Multi-Observable Scoring Framework

Interactive companion to *How Good Is DFT for Solid-Liquid Interfaces? A Comparison With the Random-Phase Approximation for Water on Graphene.*

**[Open the interactive framework →](https://xradvincula.github.io/graphene-water-benchmark/)**

We assess a broad set of DFT exchange-correlation (XC) approximations against the random-phase approximation (RPA) at the graphene-water interface, using machine-learned potentials trained on both levels of theory to reach the trajectories needed for interfacial structure, wettability, friction, and vibrational sum-frequency generation. To make sense of this landscape, each functional's fidelity to RPA is distilled into a single measure, built by averaging six observable-level scores: radial distribution functions, vibrational density of states, density profiles, orientational order, VSFG spectra, and friction.

This page reproduces Figures 4B, 4C, 4D, and S11 of the manuscript interactively. The six observables can be reweighted individually or via preset weightings, and the resulting individual, structure, dynamics, and overall scores update accordingly for every XC approximation considered, against RPA/QZ as the reference throughout.

## Reference

RPA/QZ, used as ground truth throughout the manuscript.

## Data

All data required to reproduce the findings of this work are made openly available in this repository.
