# Microscopic simulation of free riding speed dynamics in bicycle traffic - Paper Companion Code

This repository contains the simulation code used for the analysis reported in *[paper ref]*. 
It is shared as a reference implementation of the free-riding models described in the paper.

The code is tied to the specific datasets, networks, and study sites (Linköping, SE and Wuppertal, DE).
It is published to make the modelling choices, the SUMO/TraCI integration, and the implementation details available 
for readers of the paper.

## Contents

| File | Purpose                                                                                                                                                                                                                                                                                                                               |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `simulationSumo.py` | **Core implementation.** Defines:<br/> `getKinEnergy` (bicycle dynamics),<br/> `getBeta` (extracts fixed + random effects from a fitted mixed-effects model), <br/>`create_demand` (writes the SUMO route file for a single bicyclist), and<br/> `run_simulation` (the TraCI loop implementing all model variants). |
| `simulationSumoMain.py` | Loads input data, sets per-location parameters, and calls `run_simulation` for each model variant. Provided as a reference for how the inputs are structured, organised, and called for this paper.                                                                                                                                   |

## Models

The five model variants are toggled with boolean flags in
`simulationSumoMain.py` and selected inside `run_simulation`:

- `krauss_approach` — Default SUMO Krauss car-following.
- `kraussPS_approach` — Krauss car-following model with gradient-adjusted acceleration.
- `speedDG_approach` — Context-based speed distributions.
- `speedME_approach` — (Mixed-effects) Regression speed model.
- `powerME_approach` — Physics-based (mixed-effects) regression speed model, obtained from the bicycle dynamics model in `getKinEnergy`.

For each model variant reported in the paper, the corresponding flag in `simulationSumoMain.py` was set to `True`,
and the script was executed.


## Inputs

The driver script reads several pre-processed pickle files specific to the paper's datasets. 

```
  Data.pkl                                     # filtered observed trajectory data (only free riding)
  DataFull.pkl                                 # full observed trajectory data (incl. free and constrained riding) 
  ParamsUser.pkl                               # per-bicyclist parameters (mass, CdA, Crr, desired speed, desired power, …)
  CharsGrade.pkl                               # speed distributions per gradient bin
  GradeSegmentsUTransition_{Lkpg,Wupp}.pkl     # id locations of downhill-to-uphill transitions   
  IntersectionSegments_{Lkpg,Wupp}.pkl         # id locations of intersections
  SUMO_Networks/{Linköping,Wuppertal}/         # network files
  Speed_model_*.pkl                            # fitted mixed-effects for speed, for each location
  Power_model_*.pkl                            # fitted mixed-effects for power, for each location
```

The fitted mixed-effects models are pandas objects with the columns `FixedEffects` and `RandomEffects`, 
indexed by location. See `getBeta` in `simulationSumo.py` for the exact structure used.

## Output

For each model variant and location, two pickles are written:
- `SimData_<Location>.pkl` — full trajectory (time, position, speed, acceleration, power, kinetic energy, and route features).
- `SimDuration_<Location>.pkl` — simulation duration per bicyclist.

## Citation

If you refer to this code, please cite the paper:

> *[full citation, DOI]*
