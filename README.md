# Optimal action selection for cost reduction in maintenance triage

This repository contains the code for our paper "Optimal action selection for cost reduction in maintenance triage". It reproduces the reported decision calculations from the saved outputs of sixty predictors.

## Requirements

We tested the code with Python 3.12.15, NumPy 2.5.2 and matplotlib 3.10.8 on Linux. Install the dependencies with:

```sh
pip install -r requirements.txt
```

## Data

We use the public [NASA C-MAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data), [UCI Hydraulic](https://archive.ics.uci.edu/dataset/447) and [CWRU bearing](https://engineering.case.edu/bearingdatacenter) datasets. The Hydraulic data are distributed under CC BY 4.0 (Helwig, Pignanelli and Schütze, 2015, DOI 10.24432/C5CW21).

`data/predictions.npz` contains class probabilities, recorded labels, unit identifiers and cycle indices. Probability columns are Critical, Warning, Normal. Labels use 0 for Normal, 1 for Warning and 2 for Critical. `data/inputs.json` lists the predictors and the master seed for 256 whole-unit allocations per predictor. Supplementary Section S11 describes the label mappings, fitting partitions and predictor settings.

The code renormalizes each probability row to sum to one before forming the prediction sets (Supplementary Section S5).

## Usage

To reproduce the tables, figures and numerical results, run the following command from this directory:

```sh
python run.py --out results --workers 2
```

You can change `--workers` to use the available CPU cores. The complete calculation took 376 seconds with two workers, each using one numerical thread, on our Linux server.

## Results

Numerical results are written to `results/numbers.json`, tables to `results/tables/` and the ten figures to `results/figures/`, using the manuscript figure filenames.

| Results | Calculation |
|---|---|
| Table 1 and Table S2 | `ress/nested.py` |
| Table 2 and Fig. S3 | `ress/price.py` |
| Table S1 and Fig. S2 | `ress/frontier.py`, sensitivity grid |
| Tables S4–S5 and Figs. 2–3, S1, S4–S7 | `ress/frontier.py`, capacity and charge grids |
| Hydraulic guard results in Sections S7–S8 | `ress/hydraulic.py` |

`ress/figures.py` draws all ten figures, including the action-selection diagram in Fig. 1.

## License

This code is released under the MIT license.

## Contact

Yeosun Kyung, yeosun.kyung@yonsei.ac.kr
