# Trans-DPPLM
This is a project for summitted paper 'KNOWLEDGE TRANSFER FOR SPARSE PARTIAL LINEAR MODELS  WITH PRIVACY GUARANTEE ESTIMATION, INFERENCE AND MULTIPLE TESTING'

## Code layout

```text
code/
|-- scripts/
|   |-- simulation.py
|   |-- simulation_zero_noise.py
|   |-- plot_simulation_results.py
|   |-- plot_mse_summary.py
|   `-- real_data_experiment.py
`-- data/
    |-- airdata.csv
    `-- dataset/wea/                           # station weather data
```

The archived result directories cover the 16 simulation settings reported in
the paper: eight `(n, n0, p)` settings, four privacy-budget settings, and four
source-correlation settings. Each directory contains the available aggregated
Monte Carlo outputs (`MSE`, standardized bias, interval coverage/length,
e-values, p-values, true coefficients, and parameter metadata), plus any plots
or logs saved for that setting.

## Environment

Use Python 3.9 or later. The scripts require the following packages:

```bash
python -m pip install numpy scipy pandas matplotlib seaborn joblib patsy group-lasso scikit-learn
```

All commands below assume that the working directory is `final/code`.

## Simulation experiments

`scripts/simulation.py` is the final private simulation program. Its
configuration block defines `n`, `n0`, `p`, `epsilon`, `corr`, `Nsim`, and
`A_sizes`. Modify one setting at a time and run:

```bash
python scripts/simulation.py
```

The program writes the following arrays to the current directory:

- `MSE_all_trans.npy`
- `Bias_de_all_trans.npy` and `Bias_de_dp_all_trans.npy`
- `IR_all_trans.npy` and `cilen_all_trans.npy`
- `beta_true_all_trans.npy`
- `e_values.npy` and `p_values.npy`
- `parameter_list.npy`

Move a completed run into a clearly named directory under `results/` before
starting the next setting. The full experiment uses 500 Monte Carlo
replications and can require substantial CPU time and memory.

`scripts/simulation_zero_noise.py` produces the zero-noise benchmark
`MSE_all_trans_nonoise.npy`. Its configuration block must match the associated
private simulation setting.

## Plotting simulation results

The two plotting files preserve the scripts used to summarize the archived
NumPy arrays:

- `scripts/plot_simulation_results.py` contains the MSE, standardized-bias,
  confidence-interval, FDR, and power plotting cells.
- `scripts/plot_mse_summary.py` contains the final MSE summary plotting cells.

These files use `# %%` cells and setting-specific `paras`/`test_path` values.
Run the required cell in an IDE that supports Python cells, after setting its
path to the corresponding directory under `results/`. They are not intended to
be executed from top to bottom without selecting a setting.

## Real-data experiment

Run the real-data analysis from `final/code` so that its relative data paths
resolve correctly:

```bash
python scripts/real_data_experiment.py
```

The script reads `data/airdata.csv` and the station files in
`data/dataset/wea/`, uses station `1360A` as the target, and evaluates the nine
source stations listed in the manuscript. In the archived script, PM2.5 is the
response, PM10/SO2/NO2/O3 are linear predictors, and the weather variables are
the nonlinear inputs. The resulting paper figure is archived as
`results/real_data.png` and is also present in `fig/real_data.png`.

The stochastic routines do not set a global NumPy seed. Exact bit-for-bit
replication is therefore not expected unless a seed is added before rerunning;
the archived arrays are the results used to construct the submitted figures.
