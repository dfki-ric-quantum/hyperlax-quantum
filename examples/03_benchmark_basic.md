# End-to-End Benchmark Workflow: Running and Plotting

This guide walks you through the complete process of executing a multi-algorithm, multi-environment benchmark using the SLURM launcher, and then generating a suite of comparative plots from the results.


## 1. Running the Benchmark

The `run-benchmark` command is the primary tool for orchestrating large-scale comparisons.  
It automates running multiple hyperparameter sweeps, each configured for a specific algorithm–environment pair.

### 1.1 How to Run a Benchmark

You wrap the `hyperlax.cli run-benchmark` command with the `slurm/launcher.py` script.  
The launcher handles SLURM job submission, while `run-benchmark` defines the experiments to execute within that job.

#### Example Command

This example compares two algorithms (`ppo_mlp` and `sac_mlp`) across two environments (`gymnax.cartpole` and `gymnax.pendulum`), running a batched hyperparameter sweep of 8 samples for each combination.

```bash
# Use the SLURM launcher to submit the job
python3 slurm/launcher.py \
    --job-name "ppo_vs_sac_benchmark" \
    --time "2-00:00:00" \
    --gres "gpu:1" \
    --singularity-image "/path/to/hyperlax.sif" \
    --output-root-base "/path/to/my_benchmarks" \
    --auto-tag \
    -- \
    # Command that runs inside the SLURM job
    python -m hyperlax.cli run-benchmark \
        --benchmark-name "PPO_vs_SAC_Comparison" \
        --algos "ppo_mlp" "sac_mlp" \
        --envs "gymnax.cartpole" "gymnax.pendulum" \
        --sweep-modes "batched" \
        --run-length-modifier "long" \
        --num_samples_per_run 8 \
        --hparam-batch-size 4 \
        --output-root "/path/to/my_benchmarks" \
        --log-level "INFO"
````

> **Tip:** Before launching a large-scale experiment, run a quick test by setting
> `--run-length-modifier "quick"` to check if the benchmark pipeline runs successfully at a minimal scale.


### 1.2 What Happens (Output Structure)

1. **Main Benchmark Directory**
   A timestamped directory is created for the entire benchmark suite.

```text
/path/to/my_benchmarks/PPO_vs_SAC_Comparison_20250802-103000/
```

2. **Sub-Experiment Directories**
   A separate subdirectory is created for each unique combination of algorithm, environment, and sweep mode.
   Naming convention: `{algo}_{env}_{mode}_N{samples}_{run_length}`.

```text
/path/to/my_benchmarks/PPO_vs_SAC_Comparison_20250802-103000/
├── ppo_mlp_gymnax_cartpole_batched_N8_long/
│   ├── samples/
│   │   ├── batch_00000/      # if sweep-modes "sequential" selected, prefix of the dir would be run_<>/
│   │   │   ├── success.txt   # Marks batch as complete
│   │   │   └── ...
│   │   └── batch_00001/      # 
│   │       ├── success.txt
│   │       └── ...
│   ├── success.txt           # Marks entire sub-experiment as complete
│   └── ...
├── ppo_mlp_gymnax_pendulum_batched_N8_long/
│   └── ...
├── sac_mlp_gymnax_cartpole_batched_N8_long/
│   └── ...
└── sac_mlp_gymnax_pendulum_batched_N8_long/
    └── ...
```

--hparam-batch-size will be applied if the selected hyperparams are vectorizable/batchable. 
If not, it will only try to find the common hyperparam combinations to create a batch with shared shape size.
In that case, if there is not enough samples with the shared size, batch slices will be smaller than the given argument resulting more batch slices (e.g., batch_<> dirs)


## 2. Plotting the Results

After the benchmark run is complete, `plot-benchmark` aggregates all results and generates comparative plots along with an interactive HTML dashboard.

### 2.1 How to Plot the Results

This command is typically run locally on a machine with a graphical environment.
It takes the main benchmark directory from Step 1 as input.

#### Example Command

```bash
python -m hyperlax.cli plot-benchmark \
    --results-dir-to-plot "/path/to/my_benchmarks/PPO_vs_SAC_Comparison_20250802-103000" \
    --plot-mode "basic" \
    --log-level "INFO"
```

* **`--results-dir-to-plot`**: Path to the main benchmark directory.
* **`--plot-mode`**: `basic` for a quick overview, `full` for a comprehensive set of plots.

  > **Note:** `"full"` mode is under development and may not work properly.


### 2.2 What Happens (Plotting Output)

1. **Plots Directory**
   A new directory for plots is created *next to* the results directory.

```text
/path/to/my_benchmarks/PPO_vs_SAC_Comparison_20250802-103000_plots/
```

2. **Interactive Dashboard**
   The main output is `index.html`, an interactive dashboard for viewing plots and descriptions.

3. **Plot Subdirectories**
   Plots are organized into subdirectories by category (e.g., `01_summary`, `02_learning_curves`).


**Final Takeaway:**
This two-step workflow lets you systematically run large-scale comparisons and produce a visualization suite with two commands.

