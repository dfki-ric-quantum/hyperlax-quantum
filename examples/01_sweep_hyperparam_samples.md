# Hyperparameter Sweeps with `sweep-hp-samples`

`sweep-hp-samples` is designed for systematically exploring a predefined hyperparameter search space. It generates a fixed number of configurations using specific sampling strategies and executes them efficiently.


## 1. The Core Concept: Sample, Group, and Execute

The workflow follows three main stages:

1. **Sample**: Generates `N` hyperparameter configurations from the distributions defined in your algorithm's recipe file. Supports two main methods:
   - **`qmc_sobol` (Default):** Uses a Sobol sequence (Quasi-Monte Carlo) for more uniform coverage than random sampling — especially effective in high-dimensional spaces.
   - **`random`:** Standard random sampling.
2. **Group & Slice**: Groups the `N` samples based on properties, then slices them into smaller executable batches for the GPU.
3. **Execute**: Runs the batches sequentially or in parallel depending on the chosen execution mode.


## 2. Running a Hyperparameter Sweep

You launch a sweep using the `sweep-hp-samples` subcommand. Execution behavior is determined by a few key flags.


### 2.1 Mode 1: Batched Sweep (Default)

Runs multiple hyperparameter configurations in parallel within a single JAX-jitted function, maximizing GPU utilization.

```bash
python -m hyperlax.cli sweep-hp-samples \
    --algo-and-network-config "ppo_mlp" \
    --env-config "gymnax.pendulum" \
    --num-samples 64 \
    --hparam-batch-size 16 \
    --sampling-method "qmc_sobol" \
    --output-root "/path/to/my_sweeps" \
    --log-level "INFO"
````

#### Key Arguments Explained

* `--num-samples`: Total number of hyperparameter configurations to generate and run.
* `--hparam-batch-size`: Maximum configurations per GPU batch. The runner will split 64 samples into `64 / 16 = 4` batches.


### 2.2 Mode 2: Sequential Sweep

Runs each configuration one by one. This can be faster if hyperparameter differences are large (e.g., one agent uses `rollout_length=1024` and another `rollout_length=64`).

```bash
python -m hyperlax.cli sweep-hp-samples \
    --algo-and-network-config "ppo_mlp" \
    --env-config "gymnax.pendulum" \
    --num-samples 64 \
    --sequential "True" \
    --output-root "/path/to/my_sweeps" \
    --log-level "INFO"
```

> **Key Change:** Added the `--sequential` flag.


### 2.3 Mode 3: Grouped Batched Sweep (Smart Batching)

Groups samples by structural hyperparameters (e.g., number of layers, layer widths) before batching. This avoids repeated JIT compilation when architectures differ.

```bash
python -m hyperlax.cli sweep-hp-samples \
    --algo-and-network-config "ppo_mlp" \
    --env-config "gymnax.pendulum" \
    --num-samples 64 \
    --hparam-batch-size 16 \
    --group-by-structural-hparams "True" \
    --output-root "/path/to/my_sweeps" \
    --log-level "INFO"
```

> **Key Change:** Added the `--group-by-structural-hparams` flag.


## 3. Output Directory Structure

### Sequential Sweeps

Each run gets its own subdirectory.

```text
/path/to/my_sweeps/ppo_mlp/gymnax_pendulum/samples/
├── run_0000/
│   ├── success.txt
│   └── ...
├── run_0001/
│   └── ...
└── run_0063/
```

### Batched Sweeps

Each vectorized batch has its own subdirectory.

```text
/path/to/my_sweeps/ppo_mlp/gymnax_pendulum/samples/
├── batch_00000/  # Contains samples 0–15
│   ├── success.txt
│   └── ...
├── batch_00001/  # Contains samples 16–31
│   └── ...
└── batch_00003/
```


## 4. Resuming an Interrupted Sweep

Sweeps are resumable with the `--resume-from` flag.

```bash
python -m hyperlax.cli sweep-hp-samples \
    --algo-and-network-config "ppo_mlp" \
    --env-config "gymnax.pendulum" \
    --num-samples 64 \
    --sequential \
    --resume-from "/path/to/my_sweeps/ppo_mlp/gymnax_pendulum/" \
    --log-level "INFO"
```

### How It Works

* **Sequential sweep:** Checks each `run_XXXX` folder for `success.txt`.
* **Batched sweep:** Checks each `batch_XXXXX` folder.
* Any completed runs/batches are skipped.


## 5. Advanced Customization: Defining the Search Space

The hyperparameter search space is defined in `get_base_hyperparam_distributions()` inside the algorithm's recipe file.

For example, to modify `ppo_mlp`:

### Before

```python
# hyperlax/configs/algo/ppo_mlp.py
def get_base_hyperparam_distributions() -> dict[str, Any]:
    return {
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-3)),
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(
            values=[[32, 32], [64, 64], [128, 128]]
        ),
        # ... other distributions
    }
```

### After (Custom Search Space)

```python
# hyperlax/configs/algo/ppo_mlp.py
def get_base_hyperparam_distributions() -> dict[str, Any]:
    return {
        # Widen the learning rate range
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-6, 1e-2)),

        # Add a deeper architecture
        "algorithm.network.actor_network.pre_torso.layer_sizes": Categorical(
            values=[[64, 64], [128, 128], [256, 256, 256]]
        ),

        # Restrict activation function search
        "algorithm.network.actor_network.pre_torso.activation": Categorical(
            values=["silu", "gelu"]
        ),
        # ... other distributions
    }
```

> **Tip:** Create new recipe files (e.g., `ppo_mlp_wide_search.py`) with modified distributions to fully control the search space.


**Final Takeaway:**
`sweep-hp-samples` offers flexible sampling, batching, and grouping strategies, letting you systematically explore complex hyperparameter spaces while optimizing GPU efficiency.
