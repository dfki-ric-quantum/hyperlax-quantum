# Advanced Guide: Customizing Benchmarks and Experiments

While hyperlax provides pre-built recipes for common algorithms and networks, its real power lies in its modular and extensible configuration system. This guide will walk you through customizing every aspect of your experiment — from creating new network architectures to modifying core training logic and adding new environments.

## 1. The Core Concept: How hyperlax Finds Your Configurations
    
hyperlax uses a **"convention over configuration"** approach.  
The string names you provide in the CLI (e.g., `--algo-and-network-config sac_mlp`) directly map to Python files in the `hyperlax/configs/` directory. Understanding this mapping is key to customization.

- **Algorithm Recipes (`--algos`):** The string `ppo_mlp` maps to the file `hyperlax/configs/algo/ppo_mlp.py`. This file is the "recipe" that defines the algorithm's components (network, hyperparameters) and its tunable distributions.

- **Environments (`--envs`):** The string `gymnax.pendulum` maps to `hyperlax/configs/env/gymnax/pendulum.py`. The structure is `{framework}.{environment_name}`.

**Key Takeaway:** To create a new algorithm or environment configuration, you simply create a new Python file in the corresponding directory following this convention.

## 2. Scenario: Comparing a Quantum vs. Classical Network for SAC

Let's imagine our goal is to run a benchmark comparing the standard `sac_mlp` against a new version of SAC that uses a **Data Re-uploading Parametrized Quantum Circuit (DRPQC)** as its network torso.

### 2.1. Create a New Algorithm Recipe (`sac_drpqc`)

The easiest way to do this is to copy an existing recipe and modify it.

1. Copy `hyperlax/configs/algo/sac_mlp.py` to `hyperlax/configs/algo/sac_drpqc.py`.
2. Open the new `sac_drpqc.py` and make the following changes.

**Update the Network Configuration:**  
In the `get_base_config()` function, change the `network` component to point to the DRPQC network configuration instead of the MLP one.

`hyperlax/configs/algo/sac_drpqc.py`
```python
# ... (imports)
from hyperlax.configs.network.actorcritic_sac_drpqc import SACDRPQCActorCriticConfig  # NEW IMPORT

def get_base_config() -> BaseExperimentConfig:
    # Change the network from SACMLPActorCriticConfig to our new quantum one
    algorithm_component = SACConfig(
        network=SACDRPQCActorCriticConfig(),  # MODIFIED LINE
        hyperparam=SACHyperparams()
    )
    # Update config names for clarity
    config = BaseExperimentConfig(
        algorithm=algorithm_component,
        # ... (rest of the config)
        logger=LoggerConfig(base_exp_path="results_base/sac_drpqc_base"),  # MODIFIED
        config_name="sac_drpqc_base",  # MODIFIED
    )
    return config
````

**Update the Hyperparameter Distributions:**
The DRPQC network has different tunable parameters than the MLP (e.g., `n_layers`, `n_vstack` instead of `layer_sizes`).
Update the `get_base_hyperparam_distributions()` function accordingly.

`hyperlax/configs/algo/sac_drpqc.py`

```python
def get_base_hyperparam_distributions() -> dict[str, Any]:
    return {
        # ... (keep the algorithm HPs like actor_lr, gamma, etc.)
        "algorithm.hyperparam.actor_lr": LogUniform(domain=(1e-5, 1e-3)),
        # ...

        # REMOVE the old MLP-specific distributions
        # "algorithm.network.actor_network.pre_torso.layer_sizes": ...,

        # ADD the new DRPQC-specific distributions
        "algorithm.network.actor_network.pre_torso.n_layers": Categorical(values=[5, 10, 15]),
        "algorithm.network.actor_network.pre_torso.n_vstack": Categorical(values=[1, 2, 3]),
        "algorithm.network.critic_network.pre_torso.n_layers": Categorical(values=[5, 10, 15]),
        "algorithm.network.critic_network.pre_torso.n_vstack": Categorical(values=[1, 2, 3]),
    }
```

### 2.2. Create a New Benchmark Configuration File

Now, create a new Python file in `hyperlax/configs/benchmark/` to define your custom benchmark. Let's call it `my_quantum_vs_classical.py`.

`hyperlax/configs/benchmark/my_quantum_vs_classical.py`

```python
from pathlib import Path
from hyperlax.runner.launch_args import BenchmarkRunConfig

def get_benchmark_config() -> BenchmarkRunConfig:
    return BenchmarkRunConfig(
        benchmark_name="SAC_Quantum_vs_Classical_Pendulum",
        algos=["sac_mlp", "sac_drpqc"],  # Compare our new recipe with the old one
        envs=["gymnax.pendulum"],
        sweep_modes=["batched"],
        num_samples_per_run=16,
        hparam_batch_size=2,
        run_length_modifier="long",
        output_root=Path("results_benchmarks/quantum_comparison"),
        log_level="INFO",
    )
```

### 2.3. Run Your Custom Benchmark

Finally, launch the benchmark using the name of your new benchmark config file.

```bash
python3 slurm/launcher.py \
    --job-name "q_vs_c_sac" \
    --time "3-00:00:00" \
    --gres "gpu:1" \
    --singularity-image "/path/to/hyperlax.sif" \
    --output-root-base "/path/to/my_benchmarks" \
    -- \
    python -m hyperlax.cli run-benchmark \
        --base-config my_quantum_vs_classical  # Use the name of your new file
```

**Key Takeaway:** You can create entirely new algorithm-network combinations by creating a new recipe file that composes existing or new sub-configurations and defines the associated hyperparameter search space.

## 3. Advanced Customizations

### How to Modify Central Training Parameters

Parameters that control the training duration and frequency of evaluations (like `total_timesteps`, `num_evaluation`) are managed by **"run length modifiers"**. You can edit these modifiers or create new ones in:

`hyperlax/configs/modifiers/common_settings.py`

For example, to change what `--run-length-modifier "quick"` does, you would edit the `apply_quick_test_settings` function. This allows you to define standardized training protocols for your entire experiment.

### How to Add a Completely New Hyperparameter to an Algorithm

Suppose you want to add a new hyperparameter, `use_fancy_feature`, to PPO. This requires changes in three places:

1. **The Hyperparameter Struct** — Add the new field to the NamedTuple that defines the vectorized hyperparameters.

`hyperlax/algo/ppo/struct_ppo.py`

```python
class PPOVectorizedHyperparams(NamedTuple):
    # ... (existing fields)
    use_fancy_feature: chex.Array  # NEW
    sample_id: chex.Array
```

2. **The Hyperparameter Config** — Add a corresponding `Tunable` field to the PPO hyperparameter configuration class.

`hyperlax/algo/ppo/hyperparam.py`

```python
@dataclass(frozen=True)
class PPOHyperparams:
    # ... (existing fields)
    use_fancy_feature: Tunable = field(
        default_factory=lambda: Tunable(
            value=False,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
```

3. **The Core Logic** — Use the new hyperparameter in the algorithm's core logic file.

`hyperlax/algo/ppo/core.py`

```python
def _apply_single_update_cycle(learner_state_unit, _):
    # ...
    # Access the hyperparameter via the learner state
    hyperparams = learner_state_unit.algo_hyperparams
    use_fancy_feature = hyperparams.use_fancy_feature

    # Use a jax.lax.cond to conditionally apply your new feature
    loss = jax.lax.cond(
        use_fancy_feature,
        lambda: compute_loss_with_fancy_feature(...),
        lambda: compute_standard_loss(...),
    )
    # ...
```

**Note:** The core update logic function must be fully JAX-compatible (no dynamic shapes). Current strategy is to use masking (see `hyperlax/algo/ppo/core.py`).

### Performance Considerations

* Introducing new hyperparameters may cause **shape mismatches** between agents.
* If differences are large (e.g., rollout length = 1024 vs. 64), padding overhead may outweigh vectorization gains.
* In such cases, consider sequential or asynchronous updates, while still parallelizing environment simulations.
* The `--group_by_structural_hparams True` flag groups agents by common shapes before running experiments.

**Limitation:** When sampling heterogeneous architectures, grouping may create smaller batches than your desired `--hparam-batch-size`.

