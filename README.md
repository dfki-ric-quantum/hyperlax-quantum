# Benchmarking Classical and Quantum Reinforcement Learning Algorithms with _hyperlax_

_hyperlax_ is a unified JAX-based framework for high-throughput reinforcement learning, designed to benchmark and optimize both **classical and quantum machine learning models**. It accelerates research by enabling massively parallel hyperparameter execution, transforming the traditional "one experiment, one process" paradigm into a vectorized "many experiments, one process" workflow.

By leveraging `jax.vmap` and `jax.pmap` across hyperparameter configurations, hyperlax allows for direct, fair, and efficient performance comparisons between different model families (e.g., MLP vs. PQC) on the same hardware, speeding up the research cycle.

## Installation


### Local

We developed and tested with Python 3.10.

``` shell
conda create --name hyperlax python=3.10 # or python3.10 -m venv .venv
conda activate hyperlax                  # or source .venv/bin/activate
```

Install the JAX version we develop and test according to your hardware.
For CUDA:
``` shell
pip install -U "jax[cuda12_pip]==0.4.28" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> NOTE: if you are getting "Segmentation Fault" with GPU setup, we recommend installing JAX with CUDA >=12.6 wheel. (See [JAX docs](https://docs.jax.dev/en/latest/installation.html#installation)).

For CPU-only:
``` shell
pip install -U "jax[cpu]==0.4.28"
```

Install hyperlax and its dependencies:
``` shell
git clone git@github.com:dfki-ric-quantum/hyperlax-quantum.git
cd hyperlax-quantum
pip install -e '.' # or `'.[dev]'
```

### Container (Optional)

This setup is primarily for running large-scale experiments on a high-performance computing (HPC) cluster.

``` shell
cd containers
singularity build hyperlax.sif singularity.def
```

Note on Singularity Installation: Official installation guides may list outdated dependencies for modern Linux distributions (e.g., Ubuntu 22.04+). If you encounter issues (e.g., regarding =fuse=), we recommend consulting the [SingularityCE GitHub Releases](https://github.com/sylabs/singularity/releases) page for the latest packages and platform-specific instructions.

If you want to have pre-built singularity image used to develop and test, you can download from [zenodo.org/records/17426400](https://zenodo.org/records/17426400).

## Quick Start

Run a quick, small-scale hyperparameter sweep for a classical PPO agent with an MLP policy on the Pendulum environment. This command will train 4 different hyperparameter configurations in a single vectorized run.

``` bash
conda activate hyperlax
source envs/hyperlax_setup.sh

python hyperlax/cli.py sweep-hp-samples \
    --algo-and-network-config ppo_mlp_no_network_search \
    --env-config gymnax.pendulum \
    --run-length-modifier quick \
    --num-samples 4 \
    --hparam-batch-size 4 \
    --log-level INFO \
    --output-root ./results/quickstart_classical
```

After the run completes, you'll find all results, logs, and metrics in the `results/quickstart_classical/ppo_mlp/gymnax_pendulum` directory.

NOTE: Use --run-length-modifier `long` option for longer/realistic training session.

### Example: Running a Quantum RL Experiment

Run a single experiment using a **Parametrized Quantum Circuit (PQC)** as the policy network for the SAC algorithm.

``` bash
python hyperlax/cli.py run-single-hp \
    --algo-and-network-config sac_drpqc \
    --env-config gymnax.pendulum \
    --run-length-modifier quick \
    --log-level INFO \
    --output-root ./results/quickstart_quantum
```


## Command-Line Interface (CLI)

The main entry point is `hyperlax/cli.py`.

### `run-single-hp`: Run a single experiment  
Runs one experiment using default hyperparameters.

```bash
python hyperlax/cli.py run-single-hp \
    --algo-and-network-config sac_mlp \
    --env-config gymnax.cartpole \
    --log-level INFO
```

### `sweep-hp-samples`: Run a hyperparameter sweep

Generates samples and runs them in batches.

```bash
python hyperlax/cli.py sweep-hp-samples \
    --algo-and-network-config ppo_mlp_no_network_search \
    --env-config gymnax.pendulum \
    --num-samples 64 \
    --hparam-batch-size 16 \
    --log-level INFO 
```

Use the `--sequential True` flag to run one-by-one for comparison.

### `optuna-hp-search`: Run an optimization study

Uses Optuna to go beyond random sampling for hyperparam search for finding the top performing hyperparameters.

```bash
python hyperlax/cli.py optuna-hp-search \
    --algo-and-network-config ppo_mlp \
    --env-config gymnax.pendulum \
    --n-trials 100
```

### `run-benchmark`: Run a predefined suite of experiments

Executes a benchmark defined in `hyperlax/configs/benchmark/`, comparing multiple algorithms, environments, and sweep modes.

```bash
python hyperlax/cli.py run-benchmark \
    --algos "ppo_mlp" "ppo_drpqc" \
    --envs "gymnax.pendulum" "gymnax.cartpole" \
    --num-samples-per-run 16
```

Or use the pre-defined config:
```bash
python hyperlax/cli.py run-benchmark --base-config ppo_mlp_vs_drpqc
```

### `plot-benchmark`: Analyze and plot benchmark results

Post-processes the output of a benchmark run to generate summary plots.

```bash
python hyperlax/cli.py plot-benchmark --results-dir-to-plot ./results_bench
```

### `generate-hp-samples`: Create a hyperparameter file

Generates samples and saves them to a CSV file without running experiments.

```bash
python hyperlax/cli.py generate-hp-samples \
    --algo-and-network-config ppo_mlp \
    --num-samples 16 \
    --output-file ./ppo_mlp_samples.csv
```

## Benchmark Results

![Benchmark Results](benchmark_results/benchmark_best_found_return_distribution_combined.png)

[hyperlax/configs/algo/benchmarked/01_static_qmc_sampling_S64](hyperlax/configs/algo/benchmarked/01_static_qmc_sampling_S64) is the directory containing the algorithm configurations used for benchmarking. We sample 64 hyperparameter sets using the QMC method (no search). The benchmark data are stored under [benchmark_results](benchmark_results/).

The hyperparameter distributions are chosen so that algorithms sharing the same parameters also share the same distributions. This provides an unbiased setup (as much as possible) and allows us to assess hyperparameter sensitivity.

We also ran experiments involving hyperparameter search (to be released soon).

Feel free to try and beat the current results!

## Reproducing Benchmark Results

Note that the full benchmark run takes about one month on a single GPU (e.g., A100).
The slowest configuration is ppo-drpqc on reacher, which alone takes around 8 days, whereas the classical models finish in just a few hours. Therefore, you’ll need to strategize how to distribute the quantum models (especially ppo variants) across your cluster setup to optimize runtime.

``` bash
python hyperlax/cli.py run-benchmark \
  --algos benchmarked.01_static_qmc_sampling_S64.{dqn,ppo,sac}_{mlp,tmlp,drpqc} \
  --envs gymnax.{cartpole,pendulum,reacher} brax.inverted_double_pendulum \
  --num-samples-per-run 64 \
  --sweep-modes "sequential" \ # HP configs include arch. choices such as n_layers and no batch support for those!
  --run-length-modifier long \
  --sampling-method "qmc_sobol" \
  --log-level "INFO" \
  --output-root "./results_benchmark_reproduce"
```

* Crazy Long Runs
  * `{ppo}_{drpqc}` on inverted dp and reacher (due to increased obs dim.)
* Very Long Runs
  * `{ppo}_{drpqc}` on cartpole and pendulum
* Long Runs
  * `{dqn,sac}_{drpqc}`
* Medium Runs
  * `{ppo,dqn,sac}_{tmlp}`
* Light Runs
  * `{ppo,dqn,sac}_{mlp}`

If you are not performing architecture searches or sampling, you can use
--sweep-modes "batched" to run hyperparameter batches in parallel.
All algorithms are vectorized for efficient single-GPU utilization, enabling multiple configurations to run simultaneously.

## Examples

See [hyperlax/examples](examples/) directory to find more usage examples.

## Extending hyperlax

Adding a new algorithm or network (classical or quantum) is:

1. **Create Configuration**: Define dataclasses for your algorithm's config, network architectures, and hyperparameters in `hyperlax/configs/`.
2. **Implement Core Logic**: Write the core algorithm update step with vectorized hyperparams (see how existing algorithm implementation achieves it) and loss functions in `hyperlax/algo/`. For custom networks, add the Flax module in `hyperlax/network/`.
3. **Implement the `AlgorithmInterface`**: Create a `setup_my_algo.py` file that provides the necessary functions (network builder, optimizer builder, etc.) and packages them into an `AlgorithmInterface` dataclass.
4. **Create a Recipe**: Add a recipe file like `hyperlax/configs/algo/my_algo.py` that provides `get_base_config` and `get_base_hyperparam_distributions` functions.
5. **Run it!**: Your new algorithm is now available via the CLI, e.g., `--algo-and-network-config my_algo`.

## Project Philosophy

- **Unified Benchmarking**: Provide a single, consistent platform to fairly evaluate and compare the performance and data efficiency of classical, quantum, and tensor network models for reinforcement learning.
- **Maximize Hardware Throughput**: Minimize wall-clock time for research by fully utilizing available hardware, especially multiple GPUs on a cluster setup.
- **Configuration as Code**: Experiment configurations, including hyperparameter search spaces, are version-controllable, readable, and strongly-typed Python code.
- **Vectorize Everything Possible**: We aggressively apply `vmap` not just to environments but to distinct model architectures and training hyperparameters.
- **Immutable and Functional**: We adhere to JAX's functional programming paradigm. State is explicitly passed and returned, and configurations are treated as immutable.

## Credits

_hyperlax_ extends ideas from prior JAX-based RL systems and quantum ML benchmarks:

- [**purejaxrl**](https://github.com/luchris429/purejaxrl) demonstrated fully JIT-compiled RL loops to keep environment rollouts on-device.
- [**Stoix**](https://github.com/EdanToledo/Stoix) introduced modular multi-device abstractions for distributed training.
- _hyperlax_ takes these principles further by vectorizing across **hyperparameter configurations**, enabling batched, parallel experimentation in a single compiled computation.
- [**qml-benchmarks**](https://github.com/XanaduAI/qml-benchmarks) provided our baseline quantum model (i.e., Data-Reuploading Parameterized Quantum Circuit).
- [**gymnax**](https://github.com/RobertTLange/gymnax) and [**brax**](https://github.com/google/brax) supplied fast, JAX-native environments crucial for large-scale, differentiable RL benchmarks.

Thanks to the broader open-source research community for advancing transparent, reproducible, and scalable machine learning tools.

## Tests

```bash
pytest tests/
```

## Limitations and Open Issues

- Vectorized/Batched hyperparameter computation is supported for algorithmic hyperparameters: both scalar (e.g., learning rates) and structural (e.g., rollout length) but not for function approximation related parameters (e.g., hidden dimensions). An experimental vectorized MLP implementation is available as a reference (see parametric_torso.py).
- In `dqn-drpqc_gymnax.acrobot`, 9 out of 64 Acrobot samples trigger JAX’s `XlaRuntimeError: INTERNAL: ptxas exited`, indicating a possible synchronization issue in the multi-GPU setup. This issue was not further investigated.
- `tmlp` models are highly sensitive to specific learning rates; causin gradient explosions leading.
- The current parameterized quantum circuit implementation, combined with the JAX-backed PennyLane version used, results in long JIT compilation times, likely due to non-JAX-compatible components in PennyLane. Interested researchers may explore the [PennyLaneAI/catalyst](https://github.com/PennyLaneAI/catalyst) project for potential JIT and execution improvements, though this has not been tested here.

## Contributing

See [CONTRIBUTING](./CONTRIBUTING.md) for details.

## Citation

```bibtex
@software{bolat_hyperlax_quantum_2025,
    author = {Bolat, Ugur},
    doi = {10.5281/zenodo.17426400},
    month = {10},
    title = {{Benchmarking Classical and Quantum Reinforcement Learning Algorithms with JAX}},
    url = {https://github.com/dfki-ric-quantum/hyperlax-quantum},
    version = {0.0.1},
    year = {2025}
}
```

## Releases

### Semantic Versioning

Semantic versioning must be used, that is, the major version number will be incremented when the API changes in a backwards incompatible way, the minor version will be incremented when new functionality is added in a backwards compatible manner, and the patch version is incremented for bugfixes, documentation, etc.

## License

Licensed under the BSD 3-clause license, see `LICENSE` for details.

## Acknowledgments

This work was funded by the German Ministry of Economic Affairs and Climate Action (BMWK) and the German Aerospace Center (DLR) in project QuBER-KI (grants: 50RA2207A, 50RA2207B).
