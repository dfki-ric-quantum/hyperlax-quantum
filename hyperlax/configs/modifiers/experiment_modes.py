import copy
import dataclasses

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.runner.launcher_utils import create_single_hp_batch_from_config_defaults


def apply_single_run_mode(
    config: BaseExperimentConfig,
) -> BaseExperimentConfig:
    """
    Configures an experiment for a single run using its default Tunable values.

    This function makes a single run compatible with the unified `run_experiment`
    function, which is designed to handle batches of hyperparameters. It achieves
    this by:
    1. Extracting the default `.value` from every Tunable field in the config.
    2. Packaging these values into a "batch of size 1".
    3. Injecting this batch into `config.training.hyperparam_batch_samples`.
    4. Setting `hyperparam_batch_enabled=True`.

    This allows the same core training logic to be used for both single runs
    and large-scale hyperparameter sweeps.
    """
    mod_config = copy.deepcopy(config)
    mod_config.experiment_mode = "single"

    # Create a batch of size 1 from the config's default values.
    hp_batch_from_defaults = create_single_hp_batch_from_config_defaults(mod_config)

    mod_config.training = dataclasses.replace(
        mod_config.training,
        hyperparam_batch_enabled=True,  # The runner expects this to be True.
        hyperparam_batch_size=1,
        hyperparam_batch_samples=hp_batch_from_defaults,
        hyperparam_batch_sample_ids=[0],  # Default sample_id for this single run.
    )
    return mod_config


# def apply_batched_sampling_mode(
#     config: BaseExperimentConfig,
# ) -> BaseExperimentConfig:
#     """
#     Configures an experiment for a batched hyperparameter sweep.

#     This is a lightweight modifier. Its main purpose is to set the experiment
#     mode and ensure the hyperparam_batch_enabled flag is set.

#     The actual generation of hyperparameter samples is NOT done here. It is handled
#     by the experiment launcher (e.g., `launch_sampling_sweep`), which uses the
#     `.distribution` attributes of the Tunable fields in the config to create
#     the samples before calling `run_experiment`.
#     """
#     mod_config = copy.deepcopy(config)
#     mod_config.experiment_mode = "batched"
#     mod_config.training = dataclasses.replace(
#         mod_config.training,
#         hyperparam_batch_enabled=True,
#     )
#     return mod_config
