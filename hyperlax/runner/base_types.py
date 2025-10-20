from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import chex
from typing_extensions import NamedTuple

from hyperlax.trainer.phaser import AlgoSetupFns


class RunResult(NamedTuple):
    hyperparams: dict[str, Any]
    success: bool


@dataclass(frozen=True)
class AlgoSpecificExperimentConfigContainer:
    algo_name: str
    algo_main_fn: Callable[[Any], Any]  # Function like main() or run_experiment()
    experiment_config: Any  # The full experiment config dataclass (e.g. PPOExperimentConfig)
    hyperparam_dist_config: dict[str, Any]
    hyperparams_container_spec: Any


@dataclass(frozen=True)
class AlgorithmInterface:
    """Encapsulates all algorithm-specific functions and classes for the experiment runner."""

    # Data structure for a single vectorized HP struct. e.g., DQNVectorizedHyperparams
    vectorized_hyperparams_cls: type[Any]
    non_vectorized_hyperparams_cls: type[Any]
    # Factory that returns the setup/re-setup functions for the phaser.
    algo_setup_fns_factory: Callable[[], AlgoSetupFns]
    # Function to generate algorithm-specific JAX keys.
    key_setup_fn: Callable[[chex.PRNGKey], tuple[chex.PRNGKey, ...]]
    # Callback to create the final =act_fn= for the evaluator.
    get_eval_act_fn_callback_for_algo: Callable[..., Any]  # You can make this more specific
    # A simple string prefix for logging.
    algorithm_name_prefix: str
    # Builder functions exposed for testing purposes
    build_network_setup_fn: Callable
    build_network_fn: Callable
    build_optimizer_fn: Callable
    build_update_step_fn: Callable
    build_distributed_layout_fn: Callable
