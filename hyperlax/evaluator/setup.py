import logging
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment

from hyperlax.base_types import ActFn, AnakinTrainOutput, EvalFn, EvaluationMetrics
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.evaluator.core import get_ff_evaluator_fn_w_normalizer
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.layout.data import distribute_keys_across_axes_wo_update_batch_dim
from hyperlax.layout.ops import transform_function_by_strategy
from hyperlax.normalizer.running_stats import (
    InitNormFn,
    NormalizeFn,
    NormParams,
    UpdateNormFn,
)

logger = logging.getLogger(__name__)

# Callback to get the algorithm-specific action selection function
GetEvalActFnCallbackT = Callable[
    [
        BaseExperimentConfig,  # config
        Callable[[FrozenDict, chex.Array], Any],  # model_apply_fn_for_eval
    ],
    ActFn,  # Returns: (params, observation, key) -> action
]

# Callback to extract relevant parameters for evaluation from the full learner state
ExtractParamsForEvalFnT = Callable[
    [
        Any,  # learner_state (full, train_strategy layout)
        DistributionStrategy,  # train_strategy
        DistributionStrategy,  # eval_strategy
    ],
    FrozenDict,  # Distributed model parameters for evaluation (eval_strategy layout)
]

# Callback to extract normalization parameters for evaluation
ExtractNormParamsForEvalFnT = Callable[
    [
        Any,  # learner_state (full, train_strategy layout)
        DistributionStrategy,  # train_strategy
        DistributionStrategy,  # eval_strategy
    ],
    NormParams,  # Distributed normalization parameters for evaluation (eval_strategy layout)
]


def setup_distributed_evaluator(
    eval_env: Environment,
    eval_key_base: chex.PRNGKey,
    get_eval_act_fn_callback: GetEvalActFnCallbackT,  # Use the Callable type hint
    model_apply_fn_for_eval: Callable[[FrozenDict, chex.Array], Any],
    normalizer_fns: tuple[InitNormFn, UpdateNormFn, NormalizeFn],
    config: BaseExperimentConfig,
    eval_strategy: DistributionStrategy,
    train_strategy: DistributionStrategy,
    extract_params_for_eval_fn: ExtractParamsForEvalFnT,
    extract_norm_params_for_eval_fn: ExtractNormParamsForEvalFnT,
    log_solve_rate_key: str | None = "solve_episode",
) -> EvalFn:
    eval_setup_logger = logging.getLogger(f"{logger.name}.EVAL_SETUP")

    core_act_fn = get_eval_act_fn_callback(config, model_apply_fn_for_eval)
    eval_setup_logger.debug(f"Core action function for evaluation: {core_act_fn}")

    core_eval_one_config_fn = get_ff_evaluator_fn_w_normalizer(
        env=eval_env,
        act_fn=core_act_fn,
        normalizer_fns=normalizer_fns,
        config=config,
        log_win_rate=True if log_solve_rate_key else False,
        eval_multiplier=1,
    )
    eval_setup_logger.debug(f"Core evaluator function (for one config): {core_eval_one_config_fn}")

    distributed_eval_keys, _ = distribute_keys_across_axes_wo_update_batch_dim(
        eval_key_base, eval_strategy
    )
    eval_setup_logger.debug(
        f"Distributed evaluation keys shape: {jax.tree_util.tree_map(lambda x: x.shape, distributed_eval_keys)}"
    )

    transformed_eval_fn = transform_function_by_strategy(
        core_eval_one_config_fn, eval_strategy, config.training.jit_enabled
    )
    eval_setup_logger.debug("Transformed (distributed) evaluation function created.")

    def final_distributed_eval_fn(
        learner_state_full_train_layout: Any,
    ) -> EvaluationMetrics:
        # train_strategy should be attached to learner_state by the phaser/caller.
        # train_strategy_from_state = getattr(learner_state_full_train_layout, 'train_strategy', None)
        # if train_strategy_from_state is None:
        #    eval_setup_logger.error("Learner state passed to eval_fn is missing 'train_strategy' attribute! Assuming eval_strategy.")
        #    train_strategy_from_state = eval_strategy # Risky fallback

        model_params_for_eval_dist = extract_params_for_eval_fn(
            learner_state_full_train_layout, train_strategy, eval_strategy
        )
        norm_params_for_eval_dist = extract_norm_params_for_eval_fn(
            learner_state_full_train_layout, train_strategy, eval_strategy
        )

        # eval_setup_logger.debug(f"Extracted model_params_for_eval_dist shapes: {get_pytree_shapes(model_params_for_eval_dist)}")
        # eval_setup_logger.debug(f"Extracted norm_params_for_eval_dist shapes: {get_pytree_shapes(norm_params_for_eval_dist)}")

        eval_output_anakin: AnakinTrainOutput = transformed_eval_fn(
            model_params_for_eval_dist, distributed_eval_keys, norm_params_for_eval_dist
        )

        return EvaluationMetrics(
            episode_metrics=eval_output_anakin.episode_metrics,
            other_metrics=eval_output_anakin.train_metrics,
        )

    eval_setup_logger.info("Distributed evaluation function setup complete.")
    return final_distributed_eval_fn


def slice_extra_batch_dims(
    data_tree: Any,  # Pytree of data shaped by train_strategy
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> Any:
    """
    Slices out dimensions present in train_strategy but not in eval_strategy.
    Currently, this primarily targets the 'update_batch' dimension.
    Assumes that eval_strategy is a subset of train_strategy in terms of named axes,
    or has fewer axes.
    """
    slicing_logger = logging.getLogger(f"{logger.name}.SLICE_EVAL_DIMS")
    # Identify axes in train_strategy not present by name in eval_strategy
    train_axes_names = {axis.name for axis in train_strategy.axes}
    eval_axes_names = {axis.name for axis in eval_strategy.axes}
    axes_to_potentially_slice_names = list(train_axes_names - eval_axes_names)

    # We primarily care about slicing out 'update_batch' if it's the difference.
    axis_to_slice_out_pos = -1
    if "update_batch" in axes_to_potentially_slice_names:
        try:
            axis_to_slice_out_pos = train_strategy.get_axis_position("update_batch")
            slicing_logger.debug(
                f"Identified 'update_batch' at original training position {axis_to_slice_out_pos} for potential slicing."
            )
        except ValueError:
            slicing_logger.debug(
                "'update_batch' in difference but not found in train_strategy. No slicing."
            )

    if axis_to_slice_out_pos == -1:
        slicing_logger.debug(
            "No 'update_batch' dimension to slice out based on strategy difference."
        )
        return data_tree  # No relevant dimension to slice

    def _slice_leaf(leaf: jnp.ndarray) -> jnp.ndarray:
        if leaf.ndim > axis_to_slice_out_pos:
            # Check if the dimension size is > 1, otherwise slicing index 0 on size 1 is fine but redundant
            if leaf.shape[axis_to_slice_out_pos] > 1:
                slicing_logger.debug(
                    f"Slicing leaf (shape {leaf.shape}) at axis {axis_to_slice_out_pos}"
                )
                idx = [slice(None)] * leaf.ndim
                idx[axis_to_slice_out_pos] = 0
                return leaf[tuple(idx)]
            else:  # Dimension size is 1, effectively already sliced
                slicing_logger.debug(
                    f"Leaf (shape {leaf.shape}) axis {axis_to_slice_out_pos} already size 1. No slice needed."
                )
                # We still need to remove the dimension to match eval_strategy's rank expectations
                return jnp.squeeze(leaf, axis=axis_to_slice_out_pos)
        return leaf

    return jax.tree_util.tree_map(_slice_leaf, data_tree)


def extract_common_norm_params(
    learner_state: Any,  # PPOOnPolicyLearnerState or DQNLearnerState
    train_strategy: DistributionStrategy,
    eval_strategy: DistributionStrategy,
) -> NormParams:
    norm_params_train_layout = learner_state.normalization_params
    return slice_extra_batch_dims(norm_params_train_layout, train_strategy, eval_strategy)
