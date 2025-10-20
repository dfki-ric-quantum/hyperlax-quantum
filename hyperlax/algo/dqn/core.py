import logging
from collections.abc import Callable
from typing import Any

import chex
import distrax
import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep

from hyperlax.algo.dqn.loss import double_q_learning_loss, q_learning_loss
from hyperlax.algo.dqn.struct_dqn import (
    DQNLearnerState,
    DQNTransition,
    DQNVectorizedHyperparams,
    get_dqn_max_masked_dims,
)
from hyperlax.base_types import (
    ActorApply,
    AnakinTrainOutput,
    LogEnvState,
    OnlineAndTarget,
)
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.logger.jax_debug import enable_jax_debug_for_context
from hyperlax.normalizer.running_stats import (
    InitNormFn,
    NormalizeFn,
    NormParams,
    UpdateNormFn,
)
from hyperlax.rbuffer.masked_item_buffer import (
    MaskedItemBufferState,
    build_masked_item_buffer,
)
from hyperlax.utils.algo_setup import (
    NetworkAndOptimizerOnlyFns,
    NetworkAndOptimizerWithParamsTuple,
)

logger = logging.getLogger(__name__)

# JAX Debug Log Message Structure:
# TAG_PREFIX,SID,CURRENT_ENV_STEP,CONTEXT_TAG,METRIC_NAME_1,{{metric_val_1}},
# METRIC_NAME_2,{{metric_val_2}},...
JAX_LOG_PREFIX = "DQN_CORE"

# Context tags for different stages
CTX_WARMUP_ROLLOUT = "WARMUP_ROLLOUT"
CTX_WARMUP_STEP = "WARMUP_STEP"
CTX_ROLLOUT_STEP = "ROLLOUT_STEP"
CTX_BUFFER_SAMPLE = "BUFFER_SAMPLE"
CTX_LOSS_CALC = "LOSS_CALC"
CTX_OPTIM_STEP = "OPTIM_STEP"
CTX_EPOCH_LOOP = "EPOCH_LOOP"


def build_warmup_rollout_fn(
    env: Environment,
    nets_and_opts: NetworkAndOptimizerWithParamsTuple,
    normalizer_fns: tuple[InitNormFn, UpdateNormFn, NormalizeFn],
    hyperparam_batch_wrappers: dict[str, Any],
    config: BaseExperimentConfig,
) -> Callable[
    [
        LogEnvState,
        TimeStep,
        MaskedItemBufferState,
        chex.PRNGKey,
        NormParams,
        DQNVectorizedHyperparams,
    ],
    tuple[
        LogEnvState,
        TimeStep,
        chex.PRNGKey,
        MaskedItemBufferState,
        NormParams,
        chex.Array,
    ],
]:
    warmup_factory_logger = logging.getLogger(f"{logger.name}.WARMUP_FN_FACTORY")

    if config.logger.enabled and config.logger.enable_jax_debug_prints:
        # enable_jax_debug_for_context(JAX_LOG_PREFIX)
        enable_jax_debug_for_context("DQN_CORE")
        enable_jax_debug_for_context(CTX_WARMUP_ROLLOUT)

    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    max_dims = get_dqn_max_masked_dims(
        algo_hp_batch, config.training.num_devices, config.training.update_batch_size
    )

    buffer_fns = build_masked_item_buffer(
        max_parallel_envs=max_dims.num_envs,
        buffer_size_per_env=max_dims.buffer_size // max_dims.num_envs,
        effective_buffer_size_per_env=max_dims.buffer_size,
        sample_batch_size=max_dims.batch_size,
    )

    online_q_params_for_rollout = nets_and_opts.params["q_network"]
    q_network_apply_for_rollout = nets_and_opts.networks["q_network"].apply
    add_to_buffer_fn = buffer_fns.add
    max_warmup_rollout_length = max_dims.warmup_rollout_length
    max_num_envs = max_dims.num_envs

    warmup_factory_logger.debug(
        f"Max_warmup_rollout_length: {max_warmup_rollout_length}, max_num_envs_for_padding: {max_num_envs}"
    )

    _, update_normalization_stats_warmup, normalize_observation_warmup = normalizer_fns
    warmup_rollout_indices_range = jnp.arange(max_warmup_rollout_length)
    env_indices_range_warmup = jnp.arange(max_num_envs)

    def _execute_warmup_rollout(
        initial_env_state: LogEnvState,
        initial_timestep: TimeStep,
        initial_buffer_state: MaskedItemBufferState,
        base_key_warmup: chex.PRNGKey,
        initial_norm_params: NormParams,
        current_hyperparams: DQNVectorizedHyperparams,  # This is the per-instance HPs
    ) -> tuple[
        LogEnvState,
        TimeStep,
        chex.PRNGKey,
        MaskedItemBufferState,
        NormParams,
        chex.Array,
    ]:
        # SID and STEP for the whole warmup phase for this HP instance
        sid_warmup = current_hyperparams.sample_id  # Already scalar per lane
        # Warmup doesn't usually contribute to global step count in the same way,
        # but we can use a local step counter for its duration if needed or pass 0.
        # For consistency, let's assume env_steps_counter starts at 0 for warmup.
        env_steps_counter_warmup = jnp.zeros_like(
            current_hyperparams.critic_lr, dtype=jnp.float32
        )  # Scalar 0.0

        jax.debug.print(
            f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_WARMUP_ROLLOUT},start,warmup_len:{{wu_len}},num_envs:{{ne}}",
            sid=sid_warmup,
            step=env_steps_counter_warmup,
            wu_len=current_hyperparams.warmup_rollout_length,
            ne=current_hyperparams.total_num_envs,
        )

        def _step_environment_warmup(
            env_carry_state: tuple[
                LogEnvState, TimeStep, chex.PRNGKey, NormParams, chex.Array
            ],  # Last element is local step counter
            step_idx: int,
        ) -> tuple[
            tuple[LogEnvState, TimeStep, chex.PRNGKey, NormParams, chex.Array],
            DQNTransition,
        ]:
            env_state, timestep, key, norm_params, env_step_count = env_carry_state

            # SID is from current_hyperparams in outer scope
            rollout_active_mask = (step_idx < current_hyperparams.warmup_rollout_length).astype(
                jnp.float32
            )
            env_active_mask = (
                env_indices_range_warmup < current_hyperparams.total_num_envs
            ).astype(jnp.float32)
            step_mask = rollout_active_mask * env_active_mask

            raw_observation = timestep.observation.agent_view
            # Normalize observations if flag is true
            obs_for_norm_update = raw_observation
            updated_norm_params_temp = update_normalization_stats_warmup(
                norm_params, obs_for_norm_update, weights=step_mask
            )
            normalized_observation_if_true = normalize_observation_warmup(
                updated_norm_params_temp, raw_observation
            )

            normed_obs = jnp.where(
                current_hyperparams.normalize_observations > 0.5,
                normalized_observation_if_true,
                raw_observation,
            )
            updated_norm_params = updated_norm_params_temp  # Use the updated params

            step_mask_for_obs = rollout_active_mask * env_active_mask[:, jnp.newaxis]
            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_WARMUP_STEP},"
                + "wu_step_idx:{wu_idx},norm_obs_flag:{no_f},raw_obs_sum:{ros},normed_obs_sum:{nos}",
                sid=sid_warmup,
                step=env_step_count,
                wu_idx=step_idx,
                no_f=current_hyperparams.normalize_observations,
                # ros=jnp.sum(raw_observation * step_mask),
                # nos=jnp.sum(normed_obs * step_mask)
                ros=jnp.sum(raw_observation * step_mask_for_obs),
                nos=jnp.sum(normed_obs * step_mask_for_obs),
            )

            key, policy_generation_key = jax.random.split(key)
            # Q-network uses non-distributed initial params during warmup
            online_q_preferences = q_network_apply_for_rollout(
                online_q_params_for_rollout, normed_obs
            ).preferences
            epsilon_greedy_policy = distrax.EpsilonGreedy(
                preferences=online_q_preferences,
                epsilon=current_hyperparams.training_epsilon,
                dtype=int,
            )
            # FIX: keep pre-step timestep so obs = s_t
            prev_timestep = timestep

            action_taken = epsilon_greedy_policy.sample(seed=policy_generation_key)
            next_env_state, next_timestep = jax.vmap(env.step, in_axes=(0, 0))(
                env_state, action_taken
            )

            # NOTE we don't count steps in warmup!
            updated_env_step_count = env_step_count

            # Use pre-step observation (s_t), reward/done/next_obs from post-step (r_{t+1}, s_{t+1})
            transition = DQNTransition(
                obs=prev_timestep.observation,
                action=action_taken,
                reward=next_timestep.reward,
                done=next_timestep.last().reshape(-1),
                next_obs=next_timestep.extras["next_obs"],
                info={**next_timestep.extras["episode_metrics"], "valid": step_mask},
            )
            new_env_carry_state = (
                next_env_state,
                next_timestep,
                key,
                updated_norm_params,
                updated_env_step_count,
            )

            return new_env_carry_state, transition

        scan_initial_carry = (
            initial_env_state,
            initial_timestep,
            base_key_warmup,
            initial_norm_params,
            env_steps_counter_warmup,
        )
        (
            (
                final_env_state,
                final_timestep,
                final_key,
                final_norm_params,
                final_env_step_counts,
            ),
            collected_transitions,
        ) = jax.lax.scan(
            _step_environment_warmup, scan_initial_carry, warmup_rollout_indices_range
        )
        jax.debug.print(
            f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_WARMUP_ROLLOUT},"
            + "end,final_key:{fk},total_warmup_steps_taken:{fes}",
            sid=sid_warmup,
            step=final_env_step_counts,
            fk=final_key,
            fes=final_env_step_counts,
        )

        transposed_transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), collected_transitions
        )
        updated_buffer_state = add_to_buffer_fn(
            initial_buffer_state,
            transposed_transitions,
            transposed_transitions.info["valid"],
        )

        return (
            final_env_state,
            final_timestep,
            final_key,
            updated_buffer_state,
            final_norm_params,
            final_env_step_counts,
        )

    return _execute_warmup_rollout


# NOTE if/else remove but could be reverted since jit was complaining? or did you include it into hyperparams?
def _compute_q_loss_and_gradients(
    online_params: FrozenDict,
    target_params: FrozenDict,
    transitions: DQNTransition,
    hyperparams: DQNVectorizedHyperparams,
    loss_mask: chex.Array,
    q_apply_fn: ActorApply,
    use_double_q_learning: bool,
) -> tuple[jnp.ndarray, dict]:
    """Compute Q loss and gradients for DQN update."""
    # jax.debug.print(f"[{JAX_DEBUG_CTX_LOSS_GRAD}] hyperparams (gamma, huber_loss_parameter): {{g}}, {{hlp}}",
    #                g=hyperparams.gamma, hlp=hyperparams.huber_loss_parameter)
    # jax.debug.print(f"[{JAX_DEBUG_CTX_LOSS_GRAD}] transitions.reward[:5]: {{r}}, transitions.done[:5]: {{d}}, transitions.action[:5]: {{a}}",
    #                r=transitions.reward[:5], d=transitions.done[:5], a=transitions.action[:5])
    q_online_current_obs = q_apply_fn(online_params, transitions.obs).preferences
    q_target_next_obs = q_apply_fn(target_params, transitions.next_obs).preferences
    discount_factor = 1.0 - transitions.done.astype(jnp.float32)
    discounted_gamma = (discount_factor * hyperparams.gamma).astype(jnp.float32)
    clipped_reward = jnp.clip(
        transitions.reward, -hyperparams.max_abs_reward, hyperparams.max_abs_reward
    ).astype(jnp.float32)
    if use_double_q_learning:
        q_online_next_obs_selector = q_apply_fn(online_params, transitions.next_obs).preferences
        loss_sum, aux_info = double_q_learning_loss(
            q_online_current_obs,
            transitions.action,
            clipped_reward,
            discounted_gamma,
            q_target_next_obs,
            q_online_next_obs_selector,
            hyperparams.huber_loss_parameter,
            loss_mask,
        )
    else:
        loss_sum, aux_info = q_learning_loss(
            q_online_current_obs,
            transitions.action,
            clipped_reward,
            discounted_gamma,
            q_target_next_obs,
            hyperparams.huber_loss_parameter,
            loss_mask,
        )
    return loss_sum, aux_info


def build_dqn_update_step_fn(
    env: Environment,
    nets_and_opts: NetworkAndOptimizerOnlyFns | NetworkAndOptimizerWithParamsTuple,
    normalizer_fns: tuple[InitNormFn, UpdateNormFn, NormalizeFn],
    config: BaseExperimentConfig,
    train_strategy: DistributionStrategy,
    hyperparam_batch_wrappers: dict[str, Any],
    hyperparam_non_vectorizeds: Any,
    # buffer_fns,
):
    q_apply_fn = nets_and_opts.networks["q_network"].apply
    q_update_fn = nets_and_opts.optimizers["q_network"].update

    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    max_dims = get_dqn_max_masked_dims(
        algo_hp_batch, config.training.num_devices, config.training.update_batch_size
    )

    buffer_fns = build_masked_item_buffer(
        max_parallel_envs=max_dims.num_envs,
        buffer_size_per_env=max_dims.buffer_size // max_dims.num_envs,
        effective_buffer_size_per_env=max_dims.buffer_size,
        sample_batch_size=max_dims.batch_size,
    )

    pmean_axis_name_for_update_batch = (
        train_strategy.get_axis_spec("update_batch").axis_name
        if train_strategy.has_axis("update_batch")
        else None
    )
    pmean_axis_name_for_device = (
        train_strategy.get_axis_spec("device").axis_name
        if train_strategy.has_axis("device")
        and train_strategy.get_axis_spec("device").method == "pmap"
        else None
    )

    max_num_envs, max_rollout_length, max_epochs = (
        max_dims.num_envs,
        max_dims.rollout_length,
        max_dims.epochs,
    )
    target_env_steps_global = config.training.total_timesteps

    core_logic_logger = logging.getLogger(f"{logger.name}.DQN_CORE_UPDATE_LOGIC")

    if config.logger.enabled and config.logger.enable_jax_debug_prints:
        enable_jax_debug_for_context(JAX_LOG_PREFIX)
        enable_jax_debug_for_context(CTX_EPOCH_LOOP)
        enable_jax_debug_for_context(CTX_ROLLOUT_STEP)
        enable_jax_debug_for_context(CTX_LOSS_CALC)
        enable_jax_debug_for_context(CTX_OPTIM_STEP)
        enable_jax_debug_for_context(CTX_BUFFER_SAMPLE)

    core_logic_logger.debug(
        f"Max dims for scans: num_envs={max_num_envs}, rollout={max_rollout_length}, epochs={max_epochs}"
    )
    buffer_add_fn = buffer_fns.add
    buffer_sample_fn = buffer_fns.sample
    _, update_norm_stats_fn, normalize_fn = normalizer_fns
    rollout_indices = jnp.arange(max_rollout_length)
    vec_env_indices = jnp.arange(max_num_envs)
    epoch_indices = jnp.arange(max_epochs)

    def _apply_single_update_cycle(
        learner_state_unit: DQNLearnerState, _: Any
    ) -> tuple[DQNLearnerState, AnakinTrainOutput]:
        def _step_environment_rollout(
            env_carry_state: DQNLearnerState, step_idx: Any
        ) -> tuple[DQNLearnerState, DQNTransition]:
            (
                q_params,
                opt_states,
                buffer_state,
                key,
                env_state,
                timestep,
                hyperparams,
                norm_params,
                env_step_counter,
            ) = env_carry_state
            sid = hyperparams.sample_id

            masks = {
                "global_completion": (env_step_counter < target_env_steps_global).astype(
                    jnp.float32
                ),
                "rollout": (step_idx < hyperparams.rollout_length).astype(jnp.float32),
                "env": (vec_env_indices < hyperparams.total_num_envs).astype(jnp.float32),
            }
            step_mask = masks["global_completion"] * masks["rollout"] * masks["env"]
            raw_obs = timestep.observation.agent_view
            obs_for_norm_update = raw_obs
            updated_norm_params_temp = update_norm_stats_fn(
                norm_params, obs_for_norm_update, weights=step_mask
            )
            normalized_observation_if_true = normalize_fn(updated_norm_params_temp, raw_obs)

            normed_obs = jnp.where(
                hyperparams.normalize_observations > 0.5,
                normalized_observation_if_true,
                raw_obs,
            )
            updated_norm_params = updated_norm_params_temp

            step_mask_for_obs = step_mask[:, jnp.newaxis]
            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_ROLLOUT_STEP},"
                + "rollout_idx:{r_idx},norm_flag:{n_f},raw_obs_sum:{ros},normed_obs_sum:{nos}",
                sid=sid,
                step=env_step_counter,
                r_idx=step_idx,
                n_f=hyperparams.normalize_observations,
                # ros=jnp.sum(raw_obs*step_mask), nos=jnp.sum(normed_obs*step_mask)
                ros=jnp.sum(raw_obs * step_mask_for_obs),
                nos=jnp.sum(normed_obs * step_mask_for_obs),
            )

            key, policy_key = jax.random.split(key)
            actor_policy = q_apply_fn(q_params.online, normed_obs)
            actor_policy = distrax.EpsilonGreedy(
                preferences=actor_policy.preferences,
                epsilon=hyperparams.training_epsilon,
                dtype=int,
            )

            # FIX: keep pre-step timestep so obs = s_t
            prev_timestep = timestep

            action = actor_policy.sample(seed=policy_key)
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
            new_valid_steps = jnp.sum(step_mask).astype(jnp.float32)
            env_steps_counter_new = env_step_counter + new_valid_steps

            # Use pre-step observation (s_t); reward/done/next_obs from post-step (r_{t+1}, s_{t+1})
            transition = DQNTransition(
                obs=prev_timestep.observation,
                action=action,
                reward=timestep.reward,
                done=timestep.last().reshape(-1),
                next_obs=timestep.extras["next_obs"],
                info={**timestep.extras["episode_metrics"], "valid": step_mask},
            )
            new_env_carry_state = DQNLearnerState(
                q_params,
                opt_states,
                buffer_state,
                key,
                env_state,
                timestep,
                hyperparams,
                updated_norm_params,
                env_steps_counter_new,
            )

            return new_env_carry_state, transition

        learner_state_after_rollout, traj_batch = jax.lax.scan(
            _step_environment_rollout, learner_state_unit, rollout_indices
        )
        traj_batch_swapped = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_state_updated = buffer_add_fn(
            learner_state_after_rollout.buffer_state,
            traj_batch_swapped,
            traj_batch_swapped.info["valid"],
        )
        learner_state_for_training = learner_state_after_rollout._replace(
            buffer_state=buffer_state_updated
        )

        epoch_scan_init_state = (
            learner_state_for_training.params,
            learner_state_for_training.opt_states,
            learner_state_for_training.buffer_state,
            learner_state_for_training.key,
        )
        # Capture hyperparams and total_env_steps from the state before the epoch loop
        hyperparams_for_epochs = learner_state_for_training.algo_hyperparams
        # non_vec_hyperparams_for_epochs = learner_state_for_training.non_vec_algo_hyperparams

        total_env_steps_at_epoch_start = learner_state_for_training.total_env_steps_counter
        norm_params_at_epoch_start = (
            learner_state_for_training.normalization_params
        )  # For normalizing buffer samples
        sid_epoch_level = hyperparams_for_epochs.sample_id

        def _train_single_epoch(epoch_carry_state: tuple, epoch_idx: Any) -> tuple[tuple, dict]:
            params, opt_states, buffer_state_epoch_carry, key = epoch_carry_state

            # TODO is it necessary to create global env step completion mask inside minibatch update?
            # or should we just do step rollout only
            global_completion_mask = (
                total_env_steps_at_epoch_start < target_env_steps_global
            ).astype(jnp.float32)
            epoch_mask = (epoch_idx < hyperparams_for_epochs.epochs).astype(jnp.float32)
            # Calculate per-core batch size
            num_devices = config.training.num_devices if config.training.num_devices > 0 else 1
            num_update_batches = (
                config.training.update_batch_size if config.training.update_batch_size > 0 else 1
            )
            cores_for_dist = num_devices * num_update_batches
            batch_size_per_core_for_sampling = (
                hyperparams_for_epochs.total_batch_size // cores_for_dist
            )

            # This batch_mask is to ensure we only sample valid number of items if max_batch_size (padding) > actual batch_size_per_core
            # However, the buffer_sample_fn itself should handle the effective batch_size for this HP.
            # The primary mask here is the update_mask for grads.
            update_mask = global_completion_mask * epoch_mask

            key, sample_key = jax.random.split(key)
            # buffer_sample_fn should use hyperparams_for_epochs.total_batch_size (per core)
            transition_sample = buffer_sample_fn(buffer_state_epoch_carry, sample_key)
            transitions: DQNTransition = transition_sample.experience

            # Normalize observations from buffer if normalize_observations is True
            obs_to_use_in_loss = jnp.where(
                hyperparams_for_epochs.normalize_observations > 0.5,
                normalize_fn(norm_params_at_epoch_start, transitions.obs.agent_view),
                transitions.obs.agent_view,
            )
            next_obs_to_use_in_loss = jnp.where(
                hyperparams_for_epochs.normalize_observations > 0.5,
                normalize_fn(norm_params_at_epoch_start, transitions.next_obs.agent_view),
                transitions.next_obs.agent_view,
            )
            transitions_for_loss = transitions._replace(
                obs=transitions.obs._replace(agent_view=obs_to_use_in_loss),
                next_obs=transitions.next_obs._replace(agent_view=next_obs_to_use_in_loss),
            )
            final_loss_mask = (
                transitions_for_loss.info["valid"] * update_mask
            )  # Mask based on valid transitions and epoch/completion

            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_EPOCH_LOOP},"
                + "epoch_idx:{ep_idx},total_batch_size_hp:{tbs_hp},batch_size_per_core:{bs_core},final_loss_mask_sum:{flms}",
                sid=sid_epoch_level,
                step=total_env_steps_at_epoch_start,
                ep_idx=epoch_idx,
                tbs_hp=hyperparams_for_epochs.total_batch_size,
                bs_core=batch_size_per_core_for_sampling,
                flms=jnp.sum(final_loss_mask),
            )

            compute_and_differentiate_q_loss = jax.value_and_grad(
                lambda online_params,
                target_params,
                transitions,
                hyperparams,
                loss_mask: _compute_q_loss_and_gradients(
                    online_params,
                    target_params,
                    transitions,
                    hyperparams,
                    loss_mask,
                    q_apply_fn,
                    hyperparam_non_vectorizeds.use_double_q,
                ),
                argnums=0,
                has_aux=True,
            )
            (loss_sum_value, q_loss_info), q_grads = compute_and_differentiate_q_loss(
                params.online,
                params.target,
                transitions_for_loss,
                hyperparams_for_epochs,
                final_loss_mask,
            )

            q_grads_scaled = jax.tree_util.tree_map(lambda x: x * update_mask, q_grads)
            loss_sum_value_scaled = loss_sum_value * update_mask
            q_loss_info_scaled = jax.tree_util.tree_map(lambda m: m * update_mask, q_loss_info)

            current_q_grads, current_loss_sum, current_aux_metrics = (
                q_grads_scaled,
                loss_sum_value_scaled,
                q_loss_info_scaled,
            )
            if pmean_axis_name_for_update_batch:
                (current_q_grads, current_loss_sum, current_aux_metrics) = jax.lax.pmean(
                    (current_q_grads, current_loss_sum, current_aux_metrics),
                    axis_name=pmean_axis_name_for_update_batch,
                )
            if pmean_axis_name_for_device:
                (current_q_grads, current_loss_sum, current_aux_metrics) = jax.lax.pmean(
                    (current_q_grads, current_loss_sum, current_aux_metrics),
                    axis_name=pmean_axis_name_for_device,
                )

            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_LOSS_CALC},"
                + "epoch_idx:{ep_idx},q_loss_sum:{qls},valid_samples:{vs}",
                sid=sid_epoch_level,
                step=total_env_steps_at_epoch_start,
                ep_idx=epoch_idx,
                qls=current_aux_metrics["q_loss_sum"],
                vs=current_aux_metrics["valid_samples"],
            )

            progress_fraction = jnp.clip(
                total_env_steps_at_epoch_start / jnp.maximum(target_env_steps_global, 1e-6),
                0.0,
                1.0,
            )
            lr_if_decayed = hyperparams_for_epochs.critic_lr * (1.0 - progress_fraction)
            effective_lr = jnp.where(
                hyperparams_for_epochs.decay_learning_rates > 0.5,
                jnp.maximum(lr_if_decayed, 1e-7),
                hyperparams_for_epochs.critic_lr,
            )

            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_OPTIM_STEP},epoch_idx:{{ep_idx}},"
                + "decay_lr_flag:{decay_f},initial_lr:{ilr},prog_frac:{pf},eff_lr:{elr}",
                sid=sid_epoch_level,
                step=total_env_steps_at_epoch_start,
                ep_idx=epoch_idx,
                decay_f=hyperparams_for_epochs.decay_learning_rates,
                ilr=hyperparams_for_epochs.critic_lr,
                pf=progress_fraction,
                elr=effective_lr,
            )

            opt_state_to_use = opt_states
            if isinstance(opt_states, tuple) and len(opt_states) >= 2:
                # For optimizer state tuples (e.g., (clip_state, adam_state, ...))
                clip_state, adam_state = opt_states[0], opt_states[1]
                updated_adam_hyperparams = adam_state.hyperparams.copy()
                updated_adam_hyperparams["learning_rate"] = effective_lr
                updated_clip_hyperparams = {"max_norm": hyperparams_for_epochs.max_grad_norm}
                updated_clip_state = clip_state._replace(hyperparams=updated_clip_hyperparams)
                updated_adam_state = adam_state._replace(hyperparams=updated_adam_hyperparams)
                opt_state_to_use = (
                    updated_clip_state,
                    updated_adam_state,
                ) + opt_states[2:]
            else:
                # If neither condition is met, raise an error
                raise TypeError(
                    "opt_states must be a tuple of states with 'hyperparams' attributes "
                    f"Got type: {type(opt_states)}"
                )

            q_updates, q_new_opt_state = q_update_fn(
                current_q_grads, opt_state_to_use, params.online
            )
            q_new_online_params = optax.apply_updates(params.online, q_updates)
            new_target_q_params = optax.incremental_update(
                q_new_online_params, params.target, hyperparams_for_epochs.tau
            )
            q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)
            num_valid_samples = current_aux_metrics["valid_samples"] + 1e-8
            mean_q_loss_log = current_loss_sum / num_valid_samples
            epoch_log_metrics = {"q_loss": mean_q_loss_log}
            return (
                q_new_params,
                q_new_opt_state,
                buffer_state_epoch_carry,
                key,
            ), epoch_log_metrics

        (
            (
                params_after_epochs,
                opt_states_after_epochs,
                buffer_state_after_epochs,
                key_after_epochs,
            ),
            train_metrics_after_epochs,
        ) = jax.lax.scan(_train_single_epoch, epoch_scan_init_state, epoch_indices)
        final_learner_state = learner_state_for_training._replace(
            params=params_after_epochs,
            opt_states=opt_states_after_epochs,
            buffer_state=buffer_state_after_epochs,
            key=key_after_epochs,
        )
        cycle_output = AnakinTrainOutput(
            learner_state=final_learner_state,
            episode_metrics=traj_batch.info,
            train_metrics=train_metrics_after_epochs,
        )
        return final_learner_state, cycle_output

    return _apply_single_update_cycle
