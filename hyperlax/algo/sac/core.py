import logging
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
from jumanji.env import Environment
from jumanji.types import TimeStep

from hyperlax.algo.sac.loss import _actor_loss_fn, _alpha_loss_fn, _q_loss_fn
from hyperlax.algo.sac.struct_sac import (
    SACLearnerState,
    SACOptStates,
    SACParams,
    SACTransition,
    SACVectorizedHyperparams,
    get_sac_max_masked_dims,
)
from hyperlax.base_types import AnakinTrainOutput, LogEnvState, OnlineAndTarget
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
# TAG_PREFIX,SID,CURRENT_ENV_STEP,CONTEXT_TAG,METRIC_NAME_1,{{metric_val_1}},...
JAX_LOG_PREFIX = "SAC_CORE"

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
        SACVectorizedHyperparams,
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
    warmup_factory_logger = logging.getLogger(f"{logger.name}.WARMUP_FN_FACTORY_SAC")
    if config.logger.enabled and config.logger.enable_jax_debug_prints:
        enable_jax_debug_for_context("SAC_CORE")
        enable_jax_debug_for_context(CTX_WARMUP_ROLLOUT)
        enable_jax_debug_for_context(CTX_WARMUP_STEP)

    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    max_dims = get_sac_max_masked_dims(
        algo_hp_batch, config.training.num_devices, config.training.update_batch_size
    )

    buffer_fns = build_masked_item_buffer(
        max_parallel_envs=max_dims.num_envs,
        buffer_size_per_env=max_dims.buffer_size // max_dims.num_envs,
        effective_buffer_size_per_env=max_dims.buffer_size,
        sample_batch_size=max_dims.batch_size,
    )
    add_to_buffer_fn = buffer_fns.add

    # Get actor network and initial (untrained) params for warmup action selection
    actor_apply_fn = nets_and_opts.networks["actor_network"].apply
    actor_params_for_warmup = nets_and_opts.params["actor_network"]

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
        current_hyperparams: SACVectorizedHyperparams,
    ) -> tuple[
        LogEnvState,
        TimeStep,
        chex.PRNGKey,
        MaskedItemBufferState,
        NormParams,
        chex.Array,
    ]:
        sid_warmup = current_hyperparams.sample_id
        env_steps_counter_warmup = jnp.zeros_like(current_hyperparams.actor_lr, dtype=jnp.float32)

        jax.debug.print(
            f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_WARMUP_ROLLOUT},start,warmup_len:{{wu_len}},num_envs:{{ne}}",
            sid=sid_warmup,
            step=env_steps_counter_warmup,
            wu_len=current_hyperparams.warmup_rollout_length,
            ne=current_hyperparams.total_num_envs,
        )

        def _step_environment_warmup(
            env_carry_state: tuple[LogEnvState, TimeStep, chex.PRNGKey, NormParams, chex.Array],
            step_idx: int,
        ) -> tuple[
            tuple[LogEnvState, TimeStep, chex.PRNGKey, NormParams, chex.Array],
            SACTransition,
        ]:
            env_state, timestep, key, norm_params, env_step_count = env_carry_state

            rollout_active_mask = (step_idx < current_hyperparams.warmup_rollout_length).astype(
                jnp.float32
            )
            env_active_mask = (
                env_indices_range_warmup < current_hyperparams.total_num_envs
            ).astype(jnp.float32)
            step_mask = rollout_active_mask * env_active_mask

            raw_obs_struct = timestep.observation
            raw_obs_view = raw_obs_struct.agent_view

            # 1. Normalize with the *current* stats (the ones from the previous step)
            normed_obs_if_true = normalize_observation_warmup(norm_params, raw_obs_view)
            obs_for_actor = jnp.where(
                current_hyperparams.normalize_observations > 0.5,
                normed_obs_if_true,
                raw_obs_view,
            )

            # 2. Update stats with the current raw observation for the *next* step
            updated_norm_params = update_normalization_stats_warmup(
                norm_params, raw_obs_view, weights=step_mask
            )

            step_mask_for_obs = step_mask[:, jnp.newaxis]
            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_WARMUP_STEP},"
                + "wu_step_idx:{wu_idx},norm_obs_flag:{no_f},raw_obs_sum:{ros},normed_obs_sum:{nos}",
                sid=sid_warmup,
                step=env_step_count,
                wu_idx=step_idx,
                no_f=current_hyperparams.normalize_observations,
                ros=jnp.sum(raw_obs_view * step_mask_for_obs),
                nos=jnp.sum(obs_for_actor * step_mask_for_obs),
            )

            # SELECT ACTION using the (untrained) actor network
            key, action_key = jax.random.split(key)
            actor_policy = actor_apply_fn(actor_params_for_warmup, obs_for_actor)
            actions = actor_policy.sample(seed=action_key)

            next_env_state, next_timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, actions)
            updated_env_step_count = env_step_count  # No step counting in warmup

            transition = SACTransition(
                obs=raw_obs_struct,  # Store UN-normalized obs struct
                action=actions,
                reward=next_timestep.reward,
                done=next_timestep.last().reshape(-1),
                next_obs=next_timestep.extras["next_obs"],  # Store UN-normalized next_obs struct
                info={
                    **next_timestep.extras["episode_metrics"],
                    "valid": step_mask,
                },
            )
            # 3. Carry forward the new stats
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
            _step_environment_warmup,
            scan_initial_carry,
            warmup_rollout_indices_range,
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


def build_sac_update_step_fn(
    env: Environment,
    nets_and_opts: NetworkAndOptimizerOnlyFns | NetworkAndOptimizerWithParamsTuple,
    normalizer_fns: tuple[InitNormFn, UpdateNormFn, NormalizeFn],
    config: BaseExperimentConfig,
    train_strategy: DistributionStrategy,
    hyperparam_batch_wrappers: dict[str, Any],
    hyperparam_non_vectorizeds: Any,
):
    actor_apply_fn = nets_and_opts.networks["actor_network"].apply
    q_apply_fn = nets_and_opts.networks["critic_network"].apply
    actor_update_fn = nets_and_opts.optimizers["actor_network"].update
    q_update_fn = nets_and_opts.optimizers["critic_network"].update
    alpha_update_fn = nets_and_opts.optimizers["alpha"].update

    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    max_dims = get_sac_max_masked_dims(
        algo_hp_batch, config.training.num_devices, config.training.update_batch_size
    )

    buffer_fns = build_masked_item_buffer(
        max_parallel_envs=max_dims.num_envs,
        buffer_size_per_env=max_dims.buffer_size // max_dims.num_envs,
        effective_buffer_size_per_env=max_dims.buffer_size,
        sample_batch_size=max_dims.batch_size,
    )

    pmean_axis_name = train_strategy.get_axis_spec("device").axis_name

    _, update_norm_stats_fn, normalize_fn = normalizer_fns
    rollout_indices = jnp.arange(max_dims.rollout_length)
    vec_env_indices = jnp.arange(max_dims.num_envs)
    epoch_indices = jnp.arange(max_dims.epochs)

    def _apply_single_update_cycle(
        learner_state: SACLearnerState, _: Any
    ) -> tuple[SACLearnerState, AnakinTrainOutput]:
        def _env_step_rollout(
            carry_state: SACLearnerState, step_idx: Any
        ) -> tuple[SACLearnerState, SACTransition]:
            """Steps the environment for one step of the rollout."""
            (
                params,
                opt_states,
                buffer_state,
                key,
                env_state,
                timestep,
                hyperparams,
                norm_params,
                env_steps,
            ) = carry_state

            rollout_active_mask = (step_idx < hyperparams.rollout_length).astype(jnp.float32)
            env_active_mask = (vec_env_indices < hyperparams.total_num_envs).astype(jnp.float32)
            step_mask = rollout_active_mask * env_active_mask

            raw_obs_struct = timestep.observation
            obs_view = raw_obs_struct.agent_view

            # 1. Normalize using the current stats and select for actor
            normed_obs_if_true = normalize_fn(norm_params, obs_view)
            obs_for_actor = jnp.where(
                hyperparams.normalize_observations > 0.5,
                normed_obs_if_true,
                obs_view,
            )

            # 2. Update stats for the next iteration using the raw observation and mask
            new_norm_params = update_norm_stats_fn(norm_params, obs_view, weights=step_mask)

            # Select Action
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params, obs_for_actor)
            action = actor_policy.sample(seed=policy_key)

            # Step Environment
            env_state, next_timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
            new_valid_steps = jnp.sum(step_mask).astype(jnp.float32)
            next_env_steps = env_steps + new_valid_steps

            # Store transition with UN-normalized observations
            transition = SACTransition(
                obs=raw_obs_struct,
                action=action,
                reward=next_timestep.reward,
                done=next_timestep.last(),
                next_obs=next_timestep.extras["next_obs"],
                info={**next_timestep.extras["episode_metrics"], "valid": step_mask},
            )

            # 3. Carry forward the new stats
            next_carry_state = SACLearnerState(
                params,
                opt_states,
                buffer_state,
                key,
                env_state,
                next_timestep,
                hyperparams,
                new_norm_params,
                next_env_steps,
            )
            return next_carry_state, transition

        learner_state, traj_batch = jax.lax.scan(_env_step_rollout, learner_state, rollout_indices)

        # Add trajectory to buffer, transposing from (seq, env) to (env, seq)
        traj_batch_swapped = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_state = buffer_fns.add(
            learner_state.buffer_state,
            traj_batch_swapped,
            traj_batch_swapped.info["valid"],
        )
        learner_state = learner_state._replace(buffer_state=buffer_state)

        def _update_epoch(carry_state: tuple, epoch_idx: Any) -> tuple[tuple, dict]:
            params, opt_states, buffer_state_epoch, key = carry_state
            hyperparams = learner_state.algo_hyperparams
            norm_params = learner_state.normalization_params
            non_vec_hps = hyperparam_non_vectorizeds
            sid_epoch = hyperparams.sample_id
            current_env_step_epoch = learner_state.total_env_steps_counter

            epoch_mask = (epoch_idx < hyperparams.epochs).astype(jnp.float32)

            key, sample_key, actor_key, q_key, alpha_key = jax.random.split(key, 5)

            # Sample from buffer
            sampled_transitions = buffer_fns.sample(buffer_state_epoch, sample_key).experience

            # Normalize observations from buffer on-the-fly
            normed_obs_view = jnp.where(
                hyperparams.normalize_observations > 0.5,
                normalize_fn(norm_params, sampled_transitions.obs.agent_view),
                sampled_transitions.obs.agent_view,
            )
            normed_next_obs_view = jnp.where(
                hyperparams.normalize_observations > 0.5,
                normalize_fn(norm_params, sampled_transitions.next_obs.agent_view),
                sampled_transitions.next_obs.agent_view,
            )

            transitions_for_loss = sampled_transitions._replace(
                obs=sampled_transitions.obs._replace(agent_view=normed_obs_view),
                next_obs=sampled_transitions.next_obs._replace(agent_view=normed_next_obs_view),
            )

            final_loss_mask = transitions_for_loss.info["valid"] * epoch_mask

            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_EPOCH_LOOP},"
                + "epoch_idx:{ep_idx},loss_mask_sum:{lms}",
                sid=sid_epoch,
                step=current_env_step_epoch,
                ep_idx=epoch_idx,
                lms=jnp.sum(final_loss_mask),
            )

            # --- 1. COMPUTE ALL GRADIENTS BASED ON CURRENT PARAMS ---
            alpha = jax.lax.stop_gradient(jnp.exp(params.log_alpha))

            # Critic Grads
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                params.q_params.online,
                params.q_params.target,
                params.actor_params,
                alpha,
                transitions_for_loss,
                hyperparams,
                q_apply_fn,
                actor_apply_fn,
                q_key,
                final_loss_mask,
            )

            # Actor Grads
            actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
            actor_grads, actor_loss_info = actor_grad_fn(
                params.actor_params,
                params.q_params.online,
                alpha,
                transitions_for_loss,
                actor_apply_fn,
                q_apply_fn,
                actor_key,
                final_loss_mask,
            )

            # Alpha Grads (if autotuning)
            if non_vec_hps.autotune:
                alpha_grad_fn = jax.grad(_alpha_loss_fn, has_aux=True)
                alpha_grads, alpha_loss_info = alpha_grad_fn(
                    params.log_alpha,
                    params.actor_params,
                    transitions_for_loss,
                    hyperparams,
                    actor_apply_fn,
                    alpha_key,
                    final_loss_mask,
                )
            else:
                alpha_grads = jax.tree_util.tree_map(jnp.zeros_like, params.log_alpha)
                alpha_loss_info = {
                    "alpha_loss": 0.0,
                    "alpha": jnp.exp(params.log_alpha),
                }

            # --- 2. AGGREGATE GRADIENTS AND LOSSES ACROSS DEVICES ---
            (
                q_grads,
                q_loss_info,
                actor_grads,
                actor_loss_info,
                alpha_grads,
                alpha_loss_info,
            ) = jax.lax.pmean(
                (
                    q_grads,
                    q_loss_info,
                    actor_grads,
                    actor_loss_info,
                    alpha_grads,
                    alpha_loss_info,
                ),
                axis_name=pmean_axis_name,
            )

            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_LOSS_CALC},"
                + "q_loss:{ql},actor_loss:{al},alpha_loss:{alpha_l},q_error:{qe},q1:{q1},q2:{q2},gamma:{g},tau:{t}",
                sid=sid_epoch,
                step=current_env_step_epoch,
                ql=q_loss_info["q_loss"],
                al=actor_loss_info["actor_loss"],
                alpha_l=alpha_loss_info["alpha_loss"],
                qe=q_loss_info["q_error"],
                q1=q_loss_info["q1_pred"],
                q2=q_loss_info["q2_pred"],
                g=hyperparams.gamma,
                t=hyperparams.tau,
            )

            # --- 3. APPLY UPDATES ---

            # Update Critic
            original_critic_opt_state = opt_states.q_opt_state
            updated_adam_hps_critic = original_critic_opt_state[1].hyperparams.copy()
            updated_adam_hps_critic["learning_rate"] = hyperparams.q_lr
            updated_clip_hps_critic = {"max_norm": hyperparams.max_grad_norm}
            critic_opt_state_to_use = (
                original_critic_opt_state[0]._replace(hyperparams=updated_clip_hps_critic),
                original_critic_opt_state[1]._replace(hyperparams=updated_adam_hps_critic),
            )
            q_updates, new_q_opt_state = q_update_fn(
                q_grads, critic_opt_state_to_use, params.q_params.online
            )
            new_online_q_params = optax.apply_updates(params.q_params.online, q_updates)

            # Update Actor
            original_actor_opt_state = opt_states.actor_opt_state
            updated_adam_hps_actor = original_actor_opt_state[1].hyperparams.copy()
            updated_adam_hps_actor["learning_rate"] = hyperparams.actor_lr
            updated_clip_hps_actor = {"max_norm": hyperparams.max_grad_norm}
            actor_opt_state_to_use = (
                original_actor_opt_state[0]._replace(hyperparams=updated_clip_hps_actor),
                original_actor_opt_state[1]._replace(hyperparams=updated_adam_hps_actor),
            )
            actor_updates, new_actor_opt_state = actor_update_fn(
                actor_grads, actor_opt_state_to_use, params.actor_params
            )
            new_actor_params = optax.apply_updates(params.actor_params, actor_updates)

            # Update Alpha (if autotuning)
            new_log_alpha, new_alpha_opt_state = (
                params.log_alpha,
                opt_states.alpha_opt_state,
            )
            if non_vec_hps.autotune:
                original_alpha_opt_state = opt_states.alpha_opt_state
                updated_adam_hps_alpha = original_alpha_opt_state[1].hyperparams.copy()
                updated_adam_hps_alpha["learning_rate"] = hyperparams.alpha_lr
                updated_clip_hps_alpha = {"max_norm": hyperparams.max_grad_norm}
                alpha_opt_state_to_use = (
                    original_alpha_opt_state[0]._replace(hyperparams=updated_clip_hps_alpha),
                    original_alpha_opt_state[1]._replace(hyperparams=updated_adam_hps_alpha),
                )
                alpha_updates, new_alpha_opt_state = alpha_update_fn(
                    alpha_grads, alpha_opt_state_to_use, params.log_alpha
                )
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
                jax.debug.print(
                    f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_LOSS_CALC},alpha_loss:{{al}},alpha:{{a}}",
                    sid=sid_epoch,
                    step=current_env_step_epoch,
                    al=alpha_loss_info["alpha_loss"],
                    a=alpha_loss_info["alpha"],
                )

            # Update Target Q Network
            new_target_q_params = optax.incremental_update(
                new_online_q_params, params.q_params.target, hyperparams.tau
            )

            # --- 4. ASSEMBLE NEW STATE ---
            new_params = SACParams(
                new_actor_params,
                OnlineAndTarget(new_online_q_params, new_target_q_params),
                new_log_alpha,
            )
            new_opt_states = SACOptStates(
                new_actor_opt_state, new_q_opt_state, new_alpha_opt_state
            )
            loss_info = {**q_loss_info, **actor_loss_info, **alpha_loss_info}

            # Mask the parameter update based on epoch mask
            final_params = jax.tree_util.tree_map(
                lambda old, new: jnp.where(epoch_mask > 0.5, new, old),
                params,
                new_params,
            )
            final_opt_states = jax.tree_util.tree_map(
                lambda old, new: jnp.where(epoch_mask > 0.5, new, old),
                opt_states,
                new_opt_states,
            )

            return (final_params, final_opt_states, buffer_state_epoch, key), loss_info

        (
            (
                final_params,
                final_opt_states,
                final_buffer_state,
                final_key,
            ),
            train_metrics,
        ) = jax.lax.scan(
            _update_epoch,
            (
                learner_state.params,
                learner_state.opt_states,
                learner_state.buffer_state,
                learner_state.key,
            ),
            epoch_indices,
        )

        final_learner_state = learner_state._replace(
            params=final_params,
            opt_states=final_opt_states,
            buffer_state=final_buffer_state,
            key=final_key,
        )

        return final_learner_state, AnakinTrainOutput(
            learner_state=final_learner_state,
            episode_metrics=traj_batch.info,
            train_metrics=train_metrics,
        )

    return _apply_single_update_cycle
