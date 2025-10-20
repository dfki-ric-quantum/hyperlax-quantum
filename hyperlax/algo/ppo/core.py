import logging
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
from chex import Scalar
from jax import Array
from jumanji.env import Environment

from hyperlax.algo.ppo.loss import clipped_value_loss_masked, ppo_clip_loss_masked
from hyperlax.algo.ppo.struct_ppo import (
    PPOOnPolicyLearnerState,
    PPOTransition,
    get_ppo_max_masked_dims,
)
from hyperlax.base_types import (
    ActorCriticOptStates,
    ActorCriticParams,
    AnakinTrainOutput,
)
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.env.utils import identify_action_space_type
from hyperlax.layout.axes import DistributionStrategy
from hyperlax.logger.jax_debug import enable_jax_debug_for_context
from hyperlax.normalizer.running_stats import InitNormFn, NormalizeFn, UpdateNormFn
from hyperlax.utils.algo_setup import (
    NetworkAndOptimizerOnlyFns,
    NetworkAndOptimizerWithParamsTuple,
)
from hyperlax.utils.jax_utils import merge_leading_dims

logger = logging.getLogger(__name__)

# JAX Debug Log Message Structure:
# TAG_PREFIX,SID,CURRENT_ENV_STEP,CONTEXT_TAG,
# METRIC_NAME_1,{{metric_val_1}},METRIC_NAME_2,{{metric_val_2}},...
# Example: JAX_PPO_CORE,123,50000,ROLLOUT_NORM_OBS_FLAG,normalize_observations_flag,True|
# raw_obs_sum,10.5

JAX_LOG_PREFIX = "PPO_CORE"  # Unified prefix for all logs from this core logic
# Context tags for different stages
CTX_ROLLOUT = "ROLLOUT"
CTX_ROLLOUT_STEP = "ROLLOUT_STEP"
CTX_GAE = "GAE_CALC"
CTX_EPOCH_LOOP = "EPOCH_LOOP"
CTX_MINIBATCH_LOOP = "MINIBATCH_LOOP"
CTX_LOSS_ACTOR = "LOSS_ACTOR"
CTX_LOSS_CRITIC = "LOSS_CRITIC"
CTX_OPTIM_PARAMS = "OPTIM_PARAMS"


def batch_truncated_generalized_advantage_estimation(
    r_t: Array,
    discount_t: Array,
    lambda_: Array | Scalar,
    values: Array,
    stop_target_gradients: bool = False,
    time_major: bool = False,
    standardize_advantages: Array | Scalar | bool = False,
    truncation_t: Array | None = None,
) -> tuple[Array, Array]:
    """Computes truncated generalized advantage estimates for batched sequences of length k.
    MODIFIED to handle standardize_advantages as a potentially traced JAX array.
    """
    chex.assert_rank([r_t, values, discount_t], 2)
    chex.assert_type([r_t, values, discount_t], float)
    lambda_ = jnp.ones_like(discount_t) * lambda_

    if truncation_t is None:
        truncation_t = jnp.zeros_like(discount_t)
    else:
        chex.assert_rank([truncation_t], 2)
        chex.assert_equal_shape([truncation_t, discount_t])
        truncation_t = truncation_t.astype(float)
    if not time_major:
        r_t = jnp.transpose(r_t, (1, 0))
        discount_t = jnp.transpose(discount_t, (1, 0))
        values = jnp.transpose(values, (1, 0))
        lambda_ = jnp.transpose(lambda_, (1, 0))
        truncation_t = jnp.transpose(truncation_t, (1, 0))
    delta_t = r_t + discount_t * values[1:] - values[:-1]

    def _body(acc: Array, xs: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
        deltas, discounts, lambda_val, truncation_val = xs  # Use lambda_val to avoid conflict
        acc = deltas + discounts * lambda_val * acc * (1.0 - truncation_val)
        return acc, acc

    _, advantage_t = jax.lax.scan(
        _body,
        jnp.zeros(r_t.shape[1]),  # Initial accumulator for advantages
        (
            delta_t,
            discount_t,
            lambda_,
            truncation_t,
        ),  # Use lambda_ here from outer scope
        reverse=True,
    )

    # Target values are computed based on the unstandardized advantages.
    target_values = values[:-1] + advantage_t

    # Ensure standardize_advantages is a JAX array for jax.lax.cond predicate
    # If it's a Python bool, convert it. If it's already a JAX array/tracer, this is a no-op.
    cond_standardize_jax = jnp.asarray(standardize_advantages)

    # If the JAX array is float (e.g., 0.0 or 1.0), compare it to make a boolean condition.
    # This handles the case where boolean HPs are stored as floats in the data_values array.
    if jnp.issubdtype(cond_standardize_jax.dtype, jnp.floating):
        cond_predicate = cond_standardize_jax > 0.5
    elif jnp.issubdtype(cond_standardize_jax.dtype, jnp.bool_):
        cond_predicate = cond_standardize_jax
    else:
        raise TypeError(
            f"Unexpected dtype '{cond_standardize_jax.dtype}' for standardize_advantages in GAE."
        )

    def _do_standardize_advantages(adv: Array) -> Array:
        return jax.nn.standardize(adv)

    def _identity_advantages(adv: Array) -> Array:
        return adv

    advantage_t = jax.lax.cond(
        cond_predicate, _do_standardize_advantages, _identity_advantages, advantage_t
    )
    if not time_major:
        # Transpose after potential standardization effects if targets depended on standardized GAE
        advantage_t = jnp.transpose(advantage_t, (1, 0))
        target_values = jnp.transpose(target_values, (1, 0))
    if (
        stop_target_gradients
    ):  # This is a Python bool, compile-time constant for the JITted function
        advantage_t = jax.lax.stop_gradient(advantage_t)
        target_values = jax.lax.stop_gradient(target_values)
    return advantage_t, target_values


def _compute_actor_loss_and_gradients(
    actor_params,
    traj_batch,
    advantages,
    algo_hps,
    network_hps,
    mask,
    actor_apply_fn,
    key_actor,
    is_continuous_action: bool,
):
    """Compute actor loss and gradients for PPO update."""
    actor_policy = actor_apply_fn(actor_params, traj_batch.obs, network_hps)
    log_prob = actor_policy.log_prob(traj_batch.action)
    loss_sum, num_valid_samples = ppo_clip_loss_masked(
        log_prob, traj_batch.log_prob, advantages, algo_hps.clip_eps, mask=mask
    )
    if is_continuous_action:
        entropy = actor_policy.entropy(seed=key_actor)
    else:
        entropy = actor_policy.entropy()  # No seed for discrete (Categorical)

    entropy_sum = jnp.sum(entropy * mask)
    total_loss_sum = loss_sum - algo_hps.ent_coef * entropy_sum
    loss_info = {
        "actor_loss_sum": loss_sum,
        "entropy_sum": entropy_sum,
        "valid_samples": num_valid_samples,
    }
    return total_loss_sum, loss_info


def _compute_critic_loss_and_gradients(
    critic_params, traj_batch, targets, algo_hps, network_hps, mask, critic_apply_fn
):
    """Compute critic loss and gradients for PPO update."""
    value = critic_apply_fn(critic_params, traj_batch.obs, network_hps)
    loss_sum, num_valid_samples = clipped_value_loss_masked(
        value, traj_batch.value, targets, algo_hps.clip_eps, mask=mask
    )
    total_loss_sum = algo_hps.vf_coef * loss_sum
    loss_info = {
        "value_loss_sum": loss_sum,
        "valid_samples": num_valid_samples,
    }
    return total_loss_sum, loss_info


def build_ppo_update_step_fn(
    env: Environment,
    nets_and_opts: NetworkAndOptimizerOnlyFns | NetworkAndOptimizerWithParamsTuple,
    normalizer_fns: tuple[InitNormFn, UpdateNormFn, NormalizeFn],
    config: BaseExperimentConfig,
    train_strategy: DistributionStrategy,
    hyperparam_batch_wrappers: dict[str, Any],
    hyperparam_non_vectorizeds: Any,
):
    actor_apply_fn = nets_and_opts.networks["actor_network"].apply
    critic_apply_fn = nets_and_opts.networks["critic_network"].apply
    actor_update_fn = nets_and_opts.optimizers["actor_network"].update
    critic_update_fn = nets_and_opts.optimizers["critic_network"].update

    algo_hp_batch = hyperparam_batch_wrappers["algo"]
    max_dims = get_ppo_max_masked_dims(
        algo_hp_batch,
        num_devices=config.training.num_devices,
        update_batch_size=config.training.update_batch_size,
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

    max_num_envs, max_rollout_length, max_epochs, max_num_minibatches = (
        max_dims.envs,
        max_dims.rollout,
        max_dims.epochs,
        max_dims.minibatches,
    )
    target_env_steps_global = config.training.total_timesteps

    core_logic_logger = logging.getLogger(f"{__name__}.PPO_CORE_UPDATE_LOGIC")
    action_space_type = identify_action_space_type(env)
    is_continuous_action = action_space_type == "continuous"
    core_logic_logger.debug(
        f"Action space type: {action_space_type}, is_continuous_action: {is_continuous_action}"
    )

    if config.logger.enabled and config.logger.enable_jax_debug_prints:
        logger.debug("Enabling jax tags. Be hold, it might be crowded!")
        enable_jax_debug_for_context(CTX_EPOCH_LOOP)
        enable_jax_debug_for_context(CTX_ROLLOUT_STEP)
        enable_jax_debug_for_context(CTX_GAE)
        enable_jax_debug_for_context(CTX_MINIBATCH_LOOP)
        enable_jax_debug_for_context(CTX_LOSS_ACTOR)
        enable_jax_debug_for_context(CTX_LOSS_CRITIC)
        enable_jax_debug_for_context(CTX_OPTIM_PARAMS)

    core_logic_logger.debug(
        f"Max dims for scans: num_envs={max_num_envs}, "
        f"rollout={max_rollout_length}, epochs={max_epochs}, num_minibatches={max_num_minibatches}"
    )
    _, update_norm_stats_fn, normalize_fn = normalizer_fns
    rollout_indices = jnp.arange(max_rollout_length)
    vec_env_indices = jnp.arange(max_num_envs)
    epoch_indices = jnp.arange(max_epochs)
    # minibatch_indices = jnp.arange(max_num_minibatches)

    def _apply_single_update_cycle(
        learner_state_unit: PPOOnPolicyLearnerState, _: Any
    ) -> tuple[PPOOnPolicyLearnerState, AnakinTrainOutput]:
        # _instance_log_id_cycle_scalar = learner_state_unit.hyperparams.sample_id
        def _step_environment_rollout(
            env_carry_state: PPOOnPolicyLearnerState, step_idx: Any
        ) -> tuple[PPOOnPolicyLearnerState, PPOTransition]:
            (
                params,
                opt_states,
                key,
                env_state,
                timestep,
                algo_hps,
                actor_network_hps,
                critic_network_hps,
                norm_params,
                env_step_counter,
            ) = env_carry_state
            step_sid = algo_hps.sample_id

            core_masks = {
                "global_completion": (env_step_counter < target_env_steps_global).astype(
                    jnp.float32
                ),
                "rollout": (step_idx < algo_hps.rollout_length).astype(jnp.float32),
                "env": (vec_env_indices < algo_hps.total_num_envs).astype(jnp.float32),
            }
            step_mask = core_masks["global_completion"] * core_masks["rollout"] * core_masks["env"]
            raw_obs = timestep.observation.agent_view
            obs_for_norm_update = raw_obs
            norm_params = update_norm_stats_fn(norm_params, obs_for_norm_update, weights=step_mask)
            normed_obs_if_true = normalize_fn(norm_params, raw_obs)

            step_mask_for_obs = step_mask[:, jnp.newaxis]
            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{id_val}},{{step}},{CTX_ROLLOUT_STEP},"
                + "rollout_step_idx:{r_idx},norm_obs_flag:{n_flag},raw_obs_sum:{r_sum},normed_obs_sum:{no_sum}",
                id_val=step_sid,
                step=env_step_counter,
                r_idx=step_idx,
                n_flag=algo_hps.normalize_observations,
                r_sum=jnp.sum(raw_obs * step_mask_for_obs),
                no_sum=jnp.sum(normed_obs_if_true * step_mask_for_obs),
            )

            normed_obs = jnp.where(algo_hps.normalize_observations, normed_obs_if_true, raw_obs)

            key, policy_key = jax.random.split(key)
            # The network wrappers will handle the =network_hps= struct internally.
            actor_policy = actor_apply_fn(params.actor_params, normed_obs, actor_network_hps)
            value = critic_apply_fn(params.critic_params, normed_obs, critic_network_hps)

            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{id_val}},{{step}},{CTX_ROLLOUT_STEP},"
                + "rollout_step_idx:{r_idx},action_mean:{a_mean},action_std:{a_std},value_mean:{v_mean}",
                id_val=step_sid,
                step=env_step_counter,
                r_idx=step_idx,
                # Correctly expand mask for broadcasting and calculate mean/std over all valid action components.
                a_mean=jnp.sum(action * step_mask[:, jnp.newaxis])
                / jnp.maximum(1.0, jnp.sum(step_mask) * action.shape[-1]),
                a_std=jnp.sqrt(
                    jnp.sum(
                        jnp.square(
                            action
                            - (
                                jnp.sum(action * step_mask[:, jnp.newaxis])
                                / jnp.maximum(1.0, jnp.sum(step_mask) * action.shape[-1])
                            )
                        )
                        * step_mask[:, jnp.newaxis]
                    )
                    / jnp.maximum(1.0, jnp.sum(step_mask) * action.shape[-1])
                ),
                v_mean=jnp.sum(value * step_mask) / jnp.maximum(1.0, jnp.sum(step_mask)),
            )

            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)
            new_valid_steps = jnp.sum(step_mask).astype(jnp.float32)
            env_step_counter_new = env_step_counter + new_valid_steps
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            transition = PPOTransition(
                obs=normed_obs,
                action=action,
                reward=timestep.reward,
                done=done,
                truncated=truncated,
                value=value,
                log_prob=log_prob,
                info={**timestep.extras["episode_metrics"], "valid": step_mask},
            )
            new_env_carry_state = PPOOnPolicyLearnerState(
                params,
                opt_states,
                key,
                env_state,
                timestep,
                algo_hps,
                actor_network_hps,
                critic_network_hps,
                norm_params,
                env_step_counter_new,
            )
            return new_env_carry_state, transition

        learner_state_after_rollout, traj_batch = jax.lax.scan(
            _step_environment_rollout, learner_state_unit, rollout_indices
        )
        (
            params,
            opt_states,
            key,
            env_state,
            timestep,
            algo_hps,
            actor_network_hps,
            critic_network_hps,
            norm_params,
            total_env_steps,
        ) = learner_state_after_rollout
        sid_gae_scalar = algo_hps.sample_id
        current_step_gae = total_env_steps

        raw_final_obs = timestep.observation.agent_view
        normed_final_obs_if_true = normalize_fn(norm_params, raw_final_obs)

        jax.debug.print(
            f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_GAE},final_obs_norm_flag:{{n_flag}}",
            sid=sid_gae_scalar,
            step=current_step_gae,
            n_flag=algo_hps.normalize_observations,
        )

        normed_final_obs = jnp.where(
            algo_hps.normalize_observations, normed_final_obs_if_true, raw_final_obs
        )

        last_val = critic_apply_fn(params.critic_params, normed_final_obs)
        r_t, v_t = (
            traj_batch.reward,
            jnp.concatenate([traj_batch.value, last_val[None, ...]], axis=0),
        )
        done_t, truncated_t = (
            traj_batch.done.astype(jnp.float32),
            traj_batch.truncated.astype(jnp.float32),
        )
        discount_t = (1.0 - done_t) * algo_hps.gamma
        rollout_mask_for_gae = jnp.arange(max_rollout_length)[:, None] < algo_hps.rollout_length
        env_mask_for_gae = jnp.arange(max_num_envs)[None, :] < algo_hps.total_num_envs
        valid_mask_for_gae = rollout_mask_for_gae * env_mask_for_gae

        jax.debug.print(
            f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_GAE},standardize_adv_flag:{{sa_flag}}",
            sid=sid_gae_scalar,
            step=current_step_gae,
            sa_flag=algo_hps.standardize_advantages,
        )

        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            discount_t,
            algo_hps.gae_lambda,
            v_t,
            time_major=True,
            standardize_advantages=algo_hps.standardize_advantages,
            truncation_t=truncated_t * valid_mask_for_gae,
        )

        jax.debug.print(
            f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_GAE},"
            + "adv_mean:{adv_m},adv_std:{adv_s},targets_mean:{t_m},targets_std:{t_s}",
            sid=sid_gae_scalar,
            step=current_step_gae,
            adv_m=jnp.mean(advantages),
            adv_s=jnp.std(advantages),
            t_m=jnp.mean(targets),
            t_s=jnp.std(targets),
        )

        epoch_scan_init_state = (params, opt_states, key)

        def _train_single_epoch(epoch_carry_state: tuple, epoch_idx: Any) -> tuple[tuple, dict]:
            params_epoch, opt_states_epoch, key_epoch = epoch_carry_state
            sid_epoch = algo_hps.sample_id
            current_step_epoch = total_env_steps

            epoch_mask = (epoch_idx < algo_hps.epochs).astype(jnp.float32)
            global_completion_mask = (total_env_steps < target_env_steps_global).astype(
                jnp.float32
            )
            update_mask = epoch_mask * global_completion_mask
            batch_size = max_rollout_length * max_num_envs

            batch_data = (traj_batch, advantages, targets)
            batch_data_flat = jax.tree_util.tree_map(
                lambda x: merge_leading_dims(x, 2), batch_data
            )
            key_epoch, shuffle_key = jax.random.split(key_epoch)
            permutation = jax.random.permutation(shuffle_key, batch_size)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch_data_flat
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [max_num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            minibatch_scan_init_state = (params_epoch, opt_states_epoch, key_epoch)

            jax.debug.print(
                f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_EPOCH_LOOP},"
                + "epoch_idx:{ep_idx},update_mask_sum:{upd_mask_sum}",
                sid=sid_epoch,
                step=current_step_epoch,
                ep_idx=epoch_idx,
                upd_mask_sum=jnp.sum(update_mask),
            )

            def _train_single_minibatch(
                batch_carry_state: tuple, minibatch_data_and_idx: tuple
            ) -> tuple[tuple, dict]:
                params_mb, opt_states_mb, key_mb = batch_carry_state
                minibatch_idx, (traj_batch_mb, advantages_mb, targets_mb) = minibatch_data_and_idx
                sid_mb = algo_hps.sample_id

                minibatch_mask_val = (minibatch_idx < algo_hps.num_minibatches).astype(jnp.float32)
                final_minibatch_mask = update_mask * minibatch_mask_val
                # traj_batch_mb.obs.agent_view.shape[0] if hasattr(traj_batch_mb.obs, 'agent_view') else traj_batch_mb.obs.shape[0]
                obs_shape_for_mask = traj_batch_mb.obs.shape[0]
                data_validity_mask = traj_batch_mb.info.get("valid", jnp.ones(obs_shape_for_mask))
                combined_mask = data_validity_mask * final_minibatch_mask
                key_mb, actor_key = jax.random.split(key_mb)

                actor_grad_fn = jax.grad(_compute_actor_loss_and_gradients, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params_mb.actor_params,
                    traj_batch_mb,
                    advantages_mb,
                    algo_hps,
                    actor_network_hps,
                    combined_mask,
                    actor_apply_fn,
                    actor_key,
                    is_continuous_action,
                )
                critic_grad_fn = jax.grad(_compute_critic_loss_and_gradients, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params_mb.critic_params,
                    traj_batch_mb,
                    targets_mb,
                    algo_hps,
                    critic_network_hps,
                    combined_mask,
                    critic_apply_fn,
                )
                actor_grads_masked = jax.tree_util.tree_map(
                    lambda x: x * final_minibatch_mask, actor_grads
                )
                critic_grads_masked = jax.tree_util.tree_map(
                    lambda x: x * final_minibatch_mask, critic_grads
                )
                current_actor_grads, current_critic_grads = (
                    actor_grads_masked,
                    critic_grads_masked,
                )
                current_actor_loss_info, current_critic_loss_info = (
                    actor_loss_info,
                    critic_loss_info,
                )

                if pmean_axis_name_for_update_batch:
                    (
                        current_actor_grads,
                        current_critic_grads,
                        current_actor_loss_info,
                        current_critic_loss_info,
                    ) = jax.lax.pmean(
                        (
                            current_actor_grads,
                            current_critic_grads,
                            current_actor_loss_info,
                            current_critic_loss_info,
                        ),
                        axis_name=pmean_axis_name_for_update_batch,
                    )
                if pmean_axis_name_for_device:
                    (
                        current_actor_grads,
                        current_critic_grads,
                        current_actor_loss_info,
                        current_critic_loss_info,
                    ) = jax.lax.pmean(
                        (
                            current_actor_grads,
                            current_critic_grads,
                            current_actor_loss_info,
                            current_critic_loss_info,
                        ),
                        axis_name=pmean_axis_name_for_device,
                    )

                jax.debug.print(
                    f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_MINIBATCH_LOOP},"
                    + "mb_idx:{mb_idx},actor_loss_sum:{al_sum},critic_loss_sum:{cl_sum},entropy_sum:{e_sum},valid_samples:{vs}",
                    sid=sid_mb,
                    step=total_env_steps,
                    mb_idx=minibatch_idx,
                    al_sum=current_actor_loss_info["actor_loss_sum"],
                    cl_sum=current_critic_loss_info["value_loss_sum"],
                    e_sum=current_actor_loss_info["entropy_sum"],
                    vs=current_actor_loss_info["valid_samples"],
                )

                progress_fraction = jnp.clip(
                    total_env_steps / jnp.maximum(target_env_steps_global, 1e-6),
                    0.0,
                    1.0,
                )
                lr_if_decayed_actor = algo_hps.actor_lr * (1.0 - progress_fraction)
                lr_if_decayed_critic = algo_hps.critic_lr * (1.0 - progress_fraction)

                jax.debug.print(
                    f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_OPTIM_PARAMS},"
                    + "decay_lr_flag:{decay_val},initial_actor_lr:{ial},initial_critic_lr:{icl},prog_frac:{pf}",
                    sid=sid_mb,
                    step=total_env_steps,
                    decay_val=algo_hps.decay_learning_rates,
                    ial=algo_hps.actor_lr,
                    icl=algo_hps.critic_lr,
                    pf=progress_fraction,
                )

                effective_actor_lr = jnp.where(
                    algo_hps.decay_learning_rates,
                    jnp.maximum(lr_if_decayed_actor, 1e-7),
                    algo_hps.actor_lr,
                )
                effective_critic_lr = jnp.where(
                    algo_hps.decay_learning_rates,
                    jnp.maximum(lr_if_decayed_critic, 1e-7),
                    algo_hps.critic_lr,
                )

                jax.debug.print(
                    f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_OPTIM_PARAMS},effective_actor_lr:{{eal}},effective_critic_lr:{{ecl}}",
                    sid=sid_mb,
                    step=total_env_steps,
                    eal=effective_actor_lr,
                    ecl=effective_critic_lr,
                )

                original_actor_opt_state, original_critic_opt_state = (
                    opt_states_mb.actor_opt_state,
                    opt_states_mb.critic_opt_state,
                )
                actor_opt_state_to_use, critic_opt_state_to_use = (
                    original_actor_opt_state,
                    original_critic_opt_state,
                )
                # Update actor optimizer state
                if (
                    isinstance(original_actor_opt_state, tuple)
                    and len(original_actor_opt_state) >= 2
                ):
                    # Update hyperparams for adam and clip optimizers
                    updated_adam_hyperparams_actor = original_actor_opt_state[1].hyperparams.copy()
                    updated_adam_hyperparams_actor["learning_rate"] = effective_actor_lr
                    updated_clip_hyperparams_actor = {"max_norm": algo_hps.max_grad_norm}
                    actor_opt_state_to_use = (
                        original_actor_opt_state[0]._replace(
                            hyperparams=updated_clip_hyperparams_actor
                        ),
                        original_actor_opt_state[1]._replace(
                            hyperparams=updated_adam_hyperparams_actor
                        ),
                    ) + original_actor_opt_state[2:]
                else:
                    raise TypeError(
                        "original_actor_opt_state must be a tuple of states with 'hyperparams' attributes."
                        f" Got type: {type(original_actor_opt_state)}"
                    )
                # Update critic optimizer state
                if (
                    isinstance(original_critic_opt_state, tuple)
                    and len(original_critic_opt_state) >= 2
                ):
                    updated_adam_hyperparams_critic = original_critic_opt_state[
                        1
                    ].hyperparams.copy()
                    updated_adam_hyperparams_critic["learning_rate"] = effective_critic_lr
                    updated_clip_hyperparams_critic = {"max_norm": algo_hps.max_grad_norm}
                    critic_opt_state_to_use = (
                        original_critic_opt_state[0]._replace(
                            hyperparams=updated_clip_hyperparams_critic
                        ),
                        original_critic_opt_state[1]._replace(
                            hyperparams=updated_adam_hyperparams_critic
                        ),
                    ) + original_critic_opt_state[2:]
                else:
                    raise TypeError(
                        "original_critic_opt_state must be a tuple of states with 'hyperparams' attributes."
                        f" Got type: {type(original_critic_opt_state)}"
                    )

                actor_mn_dbg, actor_lr_dbg = (
                    actor_opt_state_to_use[0].hyperparams["max_norm"],
                    actor_opt_state_to_use[1].hyperparams["learning_rate"],
                )
                critic_mn_dbg, critic_lr_dbg = (
                    critic_opt_state_to_use[0].hyperparams["max_norm"],
                    critic_opt_state_to_use[1].hyperparams["learning_rate"],
                )
                jax.debug.print(
                    f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_OPTIM_PARAMS},actor_max_norm_eff:{{mn}},actor_lr_eff:{{lr}}",
                    sid=sid_mb,
                    step=total_env_steps,
                    mn=actor_mn_dbg,
                    lr=actor_lr_dbg,
                )
                jax.debug.print(
                    f"{JAX_LOG_PREFIX},{{sid}},{{step}},{CTX_OPTIM_PARAMS},critic_max_norm_eff:{{mn}},critic_lr_eff:{{lr}}",
                    sid=sid_mb,
                    step=total_env_steps,
                    mn=critic_mn_dbg,
                    lr=critic_lr_dbg,
                )

                actor_updates, new_actor_opt_state = actor_update_fn(
                    current_actor_grads, actor_opt_state_to_use, params_mb.actor_params
                )
                new_actor_params = optax.apply_updates(params_mb.actor_params, actor_updates)
                critic_updates, new_critic_opt_state = critic_update_fn(
                    current_critic_grads,
                    critic_opt_state_to_use,
                    params_mb.critic_params,
                )
                new_critic_params = optax.apply_updates(params_mb.critic_params, critic_updates)

                new_params_mb = ActorCriticParams(new_actor_params, new_critic_params)
                new_opt_states_mb = ActorCriticOptStates(new_actor_opt_state, new_critic_opt_state)
                minibatch_loss_info = {
                    **current_actor_loss_info,
                    **current_critic_loss_info,
                }
                return (new_params_mb, new_opt_states_mb, key_mb), minibatch_loss_info

            minibatch_data_with_indices = (jnp.arange(max_num_minibatches), minibatches)
            (
                (
                    params_after_minibatches,
                    opt_states_after_minibatches,
                    key_after_minibatches,
                ),
                minibatch_loss_infos,
            ) = jax.lax.scan(
                _train_single_minibatch,
                minibatch_scan_init_state,
                minibatch_data_with_indices,
            )
            return (
                params_after_minibatches,
                opt_states_after_minibatches,
                key_after_minibatches,
            ), minibatch_loss_infos

        (
            (
                params_after_epochs,
                opt_states_after_epochs,
                key_after_epochs,
            ),
            train_metrics_after_epochs,
        ) = jax.lax.scan(_train_single_epoch, epoch_scan_init_state, epoch_indices)
        final_learner_state = PPOOnPolicyLearnerState(
            params_after_epochs,
            opt_states_after_epochs,
            key_after_epochs,
            env_state,
            timestep,
            algo_hps,
            actor_network_hps,
            critic_network_hps,
            norm_params,
            total_env_steps,
        )
        cycle_output = AnakinTrainOutput(
            learner_state=final_learner_state,
            episode_metrics=traj_batch.info,
            train_metrics=train_metrics_after_epochs,
        )
        return final_learner_state, cycle_output

    return _apply_single_update_cycle
