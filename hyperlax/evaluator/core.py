import chex
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment

from hyperlax.base_types import ActFn, ActorApply, AnakinTrainOutput, EvalFn, EvalState
from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.normalizer.running_stats import (
    InitNormFn,
    NormalizeFn,
    RunningStatsMeanStd,
    UpdateNormFn,
)


def get_distribution_act_fn(
    config: BaseExperimentConfig,
    actor_apply: ActorApply,
    rngs: dict[str, chex.PRNGKey] | None = None,
) -> ActFn:
    """Get the act_fn for a network that returns a distribution."""

    def act_fn(params: FrozenDict, observation: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Get the action from the distribution."""
        if rngs is None:
            pi = actor_apply(params, observation)
        else:
            pi = actor_apply(params, observation, rngs=rngs)
        if config.training.evaluation_greedy:
            action = pi.mode()
        else:
            action = pi.sample(seed=key)
        return action

    return act_fn


## TODO
# def get_rnn_evaluator_fn(
#     env: Environment,
#     rec_act_fn: RecActFn,
#     config: BaseExperimentConfig,
#     scanned_rnn: nn.Module,
#     log_solve_rate: bool = False,
#     eval_multiplier: int = 1,
# ) -> EvalFn:
#     """Get the evaluator function for recurrent networks."""

#     def eval_one_episode(params: FrozenDict, init_eval_state: RNNEvalState) -> Dict:
#         """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

#         def _env_step(eval_state: RNNEvalState) -> RNNEvalState:
#             """Step the environment."""
#             (
#                 key,
#                 env_state,
#                 last_timestep,
#                 last_done,
#                 hstate,
#                 step_count,
#                 episode_return,
#             ) = eval_state

#             # PRNG keys.
#             key, policy_key = jax.random.split(key)

#             # Add a batch dimension and env dimension to the observation.
#             batched_observation = jax.tree_util.tree_map(
#                 lambda x: jnp.expand_dims(x, axis=0)[jnp.newaxis, :], last_timestep.observation
#             )
#             ac_in = (batched_observation, jnp.expand_dims(last_done, axis=0))

#             # Run the network.
#             hstate, action = rec_act_fn(params, hstate, ac_in, policy_key)

#             # Step environment.
#             env_state, timestep = env.step(env_state, action[-1].squeeze(0))

#             # Log episode metrics.
#             episode_return += timestep.reward
#             step_count += 1
#             eval_state = RNNEvalState(
#                 key,
#                 env_state,
#                 timestep,
#                 timestep.last().reshape(-1),
#                 hstate,
#                 step_count,
#                 episode_return,
#             )
#             return eval_state

#         def not_done(carry: Tuple) -> bool:
#             """Check if the episode is done."""
#             timestep = carry[2]
#             is_not_done: bool = ~timestep.last()
#             return is_not_done

#         final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

#         eval_metrics = {
#             "episode_return": final_state.episode_return,
#             "episode_length": final_state.step_count,
#         }
#         # Log solve episode if solve rate is required.
#         if log_solve_rate:
#             eval_metrics["solve_episode"] = jnp.all(
#                 final_state.episode_return >= config.env.solved_return_threshold
#             ).astype(int)
#         return eval_metrics


def get_ff_evaluator_fn_w_normalizer(
    env: Environment,
    act_fn: ActFn,
    normalizer_fns: tuple[InitNormFn, UpdateNormFn, NormalizeFn],
    config: BaseExperimentConfig,
    log_win_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    _, _, normalize_fn = normalizer_fns

    def eval_one_episode(
        params: FrozenDict,
        init_eval_state: EvalState,
        norm_params: RunningStatsMeanStd,
    ) -> dict:
        def _env_step(eval_state: EvalState) -> EvalState:
            key, env_state, last_timestep, step_count, episode_return = eval_state
            key, policy_key = jax.random.split(key)
            normed_obs = normalize_fn(norm_params, last_timestep.observation.agent_view)
            action = act_fn(
                params,
                jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], normed_obs),
                policy_key,
            )
            env_state, timestep = env.step(env_state, action.squeeze())
            episode_return += timestep.reward
            step_count += 1
            eval_state = EvalState(key, env_state, timestep, step_count, episode_return)
            return eval_state

        def not_done(carry: tuple) -> bool:
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)
        eval_metrics = {
            "episode_return": final_state.episode_return,
            "episode_length": final_state.step_count,
        }
        if log_win_rate:
            eval_metrics["won_episode"] = jnp.all(final_state.timestep.reward >= 1.0).astype(int)
        return eval_metrics

    def evaluator_fn(
        trained_params: FrozenDict,
        key: chex.PRNGKey,
        run_stat_params: RunningStatsMeanStd,
    ) -> AnakinTrainOutput[EvalState]:
        n_devices = len(jax.devices())
        if n_devices > config.training.num_eval_episodes:
            eval_batch = config.training.num_eval_episodes * eval_multiplier
        else:
            eval_batch = (config.training.num_eval_episodes // n_devices) * eval_multiplier

        if key.shape == (2,):
            # Single key (for the context of this specific (HP, Seed, Device) slice)
            key, *env_keys = jax.random.split(key, eval_batch + 1)
            env_states, timesteps = jax.vmap(env.reset)(
                jnp.stack(env_keys),
            )
            key, *step_keys = jax.random.split(key, eval_batch + 1)
            # Ensure step_keys is explicitly (eval_batch, 2)
            step_keys = jnp.stack(step_keys).reshape(eval_batch, 2)
        else:
            # Batched keys: shape (..., 2) - This path implies 'key' already contains the 'eval_batch' dimension
            # from an outer batching, which is generally not expected if 'transform_function_by_strategy'
            # is correctly stripping dimensions before passing to the core function.
            # However, to be robust if this path is hit:
            flat_keys = key.reshape(-1, 2)
            env_states, timesteps = jax.vmap(env.reset)(flat_keys)
            # The 'eval_batch' size is now the first dimension of 'key' here
            # (e.g., if key was (16,2), then current_eval_batch_size_from_key is 16).
            #current_eval_batch_size_from_key = key.shape[0] if len(key.shape) > 1 else 1

            # Reshape env_states/timesteps to match the original incoming key's leading dimensions
            # (e.g., if key was (16,2), env_states/timesteps leaves will be (16, ...) after vmap)
            # The reshape_fn here is for cases where 'key' might have more than one leading batch dim
            # (e.g., (HP,S,D,eval_batch,2)).
            # However, given the error, 'key' here seems to be just (eval_batch, 2).
            batch_shape_for_env_data = key.shape[:-1]

            def reshape_fn(x):
                return x.reshape(batch_shape_for_env_data + x.shape[1:])

            env_states = jax.tree_util.tree_map(reshape_fn, env_states)
            timesteps = jax.tree_util.tree_map(reshape_fn, timesteps)
            step_keys = key

        # The 'eval_batch' variable calculated at the beginning of evaluator_fn is the
        # actual batch size that the inner `jax.vmap` will operate on.
        # Explicitly use this size for initial states to ensure consistency.
        eval_state = EvalState(
            key=step_keys,
            env_state=env_states,
            timestep=timesteps,
            step_count=jnp.zeros(
                (eval_batch, 1), dtype=jnp.int32
            ),  # <-- FIXED LINE: Use eval_batch directly
            episode_return=jnp.zeros(
                (eval_batch,) + timesteps.reward.shape[1:], dtype=timesteps.reward.dtype
            ),  # <-- FIXED LINE: Use eval_batch directly
        )
        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0, None),
            axis_name="eval_batch",
        )(trained_params, eval_state, run_stat_params)
        return AnakinTrainOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
            train_metrics={},
        )

    return evaluator_fn
