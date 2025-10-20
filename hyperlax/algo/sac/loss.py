import chex
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from hyperlax.algo.sac.struct_sac import SACTransition, SACVectorizedHyperparams
from hyperlax.base_types import ActorApply, ContinuousQApply


def _masked_mean(x: chex.Array, mask: chex.Array | None) -> chex.Array:
    """Computes a masked mean, being careful about the denominator."""
    if mask is None:
        return jnp.mean(x)

    # Ensure mask is broadcastable to x
    if mask.ndim < x.ndim:
        mask = jnp.expand_dims(mask, axis=-1)

    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1e-8)


def _alpha_loss_fn(
    log_alpha: chex.Array,
    actor_params: FrozenDict,
    transitions: SACTransition,
    hyperparams: SACVectorizedHyperparams,
    actor_apply_fn: ActorApply,
    key: chex.PRNGKey,
    mask: chex.Array | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Computes the loss for the entropy temperature parameter alpha.
    Note: target_entropy is dynamically calculated here.
    """
    actor_policy = actor_apply_fn(actor_params, transitions.obs)
    action_dim = actor_policy.event_shape[-1]
    target_entropy = -hyperparams.target_entropy_scale * action_dim
    action = actor_policy.sample(seed=key)
    log_prob = actor_policy.log_prob(action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)

    mean_alpha_loss = _masked_mean(alpha_loss, mask)

    loss_info = {
        "alpha_loss": mean_alpha_loss,
        "alpha": _masked_mean(alpha, mask),
    }
    return mean_alpha_loss, loss_info


def _q_loss_fn(
    q_params: FrozenDict,
    target_q_params: FrozenDict,
    actor_params: FrozenDict,
    alpha: chex.Array,
    transitions: SACTransition,
    hyperparams: SACVectorizedHyperparams,
    q_apply_fn: ContinuousQApply,
    actor_apply_fn: ActorApply,
    key: chex.PRNGKey,
    mask: chex.Array | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Computes the soft Q-learning loss."""
    q_old_action = q_apply_fn(q_params, transitions.obs, transitions.action)
    next_actor_policy = actor_apply_fn(actor_params, transitions.next_obs)
    next_action = next_actor_policy.sample(seed=key)
    next_log_prob = next_actor_policy.log_prob(next_action)

    next_q = q_apply_fn(target_q_params, transitions.next_obs, next_action)
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob

    target_q = jax.lax.stop_gradient(
        transitions.reward + (1.0 - transitions.done) * hyperparams.gamma * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)
    q_loss = 0.5 * jnp.square(q_error)

    mean_q_loss = _masked_mean(q_loss, mask)

    jax.debug.print(
        "[SAC_LOSS_Q] "
        "q_loss_mean: {qlm} | "
        "q_error_mean: {qem} | "
        "q1_pred_mean: {q1m} | "
        "q2_pred_mean: {q2m}",
        qlm=mean_q_loss,
        qem=_masked_mean(jnp.abs(q_error), mask),
        q1m=_masked_mean(next_q[..., 0], mask),
        q2m=_masked_mean(next_q[..., 1], mask),
    )

    loss_info = {
        "q_loss": mean_q_loss,
        "q_error": _masked_mean(jnp.abs(q_error), mask),
        "q1_pred": _masked_mean(next_q[..., 0], mask),
        "q2_pred": _masked_mean(next_q[..., 1], mask),
    }
    return mean_q_loss, loss_info


def _actor_loss_fn(
    actor_params: FrozenDict,
    q_params: FrozenDict,
    alpha: chex.Array,
    transitions: SACTransition,
    actor_apply_fn: ActorApply,
    q_apply_fn: ContinuousQApply,
    key: chex.PRNGKey,
    mask: chex.Array | None = None,
) -> tuple[jnp.ndarray, dict]:
    """Computes the actor loss."""
    actor_policy = actor_apply_fn(actor_params, transitions.obs)
    action = actor_policy.sample(seed=key)
    log_prob = actor_policy.log_prob(action)
    q_action = q_apply_fn(q_params, transitions.obs, action)
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = alpha * log_prob - min_q

    mean_actor_loss = _masked_mean(actor_loss, mask)

    jax.debug.print(
        "[SAC_LOSS_ACTOR] actor_loss_mean: {alm} | entropy_mean: {em}",
        alm=mean_actor_loss,
        em=_masked_mean(-log_prob, mask),
    )

    loss_info = {
        "actor_loss": mean_actor_loss,
        "entropy": _masked_mean(-log_prob, mask),
    }
    return mean_actor_loss, loss_info
