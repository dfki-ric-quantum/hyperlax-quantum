# REF: we took from stoix and they took from rlax :)

import chex
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

# STOIX author note
# These losses are generally taken from rlax but edited to explicitly take in a batch of data.
# This is because the original rlax losses are not batched and are meant to be used with vmap,
# which is much slower.


def ppo_clip_loss_masked(
    pi_log_prob_t: chex.Array,
    b_pi_log_prob_t: chex.Array,
    gae_t: chex.Array,
    epsilon: float,
    mask: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array]:  # Returns (sum_loss, num_valid_samples)
    """PPO clipped surrogate objective. Returns sum of losses and valid sample count."""
    ratio = jnp.exp(pi_log_prob_t - b_pi_log_prob_t)
    loss_actor1_per_sample = ratio * gae_t
    loss_actor2_per_sample = (
        jnp.clip(
            ratio,
            1.0 - epsilon,
            1.0 + epsilon,
        )
        * gae_t
    )
    # PPO minimizes the negative of the objective, so we take -minimum here.
    # The objective is E[min(ratio * A, clip(ratio) * A)].
    # So, loss is -E[min(...)].
    # We want to sum the negative of the minimum objective over valid samples.
    loss_actor_per_sample = -jnp.minimum(loss_actor1_per_sample, loss_actor2_per_sample)

    if mask is not None:
        chex.assert_equal_shape((loss_actor_per_sample, mask))
        masked_loss_actor = loss_actor_per_sample * mask
        sum_loss_actor = jnp.sum(masked_loss_actor)
        num_valid_samples = jnp.sum(mask)
    else:
        sum_loss_actor = jnp.sum(loss_actor_per_sample)
        num_valid_samples = jnp.array(
            loss_actor_per_sample.size, dtype=jnp.float32
        )

    # Ensure num_valid_samples is float for potential division later by user
    num_valid_samples = num_valid_samples.astype(jnp.float32)

    return sum_loss_actor, num_valid_samples


def clipped_value_loss_masked(
    pred_value_t: chex.Array,
    behavior_value_t: chex.Array,  # This is v_old from the rollout
    targets_t: chex.Array,  # This is GAE targets (v_target)
    epsilon: float,
    mask: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array]:  # Returns (sum_loss, num_valid_samples)
    """Clipped value function loss. Returns sum of losses and valid sample count."""
    # Value prediction clipped to be within epsilon of behavior values.
    value_pred_clipped = behavior_value_t + jnp.clip(
        pred_value_t - behavior_value_t, -epsilon, epsilon
    )
    # Squared error losses.
    value_losses_unclipped_per_sample = jnp.square(pred_value_t - targets_t)
    value_losses_clipped_per_sample = jnp.square(value_pred_clipped - targets_t)
    # Combined loss.
    value_loss_per_sample = 0.5 * jnp.maximum(
        value_losses_unclipped_per_sample, value_losses_clipped_per_sample
    )

    if mask is not None:
        chex.assert_equal_shape((value_loss_per_sample, mask))
        masked_value_loss = value_loss_per_sample * mask
        sum_value_loss = jnp.sum(masked_value_loss)
        num_valid_samples = jnp.sum(mask)
    else:
        sum_value_loss = jnp.sum(value_loss_per_sample)
        num_valid_samples = jnp.array(value_loss_per_sample.size, dtype=jnp.float32)

    num_valid_samples = num_valid_samples.astype(jnp.float32)

    return sum_value_loss, num_valid_samples
