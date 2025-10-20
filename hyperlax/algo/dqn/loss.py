import chex
import jax
import jax.numpy as jnp


def huber_loss(x: chex.Array, delta: float = 1.0) -> chex.Array:
    """Huber loss implementation."""
    chex.assert_type(x, float)
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear


def l2_loss(delta: jnp.ndarray) -> jnp.ndarray:
    """Standard L2 loss function."""
    return 0.5 * jnp.square(delta)


def q_learning_loss(
    q_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t: chex.Array,
    huber_loss_parameter: chex.Array,
    mask: chex.Array | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Standard Q-learning loss with masking. Returns sum of losses and aux info."""
    jax.debug.print(
        "[DQN_LOSS] q_tm1 shape: {s}, a_tm1 shape: {s2}, r_t shape: {s3}, d_t shape: {s4}, q_t shape: {s5}, huber_param: {h}, mask_sum: {m_s}",
        s=q_tm1.shape,
        s2=a_tm1.shape,
        s3=r_t.shape,
        s4=d_t.shape,
        s5=q_t.shape,
        h=huber_loss_parameter,
        m_s=jnp.sum(mask) if mask is not None else -1,
    )
    batch_indices = jnp.arange(a_tm1.shape[0])
    target_tm1 = r_t + d_t * jnp.max(q_t, axis=-1)
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]

    jax.debug.print(
        "[DQN_LOSS] td_error shape: {s}, td_error[:5]: {td}",
        s=td_error.shape,
        td=td_error[:5],
    )

    use_huber = huber_loss_parameter > 1e-6  # Use a small epsilon for float comparison
    batch_loss = jax.lax.cond(
        use_huber,
        lambda he_param: huber_loss(td_error, he_param),
        lambda _: l2_loss(td_error),
        huber_loss_parameter,
    )

    jax.debug.print(
        "[DQN_LOSS_DOUBLE] batch_loss (before mask) shape: {s}, batch_loss[:5]: {bl}",
        s=batch_loss.shape,
        bl=batch_loss[:5],
    )

    jax.debug.print(
        "[DQN_LOSS] batch_loss (before mask) shape: {s}, batch_loss[:5]: {bl}",
        s=batch_loss.shape,
        bl=batch_loss[:5],
    )

    loss_sum_val = jnp.sum(batch_loss)
    num_samples_val = jnp.array(batch_loss.shape[0], dtype=jnp.float32)

    if mask is not None:
        mask_float = mask.astype(jnp.float32)
        masked_batch_loss = batch_loss * mask_float
        loss_sum_val = jnp.sum(masked_batch_loss)
        num_samples_val = jnp.sum(mask_float)

    jax.debug.print(
        "[DQN_LOSS] loss_sum_val: {lsv}, num_samples_val: {nsv}",
        lsv=loss_sum_val,
        nsv=num_samples_val,
    )

    return loss_sum_val, {
        "q_loss_sum": loss_sum_val,
        "valid_samples": num_samples_val,
        "mean_td_error": jnp.mean(td_error),
        "mean_abs_td_error": jnp.mean(jnp.abs(td_error)),
    }


def double_q_learning_loss(
    q_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t_target_net: chex.Array,
    q_t_online_net_selector: chex.Array,
    huber_loss_parameter: chex.Array,
    mask: chex.Array | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Double Q-learning loss with masking. Returns sum of losses and aux info."""
    jax.debug.print(
        "[DQN_LOSS_DOUBLE] q_tm1 shape: {s}, a_tm1 shape: {s2}, r_t shape: {s3}, d_t shape: {s4}, q_t_target shape: {s5}, q_t_online shape: {s6}, huber_param: {h}, mask_sum: {m_s}",
        s=q_tm1.shape,
        s2=a_tm1.shape,
        s3=r_t.shape,
        s4=d_t.shape,
        s5=q_t_target_net.shape,
        s6=q_t_online_net_selector.shape,
        h=huber_loss_parameter,
        m_s=jnp.sum(mask) if mask is not None else -1,
    )
    batch_indices = jnp.arange(a_tm1.shape[0])
    next_action = jnp.argmax(q_t_online_net_selector, axis=-1)
    next_q_value = q_t_target_net[batch_indices, next_action]
    target_tm1 = r_t + d_t * next_q_value
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]

    jax.debug.print(
        "[DQN_LOSS_DOUBLE] next_action[:5]: {na}, td_error shape: {s}, td_error[:5]: {td}",
        na=next_action[:5],
        s=td_error.shape,
        td=td_error[:5],
    )

    use_huber = huber_loss_parameter > 1e-6  # Use a small epsilon
    batch_loss = jax.lax.cond(
        use_huber,
        lambda he_param: huber_loss(td_error, he_param),
        lambda _: l2_loss(td_error),
        huber_loss_parameter,
    )

    loss_sum_val = jnp.sum(batch_loss)
    num_samples_val = jnp.array(batch_loss.shape[0], dtype=jnp.float32)

    if mask is not None:
        mask_float = mask.astype(jnp.float32)
        masked_batch_loss = batch_loss * mask_float
        loss_sum_val = jnp.sum(masked_batch_loss)
        num_samples_val = jnp.sum(mask_float)

    jax.debug.print(
        "[DQN_LOSS_DOUBLE] loss_sum_val: {lsv}, num_samples_val: {nsv}",
        lsv=loss_sum_val,
        nsv=num_samples_val,
    )

    return loss_sum_val, {
        "q_loss_sum": loss_sum_val,
        "valid_samples": num_samples_val,
        "mean_td_error": jnp.mean(td_error),
        "mean_abs_td_error": jnp.mean(jnp.abs(td_error)),
        "next_action_sample": next_action[:5],
    }
