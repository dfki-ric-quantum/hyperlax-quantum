import functools
from collections.abc import Sequence

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# from hyperlax.types.common import Observation, RNNObservation
from hyperlax.base_types import Observation, RNNObservation
from hyperlax.network.inputs import ObservationInput
from hyperlax.network.utils import parse_rnn_cell


class FeedForwardActor(nn.Module):
    """Generic Feedforward Actor that can wrap parametric or non-parametric torsos."""

    action_head: nn.Module
    torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        observation: Observation,
        network_hps: chex.ArrayTree | None = None,
    ) -> distrax.DistributionLike:
        obs_embedding = self.input_layer(observation)

        if hasattr(self.torso, "max_depth"):
            batch_size = obs_embedding.shape[0]
            if network_hps is None:
                # Initialization path: provide dummy HPs
                dummy_num_layers = jnp.full((batch_size,), 2, dtype=jnp.int32)
                dummy_layer_widths = jnp.full(
                    (batch_size, self.torso.max_depth), 64, dtype=jnp.int32
                )
                dummy_activation_idx = jnp.full((batch_size,), 0, dtype=jnp.int32)
                dummy_use_layer_norm = jnp.full((batch_size,), False, dtype=jnp.bool_)
                obs_embedding = self.torso(
                    obs_embedding,
                    dummy_num_layers,
                    dummy_layer_widths,
                    dummy_activation_idx,
                    dummy_use_layer_norm,
                )
            else:
                # Runtime path: broadcast scalar HPs from the vmapped context
                layer_widths_for_torso = jnp.broadcast_to(
                    network_hps.width, (batch_size, self.torso.max_depth)
                )
                obs_embedding = self.torso(
                    obs_embedding,
                    num_layers=jnp.broadcast_to(network_hps.num_layers, (batch_size,)),
                    layer_widths=layer_widths_for_torso,
                    activation_idx=jnp.broadcast_to(network_hps.activation, (batch_size,)),
                    use_layer_norm=jnp.broadcast_to(network_hps.use_layer_norm, (batch_size,)),
                )
        else:
            # This is a standard, non-parametric torso.
            obs_embedding = self.torso(obs_embedding)

        return self.action_head(obs_embedding)


class FeedForwardCritic(nn.Module):
    """Generic Feedforward Critic that can wrap parametric or non-parametric torsos."""

    critic_head: nn.Module
    torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        observation: Observation,
        network_hps: chex.ArrayTree | None = None,
    ) -> chex.Array:
        obs_embedding = self.input_layer(observation)

        if hasattr(self.torso, "max_depth"):
            batch_size = obs_embedding.shape[0]
            if network_hps is None:
                # Initialization path: provide dummy HPs
                dummy_num_layers = jnp.full((batch_size,), 2, dtype=jnp.int32)
                dummy_layer_widths = jnp.full(
                    (batch_size, self.torso.max_depth), 64, dtype=jnp.int32
                )
                dummy_activation_idx = jnp.full((batch_size,), 0, dtype=jnp.int32)
                dummy_use_layer_norm = jnp.full((batch_size,), False, dtype=jnp.bool_)
                obs_embedding = self.torso(
                    obs_embedding,
                    dummy_num_layers,
                    dummy_layer_widths,
                    dummy_activation_idx,
                    dummy_use_layer_norm,
                )
            else:
                # Runtime path: broadcast scalar HPs from the vmapped context
                layer_widths_for_torso = jnp.broadcast_to(
                    network_hps.width, (batch_size, self.torso.max_depth)
                )
                obs_embedding = self.torso(
                    obs_embedding,
                    num_layers=jnp.broadcast_to(network_hps.num_layers, (batch_size,)),
                    layer_widths=layer_widths_for_torso,
                    activation_idx=jnp.broadcast_to(network_hps.activation, (batch_size,)),
                    use_layer_norm=jnp.broadcast_to(network_hps.use_layer_norm, (batch_size,)),
                )
        else:
            obs_embedding = self.torso(obs_embedding)

        critic_output = self.critic_head(obs_embedding)
        return critic_output


class CompositeArchitecture(nn.Module):
    """Composite Architecture. Takes in a sequence of layers and applies them sequentially."""

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self, *network_input: chex.Array | tuple[chex.Array, ...]
    ) -> distrax.DistributionLike | chex.Array:
        x = self.layers[0](*network_input)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class MultiNetwork(nn.Module):
    """Multi Network.

    Takes in a sequence of networks, applies them separately and concatenates the outputs.
    """

    networks: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self, *network_input: chex.Array | tuple[chex.Array, ...]
    ) -> distrax.DistributionLike | chex.Array:
        """Forward pass."""
        outputs = []
        for network in self.networks:
            outputs.append(network(*network_input))
        concatenated = jnp.stack(outputs, axis=-1)
        chex.assert_rank(concatenated, 2)
        return concatenated


class ScannedRNN(nn.Module):
    hidden_state_dim: int
    cell_type: str

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, rnn_state: chex.Array, x: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Applies the module."""
        ins, resets = x
        hidden_state_reset_fn = lambda reset_state, current_state: jnp.where(
            resets[:, np.newaxis],
            reset_state,
            current_state,
        )
        rnn_state = jax.tree_util.tree_map(
            hidden_state_reset_fn,
            self.initialize_carry(ins.shape[0]),
            rnn_state,
        )
        new_rnn_state, y = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)(
            rnn_state, ins
        )
        return new_rnn_state, y

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_state_dim))


class RecurrentActor(nn.Module):
    """Recurrent Actor Architecture."""

    action_head: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int
    cell_type: str
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> tuple[chex.Array, distrax.DistributionLike]:
        observation, done = observation_done

        observation = self.input_layer(observation)
        policy_embedding = self.pre_torso(observation)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN(self.hidden_state_dim, self.cell_type)(
            policy_hidden_state, policy_rnn_input
        )
        actor_logits = self.post_torso(policy_embedding)
        pi = self.action_head(actor_logits)

        return policy_hidden_state, pi


class RecurrentCritic(nn.Module):
    """Recurrent Critic Architecture."""

    critic_head: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int
    cell_type: str
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: tuple[chex.Array, chex.Array],
        observation_done: RNNObservation,
    ) -> tuple[chex.Array, chex.Array]:
        observation, done = observation_done

        observation = self.input_layer(observation)

        critic_embedding = self.pre_torso(observation)
        critic_rnn_input = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN(self.hidden_state_dim, self.cell_type)(
            critic_hidden_state, critic_rnn_input
        )
        critic_output = self.post_torso(critic_embedding)
        critic_output = self.critic_head(critic_output)

        return critic_hidden_state, critic_output
