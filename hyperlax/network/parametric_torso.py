import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

# Maps activation function names to integer indices for vectorization.
ACTIVATION_FN_TO_IDX = {
    "relu": 0,
    "tanh": 1,
    "silu": 2,
    "linear": 3,  # Identity function for testing
}
IDX_TO_ACTIVATION_FN = {v: k for k, v in ACTIVATION_FN_TO_IDX.items()}


def _apply_activation_by_index(index: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized activation function application.
    For each sample in the batch `x`, this function applies the activation
    function specified by the corresponding integer in `index`. This is the
    key mechanism for handling vectorized activation hyperparameters.
    """

    def _single_apply(i, single_x):
        # jax.lax.switch is JIT-compatible control flow. It selects a function
        # from the list based on the integer index `i`.
        return jax.lax.switch(i, [nn.relu, nn.tanh, nn.silu, lambda y: y], single_x)

    # vmap this selection logic over the batch dimension.
    return jax.vmap(_single_apply)(index, x)


class ParametricMLPTorso(nn.Module):
    """
    A parametric MLP torso that supports vectorized architectures using jax.lax.scan.

    This "super-network" defines the largest possible architecture (max_depth, max_width)
    at compile time. The actual architecture for each sample in a batch is determined
    at runtime by input arrays, enabling parallelism over heterogeneous models.
    """

    max_depth: int
    max_width: int
    input_dim: int

    def setup(self):
        """
        Initializes the parameters for the largest possible network.
        Key Concept: Instead of a Python list of Flax Modules, we create stacked
        JAX arrays of parameters. This is essential for compatibility with
        jax.lax.scan, as we can index these arrays with a JAX tracer.
        """
        kernel_init_fn = nn.initializers.lecun_normal()
        bias_init_fn = nn.initializers.zeros

        # Parameters for the first layer, which has a unique input dimension.
        self.kernel_layer0 = self.param(
            "kernel_0", kernel_init_fn, (self.input_dim, self.max_width)
        )
        self.bias_layer0 = self.param("bias_0", bias_init_fn, (self.max_width,))

        # Stacked parameters for all subsequent layers (from layer 1 to max_depth-1).
        # The leading dimension corresponds to the layer index.
        self.kernels_rest = self.param(
            "kernels_rest",
            kernel_init_fn,
            (self.max_depth - 1, self.max_width, self.max_width),
        )
        self.biases_rest = self.param(
            "biases_rest", bias_init_fn, (self.max_depth - 1, self.max_width)
        )

        # Stacked parameters for LayerNorm for all layers.
        self.ln_scales = self.param(
            "ln_scales", nn.initializers.ones, (self.max_depth, self.max_width)
        )
        self.ln_biases = self.param(
            "ln_biases", nn.initializers.zeros, (self.max_depth, self.max_width)
        )

    def __call__(
        self,
        x: chex.Array,  # Input data, shape: (batch, input_dim)
        # Vectorized Hyperparameters:
        num_layers: chex.Array,  # Number of layers for each sample, shape: (batch,)
        layer_widths: chex.Array,  # Width for each layer per sample, shape: (batch, max_depth)
        activation_idx: chex.Array,  # Activation index for each sample, shape: (batch,)
        use_layer_norm: chex.Array,  # LayerNorm flag for each sample, shape: (batch,)
    ) -> chex.Array:
        batch_size = x.shape[0]

        # --- 1. First Layer (Handled separately due to unique input shape) ---
        # This avoids shape mismatch errors inside the jax.lax.scan loop.
        z = x @ self.kernel_layer0 + self.bias_layer0

        # Manually apply LayerNorm for layer 0, conditioned on the `use_layer_norm` flag.
        def apply_ln(operand, scale, bias):
            mean = operand.mean(axis=-1, keepdims=True)
            var = operand.var(axis=-1, keepdims=True)
            return (operand - mean) / jnp.sqrt(var + 1e-5) * scale + bias

        z = jnp.where(
            use_layer_norm[:, None],
            apply_ln(z, self.ln_scales[0], self.ln_biases[0]),
            z,
        )

        # Apply the vectorized activation function.
        z = _apply_activation_by_index(activation_idx, z)

        # Key Concept: Width Masking. Zero out units beyond the specified width for each sample.
        width_mask = jnp.arange(self.max_width) < layer_widths[:, 0, None]
        z = jnp.where(width_mask, z, 0.0)

        # Key Concept: Depth Masking. The output of the first layer is only used if a sample's
        # `num_layers` is greater than 0. Otherwise, its state is zeroed.
        layer_mask = 0 < num_layers[:, None]
        z_0 = jnp.where(layer_mask, z, jnp.zeros_like(z))

        # --- 2. Subsequent Layers (Processed with a jax.lax.scan loop) ---
        def body_fn(z_carry, i):
            # `i` is the loop counter (a JAX tracer) from 0 to max_depth-2.
            # The actual layer index is `i + 1`.
            layer_idx = i + 1

            # Perform the dense operation using parameters for the current layer.
            z_new = z_carry @ self.kernels_rest[i] + self.biases_rest[i]

            # Apply LayerNorm and Activation, same as above.
            z_new = jnp.where(
                use_layer_norm[:, None],
                apply_ln(z_new, self.ln_scales[layer_idx], self.ln_biases[layer_idx]),
                z_new,
            )
            z_new = _apply_activation_by_index(activation_idx, z_new)

            # Apply width masking for the current layer.
            width_mask_i = jnp.arange(self.max_width) < layer_widths[:, layer_idx, None]
            z_new = jnp.where(width_mask_i, z_new, 0.0)

            # Apply depth masking: if this layer is active for the sample, use z_new.
            # Otherwise, carry the previous state `z_carry` forward.
            layer_mask_i = layer_idx < num_layers[:, None]
            z_out = jnp.where(layer_mask_i, z_new, z_carry)

            # The scan body returns the next state and an output to be collected.
            return z_out, z_out

        # The initial state for the scan is the output of the first layer.
        # The loop runs for the remaining `max_depth - 1` layers.
        _, outputs_rest = jax.lax.scan(body_fn, z_0, jnp.arange(self.max_depth - 1))

        # --- 3. Gather Final Outputs ---
        # Combine the first layer's output with the outputs from the scan.
        all_outputs = jnp.concatenate([z_0[None, ...], outputs_rest], axis=0)
        # Transpose from (depth, batch, width) to (batch, depth, width) for easier indexing.
        all_outputs = jnp.transpose(all_outputs, (1, 0, 2))

        # Key Concept: Final Output Selection. For each sample, we select the hidden state
        # from its specified final layer. This is a highly efficient parallel lookup.
        last_layer_indices = jnp.clip(num_layers - 1, 0, self.max_depth - 1)
        final_output = all_outputs[jnp.arange(batch_size), last_layer_indices, :]

        return final_output
