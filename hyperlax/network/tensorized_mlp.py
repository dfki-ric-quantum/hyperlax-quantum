import math

import jax
import jax.numpy as jnp
import tensornetwork as tn
from flax import linen
from jax import random


class Perceptron(linen.Module):
    in_dim: int = 4
    out_dim: int = 4096
    activation: str = "relu"

    def setup(self):
        self.weights = self.param(
            "weights", linen.initializers.lecun_normal(), (self.in_dim, self.out_dim)
        )
        self.bias = self.param("bias", linen.initializers.zeros, (self.out_dim,))

    def __call__(self, inputs):
        output = jnp.dot(inputs, self.weights) + self.bias
        if self.activation == "relu":
            return linen.relu(output)
        elif self.activation == "sigmoid":
            return linen.sigmoid(output)
        elif self.activation == "tanh":
            return linen.tanh(output)
        elif self.activation == "silu":
            return linen.silu(output)
        elif self.activation == "none":
            return output
        else:
            raise ValueError(f"Unknown activation function {self.activation} for Dense")


class MPOLayer(linen.Module):
    in_dim: int = 4096
    out_dim: int = 4096
    bond_dim: int = 16
    num_nodes_to_decompose: int = 2
    activation: str = "relu"

    def setup(self):
        # TODO
        self.tensor_in_dim = int(math.ceil(self.in_dim ** (1 / self.num_nodes_to_decompose)))
        self.tensor_out_dim = int(math.ceil(self.out_dim ** (1 / self.num_nodes_to_decompose)))

        # Initialising the keys for JAX's randomness
        # TODO maybe custom initialization?
        # key = random.PRNGKey(0)
        # key_a, key_b, key_bias = random.split(key, 3)

        # Tensors of MPO
        params_mpo_list = []
        for i in range(self.num_nodes_to_decompose):
            if i == 0 or i == self.num_nodes_to_decompose - 1:
                # First and last node
                p = self.param(
                    f"mpo_node_{i}",
                    linen.initializers.normal(stddev=1.0),
                    (
                        self.tensor_in_dim,
                        self.bond_dim,
                        self.tensor_out_dim,
                    ),  # in, left/right leg, out
                )
                params_mpo_list.append(p)
            else:
                # Middle nodes
                p = self.param(
                    f"mpo_node_{i}",
                    linen.initializers.normal(stddev=1.0),
                    (
                        self.tensor_in_dim,
                        self.bond_dim,
                        self.bond_dim,
                        self.tensor_out_dim,
                    ),  # in, left leg, right leg, out
                )
                params_mpo_list.append(p)
            # params_mpo_list.append(p)

        self.params_mpo_list = params_mpo_list

        reshape_of_bias = [self.tensor_out_dim] * self.num_nodes_to_decompose
        self.params_bias = self.param("bias", linen.initializers.zeros, tuple(reshape_of_bias))

    def __call__(self, inputs):
        # NOTE currently we for-loops, which will slow down the jit compilation
        # TODO maybe use jax.lax.fori_loop but test if compiling is fast enough to ignore this
        def mpo_contract_fn(input_vec, params_mpo_list, params_bias):
            # Reshape to a matrix instead of a vector
            reshape_of_input_vec = [self.tensor_in_dim] * self.num_nodes_to_decompose
            input_vec = jnp.reshape(input_vec, tuple(reshape_of_input_vec))

            # Now we create the network
            x_node = tn.Node(
                input_vec,
                backend="jax",
            )

            tn_mpo_node_list = []
            for i, node_param in enumerate(params_mpo_list):
                if i == 0:
                    # First node
                    mpo_node = tn.Node(
                        node_param,
                        backend="jax",
                        axis_names=["in", "right", "out"],
                    )
                elif i == len(params_mpo_list) - 1:
                    # Last node
                    mpo_node = tn.Node(
                        node_param,
                        backend="jax",
                        axis_names=["in", "left", "out"],
                    )
                else:
                    mpo_node = tn.Node(
                        node_param,
                        backend="jax",
                        axis_names=["in", "left", "right", "out"],
                    )
                tn_mpo_node_list.append(mpo_node)

            # Connect the edges to build MPO tensor network

            connected_edges = []
            for k in range(1, len(tn_mpo_node_list)):
                edge_mpo_in_between = (
                    tn_mpo_node_list[k - 1]["right"] ^ tn_mpo_node_list[k]["left"]
                )
                connected_edges.append(edge_mpo_in_between)

            for k in range(len(tn_mpo_node_list)):
                edge_mpo_and_x = tn_mpo_node_list[k]["in"] ^ x_node[k]
                connected_edges.append(edge_mpo_and_x)

            # The TN should now look like this for a rank 2 MPO:
            #   |     |
            #   a --- b
            #    \   /
            #      x
            # The TN should now look like this for a rank 4 MPO:
            #  |   |   |   |
            #  a - b - c - d
            #  \   |   |   /
            #        x

            # g = to_graphviz(tn_mpo_node_list + [x_node])
            # g.render(filename='MPO', directory='./', format='pdf', cleanup=True, view=True)

            # NOTE for a now, we use the greedy algorithm to contract the TN
            # TODO maybe use a more sophisticated contraction algorithm
            # but for a small TN like this, greedy should be fine
            Wx = tn.contractors.greedy(
                tn_mpo_node_list + [x_node],
                output_edge_order=[tn_mpo_node["out"] for tn_mpo_node in tn_mpo_node_list],
            )

            return Wx.tensor + params_bias

        # Apply vmapped function to every item in input_vec i.e., to every batch.
        result = jax.vmap(mpo_contract_fn, in_axes=(0, None, None))(
            inputs, self.params_mpo_list, self.params_bias
        )

        result = jnp.reshape(result, (-1, self.out_dim))

        if self.activation == "relu":
            return linen.relu(result)
        elif self.activation == "sigmoid":
            return linen.sigmoid(result)
        elif self.activation == "tanh":
            return linen.tanh(result)
        elif self.activation == "silu":
            return linen.silu(result)
        elif self.activation == "none":
            return result
        else:
            raise ValueError(f"Unknown activation function {self.activation} for TNLayer")


def test_MPOLayer():
    # Create the TNLayer to test it
    tn_layer = MPOLayer(
        in_dim=4096, out_dim=16, bond_dim=4, num_nodes_to_decompose=2, activation="relu"
    )
    # note:
    # 4096**(1/4) = 8
    # 4096**(1/2) = 64
    params_tn = tn_layer.init(random.PRNGKey(0), jnp.ones((1, 4096)))
    y_tn = tn_layer.apply(params_tn, jnp.ones((1, 4096)))
    # test the output shape
    assert y_tn.shape == (1, 16), f"expected shape: (1, 16) but got {y_tn.shape}"
    # test the parameter shapes
    if tn_layer.num_nodes_to_decompose == 2:
        assert params_tn["params"]["mpo_node_0"].shape == (
            64,
            4,
            4,
        ), "expected shape: (64, 4, 4) but got {}".format(params_tn["params"]["mpo_node_0"].shape)
        assert params_tn["params"]["mpo_node_1"].shape == (
            64,
            4,
            4,
        ), "expected shape: (64, 4, 4) but got {}".format(params_tn["params"]["mpo_node_1"].shape)
    elif tn_layer.num_nodes_to_decompose == 4:
        assert params_tn["params"]["mpo_node_0"].shape == (
            8,
            4,
            2,
        ), "expected shape: (8, 4, 2) but got {}".format(params_tn["params"]["mpo_node_0"].shape)
        assert params_tn["params"]["mpo_node_1"].shape == (
            8,
            4,
            4,
            2,
        ), "expected shape: (8, 4, 4, 2) but got {}".format(
            params_tn["params"]["mpo_node_1"].shape
        )
    print("Test passed - MPOLayer")


# test_MPOLayer()

# Resources used to implement 'scan' for flax
# https://flax.readthedocs.io/en/latest/linen_intro.html#scan
# https://flax.readthedocs.io/en/latest/nnx/haiku_linen_vs_nnx.html#scan-over-layers
# In Linen, the definition of Block is a little different, __call__ will accept and return a second dummy input/output that in both cases will be None.
# In MLP, we will use nn.scan as in the previous example, but by setting split_rngs={'params': True} and variable_axes={'params': 0}
# we are telling nn.scan create different parameters for each step and slice the params collection along the first axis.
# pay attention to training argument in the __call__ method, it is used to pass the training state to the scan function,
# which is an ugly part of using linen API.
#
# NOTE NNX (new API) is more clean.


class TMLPwithMPO(linen.Module):
    in_dim: int = 4
    hidden_dim: int = 4096
    num_nodes_to_decompose: int = 2
    bond_dim: int = 8
    out_dim: int = 1
    activation: str = "relu"
    num_mpo_layers: int = 1

    @linen.compact
    def __call__(self, x, training: bool = True):
        x = Perceptron(self.in_dim, self.hidden_dim, self.activation, name="input_layer")(x)
        for i in range(self.num_mpo_layers):
            x = MPOLayer(
                in_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
                bond_dim=self.bond_dim,
                num_nodes_to_decompose=self.num_nodes_to_decompose,
                activation=self.activation,
                name=f"mpo_layer_{i}",
            )(x)
        return Perceptron(self.hidden_dim, self.out_dim, activation="none", name="output_layer")(x)
