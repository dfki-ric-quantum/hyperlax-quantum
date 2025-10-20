from math import ceil

# from qml_benchmarks.network_utils import chunk_vmapped_fn
import chex
import jax
import pennylane as qml
from flax import linen as nn
from jax import numpy as jnp
from sklearn.utils import gen_batches

# NOTE this throws an error as tf-probability is using 32 bit precision,
# maybe in even other places...
# either figure out a way to do this globally or just keep it default
# jax.config.update("jax_enable_x64", True)


def chunk_vmapped_fn(vmapped_fn, start, max_vmap):
    """
    Convert a vmapped function to an equivalent function that evaluates in chunks of size
    max_vmap. The behaviour of chunked_fn should be the same as vmapped_fn, but with a
    lower memory cost.

    The input vmapped_fn should have in_axes = (None, None, ..., 0,0,...,0)

    Args:
        vmapped (func): vmapped function with in_axes = (None, None, ..., 0,0,...,0)
        start (int): The index where the first 0 appears in in_axes
        max_vmap (int) The max chunk size with which to evaluate the function

    Returns:
        chunked version of the function
    """

    def chunked_fn(*args):
        batch_len = len(args[start])
        batch_slices = list(gen_batches(batch_len, max_vmap))
        res = [
            vmapped_fn(*args[:start], *[arg[slice] for arg in args[start:]])
            for slice in batch_slices
        ]
        # jnp.concatenate needs to act on arrays with the same shape, so pad the last array if necessary
        if batch_len / max_vmap % 1 != 0.0:
            diff = max_vmap - len(res[-1])
            res[-1] = jnp.pad(res[-1], [(0, diff), *[(0, 0)] * (len(res[-1].shape) - 1)])
            return jnp.concatenate(res)[:-diff]
        else:
            return jnp.concatenate(res)

    return chunked_fn


# class DataReuploadingClassifier(BaseEstimator, ClassifierMixin):
class DataReuploadingTorso(nn.Module):
    n_features: int
    n_layers: int = 4
    n_vstack: int = 2
    observable_type: str = "single"
    max_vmap: int = None
    jit: bool = True
    scaling: float = 1.0
    dev_type: str = "default.qubit.jax"
    draw_circuit: bool = False

    def setup(self):
        self.n_qubits_per_input_feature = ceil(self.n_features / 3)
        self.n_qubits_ = self.n_qubits_per_input_feature * self.n_vstack

        if self.observable_type == "single":
            self.observable_weight = 1
        elif self.observable_type == "half":
            self.observable_weight = max(1, self.n_qubits_ // 2)
        elif self.observable_type == "full":
            self.observable_weight = self.n_qubits_

        self.theta = self.param(
            "theta",
            jax.random.uniform,
            (self.n_qubits_, self.n_layers + 1, 3),
        )

        self.omega = self.param(
            "omega",
            jax.random.uniform,
            (self.n_layers + 1, self.n_qubits_ * 3),
        )

        self.params_ = {"thetas": self.theta, "omegas": self.omega}

        self.construct_network()

    def construct_network(self):
        # dev = qml.device(self.dev_type, wires=self.n_qubits_)
        dev = qml.device("default.qubit", wires=self.n_qubits_)

        @qml.qnode(dev, interface="jax")
        def circuit(params, x):
            for layer in range(self.n_layers):
                for vs_idx in range(self.n_vstack):
                    q_start_idx = vs_idx * self.n_qubits_per_input_feature
                    for f_idx in range(self.n_qubits_per_input_feature):
                        q_idx = q_start_idx + f_idx
                        input_slice = x[f_idx * 3 : (f_idx + 1) * 3]
                        if len(input_slice) < 3:
                            input_slice = jnp.pad(input_slice, (0, 3 - len(input_slice)))
                        angles = input_slice * params["omegas"][layer, q_idx * 3 : (q_idx + 1) * 3]
                        qml.Rot(*angles, wires=q_idx)

                        angles = params["thetas"][q_idx, layer, :]
                        qml.Rot(*angles, wires=q_idx)

                if layer % 2 == 0:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double")
                else:
                    qml.broadcast(qml.CZ, range(self.n_qubits_), pattern="double_odd")

            for vs_idx in range(self.n_vstack):
                q_start_idx = vs_idx * self.n_qubits_per_input_feature
                for f_idx in range(self.n_qubits_per_input_feature):
                    q_idx = q_start_idx + f_idx
                    input_slice = x[f_idx * 3 : (f_idx + 1) * 3]
                    if len(input_slice) < 3:
                        input_slice = jnp.pad(input_slice, (0, 3 - len(input_slice)))
                    angles = (
                        input_slice * params["omegas"][self.n_layers, q_idx * 3 : (q_idx + 1) * 3]
                        + params["thetas"][q_idx, self.n_layers, :]
                    )
                    qml.Rot(*angles, wires=q_idx)

            return [qml.expval(qml.PauliZ(wires=[i])) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def circuit_as_array(params, x):
            return jnp.array(circuit(params, x))

        if self.jit:
            circuit_as_array = jax.jit(circuit_as_array)
        self.forward = jax.vmap(circuit_as_array, in_axes=(None, 0))

        return self.forward

    def transform(self, X, preprocess=False):
        X = X * self.scaling
        # Ensure X has at least 3 columns
        if X.shape[1] < 3:
            X = jnp.pad(X, ((0, 0), (0, 3 - X.shape[1])), "constant")
        return X

    def __call__(self, observation: chex.Array) -> chex.Array:
        x = self.transform(observation)
        return self.forward(self.params_, x)


# TODO currently not happy with implementation but works :/
# 1. padding done in transform doesn't work and is needed inside the circuit function. so it needs more work
# 2. perf: we can squeeze more


# unit test for DataReuploadingTorso
# Updated test function
def test_DataReuploadingTorso():
    n_features = 11  # Changed to 11 features
    n_layers = 3
    n_vstack = 2
    observable_type = "single"
    jit = False
    scaling = 1.0
    dev_type = "default.qubit.jax"

    network = DataReuploadingTorso(
        n_features=n_features,
        n_layers=n_layers,
        n_vstack=n_vstack,
        observable_type=observable_type,
        jit=jit,
        scaling=scaling,
        dev_type=dev_type,
        draw_circuit=True,
    )

    rng = jax.random.PRNGKey(0)
    # Create dummy observation with 11 features
    dummy_observation = jnp.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        ]
    )

    params = network.init(rng, dummy_observation)

    # print("Architecture Configuration:")
    # print(f"Number of features: {n_features}")
    # print(f"Number of qubits: {network.n_qubits_}")
    # print(f"Qubits per input feature: {network.n_qubits_per_input_feature}")
    # print(f"Number of vertical stacks: {network.n_vstack}")

    print("\nParameter Shapes:")
    print("Theta shape:", params["params"]["theta"].shape)
    print("Omega shape:", params["params"]["omega"].shape)

    # Forward/apply the network
    result = network.apply(params, dummy_observation)

    print("\nArchitecture Output:")
    print("Result:", result)
    print("Result shape:", result.shape)


# Run the test
# test_DataReuploadingTorso()
