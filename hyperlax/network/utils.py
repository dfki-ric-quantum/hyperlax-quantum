import dataclasses
import importlib
import inspect
from collections.abc import Callable
from typing import Any

import chex
from flax import linen as nn

from hyperlax.hyperparam.tunable import Tunable


def parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "silu": nn.silu,
        "elu": nn.elu,
        "gelu": nn.gelu,
    }
    return activation_fns[activation_fn_name]


def parse_rnn_cell(rnn_cell_name: str) -> nn.RNNCellBase:
    """Get the rnn cell."""
    rnn_cells: dict[str, Callable[[chex.Array], chex.Array]] = {
        "lstm": nn.LSTMCell,
        "optimised_lstm": nn.OptimizedLSTMCell,
        "gru": nn.GRUCell,
        "mgu": nn.MGUCell,
        "simple": nn.SimpleCell,
    }
    return rnn_cells[rnn_cell_name]


def instantiate_from_config(config_obj: Any, **additional_kwargs: Any) -> Any:
    """
    Instantiates an object from a configuration dataclass.

    The config_obj is expected to have a =_target_= attribute
    specifying the full Python path to the class to be instantiated.
    Other attributes of config_obj are passed as keyword arguments
    to the class constructor. =additional_kwargs= can be used to
    provide or override arguments. It intelligently filters out parameters
    from the config that are not part of the target class's constructor signature.
    """
    if not dataclasses.is_dataclass(config_obj) or not hasattr(config_obj, "_target_"):
        raise ValueError("config_obj must be a dataclass and have a '_target_' attribute.")

    target_path = config_obj._target_
    try:
        module_path, class_name = target_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        target_class = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import target '{target_path}': {e}")

    config_params = {}
    for f in dataclasses.fields(config_obj):
        if f.name != "_target_":
            field_value = getattr(config_obj, f.name)
            # If the field is a Tunable object, extract its value.
            # Otherwise, use the field value as is.
            if isinstance(field_value, Tunable):
                config_params[f.name] = field_value.value
            else:
                config_params[f.name] = field_value

    final_params = {**config_params, **additional_kwargs}

    # Filter parameters to match the target class's constructor signature.
    # This is a way to handle proxy parameters (like 'arch_choice') that are in the
    # config for sampling but are not actual arguments for the network class.
    sig = inspect.signature(target_class)
    valid_param_names = set(sig.parameters.keys())
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    if has_kwargs:
        # If the class constructor accepts **kwargs, pass all parameters.
        filtered_params = final_params
    else:
        # Otherwise, only pass parameters that are explicitly in the constructor's signature.
        filtered_params = {
            key: value for key, value in final_params.items() if key in valid_param_names
        }

    return target_class(**filtered_params)
