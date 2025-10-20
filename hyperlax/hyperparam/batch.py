import chex
import jax.numpy as jnp


class HyperparamBatch:
    """Generic wrapper for hyperparameter batches."""

    def __init__(
        self,
        data_values: jnp.ndarray,
        field_name_to_index: dict[str, int],
        field_names: list[str],
    ):
        self._data_values = data_values
        self._field_name_to_index = field_name_to_index
        self._field_names = field_names

    @property
    def data_values(self):
        return self._data_values

    @property
    def field_name_to_index(self):
        return self._field_name_to_index

    def __getattr__(self, name: str) -> jnp.ndarray:
        if name in self._field_name_to_index:
            return self._data_values[:, self._field_name_to_index[name]]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}' or it's not a registered field."
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data_values.shape

    def get_slice(self, indices: chex.Array) -> "HyperparamBatch":
        """Create a new HyperparamBatch containing only selected indices."""
        return HyperparamBatch(
            data_values=self._data_values[indices],
            field_name_to_index=self._field_name_to_index,
            field_names=self._field_names,
        )

    def to_array(self) -> jnp.ndarray:
        """Convert to raw array form."""
        return self._data_values

    def tree_flatten(self):
        children = (self._data_values,)
        aux_data = (self._field_name_to_index, self._field_names)
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[dict[str, int], list[str]],
        children: tuple[jnp.ndarray, ...],
    ):
        field_name_to_index, field_names = aux_data
        return cls(
            data_values=children[0],
            field_name_to_index=field_name_to_index,
            field_names=field_names,
        )
