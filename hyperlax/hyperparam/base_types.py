import logging
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from hyperlax.hyperparam.distributions import BaseDistribution
from hyperlax.hyperparam.tunable import Tunable

logger = logging.getLogger(__name__)


def flatten_spec_container(
    spec_container: "HyperparamSpecContainer",
) -> dict[str, "HyperparamSpec"]:
    """
    Recursively traverses a HyperparamSpecContainer and creates a flat dictionary
    mapping the full, unique path to each HyperparamSpec.

    Args:
        spec_container: The top-level HyperparamSpecContainer instance.

    Returns:
        A dictionary where keys are dot-separated paths (e.g., "network_mlp.critic_network.pre_torso.layer_sizes")
        and values are the corresponding HyperparamSpec objects.
    """
    flat_map = {}

    def _recursive_flatten(obj: Any, current_path: str) -> None:
        if not is_dataclass(obj):
            return
        for field in fields(obj):
            field_name = field.name
            if field_name.startswith("_"):  # Skip private-like attributes
                continue
            field_value = getattr(obj, field_name)
            new_path = f"{current_path}.{field_name}" if current_path else field_name

            if isinstance(field_value, HyperparamSpecContainer):
                _recursive_flatten(field_value, new_path)
            elif isinstance(field_value, HyperparamSpec):
                flat_map[new_path] = field_value

    _recursive_flatten(spec_container, "")
    return flat_map


@dataclass(frozen=True)
class HyperparamSpec:
    """Definition of a single hyperparameter."""

    default_value: Any
    is_vectorized: bool
    is_fixed: bool
    expected_type: type = Any  # TODO rm Any to make it hard requirement
    distribution: BaseDistribution | None = None

    def __post_init__(self):
        if not self.is_fixed and self.distribution is None:
            # Allow missing distribution if loading from file
            # raise ValueError(f"Distribution required for variable parameter {self.dict_key_name}")
            pass  # Distribution not strictly required if loading from file
        if self.is_fixed and self.distribution is not None:
            raise ValueError("Fixed parameter cannot have distribution")


@dataclass(frozen=True)
class HyperparamBatchGroup:
    """Container for grouped hyperparameter batches."""

    non_vec_values: dict[str, Any]
    vec_batches: list[dict[str, Any]]
    default_values: dict[str, Any]
    sample_ids: list[int]


@dataclass(frozen=True)
class HyperparamSpecContainer:
    """
    Base for managing algorithm's hyperparameters.

    Provides operations for hyperparameter management with type safety
    and immutability guarantees.

    Properties:
        - Pure functional transformations: All operations return new objects
          instead of modifying existing ones
          Example:
          ```python
          # Returns new dict without modifying self
          vectorized_params = self.filter_by_vectorization(is_vectorized=True)
          ```

        - Parameter grouping stays consistent:
          Example:
          ```python
          # Same non-vectorized params always group together
          group1 = {"network_arch": [32, 32]}
          group2 = {"network_arch": [64, 64]}
          # Batches are grouped by these values
          ```

        - Fixed and sampled parameters are separate:
          Example:
          ```python
          # A parameter cannot be both fixed and sampled
          tau = HParamDefinition(
              name="tau",
              default_value=0.005,
              is_vectorized=False,
              is_fixed=True  # No distribution needed
          )

          learning_rate = HParamDefinition(
              name="learning_rate",
              default_value=0.001,
              is_vectorized=True,
              is_fixed=False,  # Requires distribution
              distribution=LogUniform(domain=(1e-4, 1e-2))
          )
          ```

        - Type safety in batch operations:
          ```python
          def group_batch_samples(self, full_batch_sample: Dict[str, List[Any]]):
              # Validates batch structure
              if len({len(values) for values in full_batch_sample.values()}) != 1:
                  raise ValueError("Inconsistent batch sizes across parameters")
          ```

        - Deterministic batch generation (we assume that sampling is already done, we are just grouping):
          ```python
          # Same input always produces same output
          batch1 = self.group_batch_samples(sample_data)
          batch2 = self.group_batch_samples(sample_data)
          assert batch1 == batch2
          ```
    Usage:
        Create algorithm-specific hyperparameters:
        ```python
        @dataclass
        class DQNHParams(HParamManager):
            learning_rate: HParamDefinition = field(
                default_factory=lambda: HParamDefinition(
                    name="learning_rate",
                    default_value=0.001,
                    is_vectorized=True,
                    is_fixed=False,
                    distribution=LogUniform(domain=(1e-4, 1e-2))
                )
            )
        ```
    """

    def filter_by_vectorization(self, *, is_vectorized: bool) -> dict[str, HyperparamSpec]:
        """
        Filters all HyperparamSpec objects in the container based on their `is_vectorized` attribute.
        This is now a one-liner using the flatten utility.
        """
        flat_map = flatten_spec_container(self)
        return {
            path: spec for path, spec in flat_map.items() if spec.is_vectorized == is_vectorized
        }

    def filter_by_sampling(self, *, is_fixed: bool) -> dict[str, HyperparamSpec]:
        """
        Filters all HyperparamSpec objects in the container based on their `is_fixed` attribute.
        """
        flat_map = flatten_spec_container(self)
        return {path: spec for path, spec in flat_map.items() if spec.is_fixed == is_fixed}

    def get_sampling_config(self) -> dict[str, BaseDistribution]:
        variable_params = self.filter_by_sampling(is_fixed=False)
        return {
            name: param.distribution
            for name, param in variable_params.items()
            if param.distribution is not None
        }

    def get_default_values(self) -> dict[str, Any]:
        """Recursively gets all default values, keyed by their full path."""
        flat_map = flatten_spec_container(self)
        return {path: spec.default_value for path, spec in flat_map.items()}

    def get_ordered_vectorized_keys(self) -> list[str]:
        """Recursively gets all keys for vectorized parameters."""
        vectorized_params = self.filter_by_vectorization(is_vectorized=True)
        return list(vectorized_params.keys())
        # return sorted(list(vectorized_params.keys())) # Sort for deterministic order

    def get_spec_by_dict_key_name(self, dict_key_name: str) -> HyperparamSpec | None:
        """Recursively finds a HyperparamSpec by its full dict_key_name."""
        return flatten_spec_container(self).get(dict_key_name)

    def group_batch_samples(
        self, full_batch_sample: dict[str, list[Any]]
    ) -> list[HyperparamBatchGroup]:
        """Group samples by non-vectorized parameter combinations."""
        if not full_batch_sample:
            return []

        vectorized_keys = set(self.filter_by_vectorization(is_vectorized=True).keys())
        non_vectorized_keys = set(self.filter_by_vectorization(is_vectorized=False).keys())
        all_default_values = self.get_default_values()
        sampled_keys = set(full_batch_sample.keys())

        if not sampled_keys:
            return []
        batch_size = len(full_batch_sample[next(iter(sampled_keys))])

        if "sample_id" not in full_batch_sample:
            raise ValueError("Missing 'sample_id' needed for tracking.")

        groups = defaultdict(lambda: {"vec_batches": [], "sample_ids": []})

        for idx in range(batch_size):
            # This sample's values for non-vectorized keys. These values form the group key.
            current_non_vec_values = {
                key: full_batch_sample[key][idx]
                for key in non_vectorized_keys
                if key in sampled_keys
            }

            def make_hashable(val: Any) -> Any:
                if isinstance(val, list):
                    return tuple(val)
                return val

            hashable_items = {k: make_hashable(v) for k, v in current_non_vec_values.items()}
            non_vec_tuple = tuple(sorted(hashable_items.items()))

            # This sample's values for vectorized keys.
            current_vec_values = {
                key: full_batch_sample[key][idx] for key in vectorized_keys if key in sampled_keys
            }
            sample_id_val = int(full_batch_sample["sample_id"][idx])

            groups[non_vec_tuple]["vec_batches"].append(current_vec_values)
            groups[non_vec_tuple]["sample_ids"].append(sample_id_val)

        # Get fixed parameters that were NEVER part of sampling.
        # These are defaults for keys in the spec that are NOT in the sample file.
        fixed_defaults = {k: v for k, v in all_default_values.items() if k not in sampled_keys}

        return [
            HyperparamBatchGroup(
                non_vec_values=dict(non_vec_tuple),
                vec_batches=group_data["vec_batches"],
                sample_ids=group_data["sample_ids"],
                default_values=fixed_defaults,
            )
            for non_vec_tuple, group_data in groups.items()
        ]


def flatten_tunables(config_obj: Any) -> dict[str, "Tunable"]:
    """
    Recursively traverses a config dataclass and creates a flat dictionary
    mapping the relative path to each Tunable object.
    """
    flat_map = {}

    def _recursive_flatten(obj: Any, current_path: str) -> None:
        if not is_dataclass(obj):
            return
        for field in fields(obj):
            field_name = field.name
            if field_name.startswith("_"):
                continue
            field_value = getattr(obj, field_name)
            new_path = f"{current_path}.{field_name}" if current_path else field_name

            if isinstance(field_value, Tunable):
                flat_map[new_path] = field_value
            elif is_dataclass(field_value):
                _recursive_flatten(field_value, new_path)

    # Start search with an empty path prefix relative to the passed object
    _recursive_flatten(config_obj, "")
    return flat_map
