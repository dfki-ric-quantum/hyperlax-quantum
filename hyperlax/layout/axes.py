from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a single distribution axis."""

    name: str
    size: int
    method: Literal["vmap", "pmap", "scan"] = "vmap"
    in_axes: Any = 0
    out_axes: Any = 0
    axis_name: str | None = None

    def __post_init__(self):
        if self.axis_name is None:
            object.__setattr__(self, "axis_name", self.name)


@dataclass(frozen=True)
class DistributionStrategy:
    """Complete distribution strategy across all axes."""

    axes: tuple[AxisSpec, ...]

    @property
    def axis_names(self) -> list[str]:
        return [axis.name for axis in self.axes]

    @property
    def axis_sizes_dict(self) -> dict[str, int]:
        return {axis.name: axis.size for axis in self.axes}

    @property
    def axis_sizes_tuple(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    def get_axis_position(self, name: str) -> int:
        """Get the position (index) of an axis by its 'name'."""
        for i, axis in enumerate(self.axes):
            if axis.name == name:
                return i
        raise ValueError(f"Axis '{name}' not found in strategy")

    def has_axis(self, name: str) -> bool:
        """Check if an axis with the given 'name' exists in the strategy."""
        return any(axis.name == name for axis in self.axes)

    def get_axis_spec(self, name: str) -> AxisSpec | None:
        """Get the AxisSpec object for a given axis 'name', or None if not found."""
        for axis in self.axes:
            if axis.name == name:
                return axis
        return None  # Or raise ValueError if preferred for "not found"

    def get_axis_spec_by_position(self, position: int) -> AxisSpec | None:
        """Get the AxisSpec object by its position (index), or None if out of bounds."""
        if 0 <= position < len(self.axes):
            return self.axes[position]
        return None
