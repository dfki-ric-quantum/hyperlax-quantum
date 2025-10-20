"""
Thin-layer Distributions

## 1. High-Level Strategy

1. **UniformContinuous**
   - \\( x(u) = a + (b - a)\\,u \\), where \\( u \\in [0,1] \\).

2. **LogUniform**
   - Typically understood as:
     \\[
       X \\sim \text{LogUniform}(a, b) \\quad\\Longleftrightarrow\\quad \\log(X) \\sim \\mathrm{Uniform}(\\log(a), \\log(b)).
     \\]
     So the transform is:
     \\[
       x(u) = \\exp\\Bigl[\\log(a) + (\\log(b) - \\log(a)) \\, u \\Bigr].
     \\]
     Equivalently, \\( x(u) = a \times (b/a)^u \\).

3. **Categorical**
   - a discrete distribution with certain probabilities \\(\\{p_i\\}\\). We can do an inverse CDF approach:
     \\[
       \text{idx}(u) = \\min\bigl\\{\\, i \\,\\mid\\, u < F(i) \bigr\\}, \\quad F(i) = \\sum_{k=0}^{i}p_k.
     \\]
   - Implementation detail: we use `jnp.searchsorted(cdf, u, side="right")`.

4. **UniformDiscrete**
   - Again standard. If \\(X\\) is uniform over \\(\\{a, a+1,\\dots,b\\}\\), you do:
     \\[
       \text{idx}(u) = \\lfloor u * (b-a+1) \rfloor, \\quad \text{clamp to }[0, b-a].
     \\]

5. **LogNormal**
   - The definition:
     \\[
       X = \\exp(\\mu + \\sigma \\cdot \\Phi^{-1}(u)), \\quad u \\in (0,1),
     \\]
     where \\(\\mu,\\sigma\\) are the underlying normal’s location & scale, and \\(\\Phi^{-1}\\) is the standard Normal PPF.

6. **DiscreteQuantized**
   - Interpreted as: we first sample continuously in \\([a,b]\\) from some distribution (often Uniform), then “snap” to discrete steps of size `scale`.
   - A typical approach is:
     \\[
       i(u) = \\lfloor (u\\, (n_{\text{steps}}+1)) \rfloor,\\quad n_{\text{steps}} = \\Bigl\\lfloor \frac{b - a}{\text{scale}} \\Bigr\rfloor.
     \\]
     Then clamp to stay at or below \\((n_{\text{steps}})\\). Finally
     \\[
       x(u) = a + i(u) * \text{scale}.
     \\]
   - Some references also do “round((u*(#steps)) + 0.5)”, etc.  The exact approach depends on how you define “quantized distribution.”
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm


class DistributionRegistry:
    _registry = {}

    @classmethod
    def register(cls, dist_cls):
        cls._registry[dist_cls.__name__] = dist_cls
        return dist_cls

    @classmethod
    def to_dict(cls, dist):
        return {"type": dist.__class__.__name__, "params": dist.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        dist_cls = cls._registry.get(data["type"])
        if not dist_cls:
            raise ValueError(f"Unknown distribution type: {data['type']}")
        return dist_cls.from_dict(data["params"])


class BaseDistribution(ABC):
    @abstractmethod
    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Map U ~ Uniform(0,1) into a sample from this distribution using the inverse CDF or PPF method.
        """

    def ppf(self, u: jnp.ndarray) -> jnp.ndarray:
        """Alias for inverse_transform to maintain compatibility with scipy.stats interface."""
        return self.inverse_transform(u)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, params: dict[str, Any]) -> "BaseDistribution":
        pass


@DistributionRegistry.register
@dataclass
class Normal(BaseDistribution):
    location: float
    scale: float

    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        return self.location + self.scale * jax.scipy.stats.norm.ppf(u)

    def to_dict(self) -> dict[str, Any]:
        return {"location": self.location, "scale": self.scale}

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "Normal":
        return cls(location=params["location"], scale=params["scale"])


@DistributionRegistry.register
@dataclass
class UniformContinuous(BaseDistribution):
    domain: tuple[float, float]

    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        # standard formula: x = a + (b-a)*u
        a, b = self.domain
        return a + (b - a) * jnp.clip(u, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {"domain": list(self.domain)}

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "UniformContinuous":
        return cls(domain=tuple(params["domain"]))


@DistributionRegistry.register
@dataclass
class LogUniform(BaseDistribution):
    domain: tuple[float, float]

    def __post_init__(self):
        a, b = self.domain
        if a <= 0 or b <= 0:
            raise ValueError("LogUniform domain must have positive endpoints.")
        if a >= b:
            raise ValueError("LogUniform domain must have a < b.")

    # NOTE e.g., when u=1.0, we're getting a value slightly larger than the upper bound (100.00001 instead of 100.0)
    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        a, b = self.domain
        # clamp input to [0,1]
        u_clamped = jnp.clip(u, 0.0, 1.0)
        # sample in [log(a), log(b)] uniformly
        log_a = jnp.log(a)
        log_b = jnp.log(b)
        log_x = log_a + (log_b - log_a) * u_clamped
        x = jnp.exp(log_x)
        return jnp.clip(x, a, b)  # clamp to domain bounds to avoid floating point issues

    def to_dict(self) -> dict[str, Any]:
        return {"domain": list(self.domain)}

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "LogUniform":
        return cls(domain=tuple(params["domain"]))


@DistributionRegistry.register
@dataclass
class Categorical(BaseDistribution):
    values: list[Any]
    probabilities: list[float] | None = None

    def __post_init__(self):
        if not self.values:
            raise ValueError("Categorical must have at least one value.")
        if self.probabilities is None:
            self.probabilities = [1.0 / len(self.values)] * len(self.values)
        if len(self.probabilities) != len(self.values):
            raise ValueError("Probabilities length must match # of values.")
        if not jnp.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.0")
        self.cdf = jnp.cumsum(jnp.array(self.probabilities))

    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        u_clamped = jnp.clip(u, 0.0, 1.0)
        idx = jnp.searchsorted(self.cdf, u_clamped, side="right")
        idx = jnp.minimum(idx, len(self.values) - 1)
        return idx

    def map_indices_to_values_as_numpy_strings(self, indices: jnp.ndarray) -> np.ndarray:
        """
        Convert integer indices into the actual category values.
        Returns a NumPy array of strings for consistent encoding.
        NOTE we resort to numpy arrays for string values
        TODO fully jax later
        """
        mapped_list = []
        for idx in indices:
            raw_value = self.values[int(idx)]
            if isinstance(raw_value, (list, tuple)):
                mapped_list.append(str(raw_value))
            elif isinstance(raw_value, bool):
                # mapped_list.append(str(int(raw_value)))
                mapped_list.append(str(raw_value))
            else:
                mapped_list.append(str(raw_value))
        return np.array(mapped_list, dtype=str)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "probabilities": self.probabilities,
        }

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "Categorical":
        return cls(values=params["values"], probabilities=params.get("probabilities"))


@DistributionRegistry.register
@dataclass
class UniformDiscrete(BaseDistribution):
    domain: tuple[int, int]

    def __post_init__(self):
        a, b = self.domain
        if a > b:
            raise ValueError("UniformDiscrete domain must have a <= b.")
        if not (isinstance(a, int) and isinstance(b, int)):
            raise ValueError("UniformDiscrete domain must be integer-based.")

    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        a, b = self.domain
        size = b - a + 1
        u_clamped = jnp.clip(u, 0.0, 1.0)
        idx = jnp.floor(u_clamped * size).astype(int)
        idx = jnp.minimum(idx, size - 1)
        return a + idx

    def to_dict(self) -> dict[str, Any]:
        return {"domain": list(self.domain)}

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "UniformDiscrete":
        return cls(domain=tuple(params["domain"]))


@DistributionRegistry.register
@dataclass
class LogNormal(BaseDistribution):
    location: float  # mu
    scale: float  # sigma

    def __post_init__(self):
        if self.scale <= 0:
            raise ValueError("LogNormal scale must be > 0")

    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Standard approach:
          X = exp(location + scale*norm.ppf(u))
        """
        u_clamped = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        z = norm.ppf(u_clamped)
        return jnp.exp(self.location + self.scale * z)

    def to_dict(self) -> dict[str, Any]:
        return {"location": self.location, "scale": self.scale}

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "LogNormal":
        return cls(location=params["location"], scale=params["scale"])


@DistributionRegistry.register
@dataclass
class DiscreteQuantized(BaseDistribution):
    domain: tuple[float, float]
    scale: float

    def __post_init__(self):
        a, b = self.domain
        if self.scale <= 0:
            raise ValueError("DiscreteQuantized scale must be > 0.")
        if a >= b:
            raise ValueError("domain must have a < b")

    def inverse_transform(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        1) sample from Uniform(a,b)
        2) quantize by scale => a + round((x-a)/scale)*scale
        """
        a, b = self.domain
        u_clamped = jnp.clip(u, 0.0, 1.0)
        x_lin = a + (b - a) * u_clamped

        # "round" to nearest step
        steps = jnp.floor((x_lin - a) / self.scale)
        x_q = a + steps * self.scale

        # clamp so as not to exceed b
        x_q = jnp.clip(x_q, a, b)
        return x_q

    def to_dict(self) -> dict[str, Any]:
        return {"domain": list(self.domain), "scale": self.scale}

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DiscreteQuantized":
        return cls(domain=tuple(params["domain"]), scale=params["scale"])


def apply_inverse_transform(distribution: BaseDistribution, u: np.ndarray) -> np.ndarray:
    """
    Apply the inverse CDF transform to uniform samples.

    Args:
        distribution (BaseDistribution): The target distribution
        u (np.ndarray): Uniform samples in [0, 1)

    Returns:
        np.ndarray: Samples from the target distribution
    """
    jax_u = jnp.array(u)
    raw_samples = distribution.inverse_transform(jax_u)
    if isinstance(distribution, Categorical):
        return np.asarray(distribution.map_indices_to_values_as_numpy_strings(raw_samples))
    return np.asarray(raw_samples)
