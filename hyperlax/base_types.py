import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, TypeVar

import chex
import jax.numpy as jnp
from distrax import DistributionLike
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from optax import OptState

from hyperlax.configs.main_base import BaseExperimentConfig
from hyperlax.normalizer.running_stats import (
    InitNormFn,
    NormalizeFn,
    UpdateNormFn,
)

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    pass
else:
    pass


Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
Truncated: TypeAlias = chex.Array
First: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any
Parameters: TypeAlias = Any
OptStates: TypeAlias = Any
HiddenStates: TypeAlias = Any


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: chex.Array  # (,)


class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralised algorithms.
    Extends `Observation` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agent_view: chex.Array
    action_mask: chex.Array
    global_state: chex.Array
    step_count: chex.Array


class LogEnvState(NamedTuple):
    """State of the `LogWrapper`."""

    env_state: State
    episode_returns: chex.Numeric
    episode_lengths: chex.Numeric
    # Information about the episode return and length for logging purposes.
    episode_return_info: chex.Numeric
    episode_length_info: chex.Numeric


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count: chex.Array
    episode_return: chex.Array


class ActorCriticParams(NamedTuple):
    """Parameters of an actor critic fnapprox."""

    actor_params: FrozenDict
    critic_params: FrozenDict


class ActorCriticOptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class OnlineAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


GenericState = TypeVar(
    "GenericState",
)

class AnakinTrainOutput(NamedTuple):
    """Experiment output."""

    learner_state: GenericState
    episode_metrics: dict[str, chex.Array]
    train_metrics: dict[str, chex.Array]


RNNObservation: TypeAlias = tuple[Observation, Done]
LearnerFn = Callable[[GenericState], AnakinTrainOutput[GenericState]]
EvalFn = Callable[[FrozenDict, chex.PRNGKey], AnakinTrainOutput[GenericState]]

ActorApply = Callable[[FrozenDict, Observation], DistributionLike]
ActFn = Callable[[FrozenDict, Observation, chex.PRNGKey], chex.Array]
CriticApply = Callable[[FrozenDict, Observation], Value]
ContinuousQApply = Callable[[FrozenDict, Observation, Action], Value]

RecActorApply = Callable[
    [FrozenDict, HiddenState, RNNObservation], tuple[HiddenState, DistributionLike]
]
RecActFn = Callable[
    [FrozenDict, HiddenState, RNNObservation, chex.PRNGKey],
    tuple[HiddenState, chex.Array],
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RNNObservation], tuple[HiddenState, Value]]


class HPRuntimeState(NamedTuple):
    original_index: int
    env_steps: int
    is_active: bool
    current_milestone_target_idx: int


@dataclasses.dataclass(frozen=True)
class EvaluationMetrics:
    """Standardized evaluation result that algorithms must return."""

    episode_metrics: dict
    other_metrics: dict[str, jnp.ndarray] | None = None


@dataclasses.dataclass(frozen=True)
class AlgorithmGlobalSetupArgs:
    env: Environment
    eval_env: Environment
    config: BaseExperimentConfig
    normalizer_fns: tuple[InitNormFn, UpdateNormFn, NormalizeFn]
    get_eval_act_fn_callback: Callable
    algo_specific_keys: tuple[chex.PRNGKey, ...]
