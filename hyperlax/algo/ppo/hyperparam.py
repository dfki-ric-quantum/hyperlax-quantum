from dataclasses import dataclass, field

from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous
from hyperlax.hyperparam.tunable import Tunable


@dataclass(frozen=True)
class PPOHyperparams:
    """Complete Hyperparameters for PPO, with values and specifications merged using Tunable."""

    actor_lr: Tunable = field(
        default_factory=lambda: Tunable(
            value=3e-4,
            is_vectorized=True,
            is_fixed=False,
            distribution=LogUniform(domain=(1e-5, 1e-3)),
            expected_type=float,
        )
    )
    critic_lr: Tunable = field(
        default_factory=lambda: Tunable(
            value=3e-4,
            is_vectorized=True,
            is_fixed=False,
            distribution=LogUniform(domain=(1e-5, 1e-3)),
            expected_type=float,
        )
    )
    gamma: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.99,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.9, 0.999)),
            expected_type=float,
        )
    )
    gae_lambda: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.95,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.9, 1.0)),
            expected_type=float,
        )
    )
    clip_eps: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.2,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.1, 0.3)),
            expected_type=float,
        )
    )
    ent_coef: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.01,
            is_vectorized=True,
            is_fixed=False,
            distribution=LogUniform(domain=(1e-4, 1e-1)),
            expected_type=float,
        )
    )
    vf_coef: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.5,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.25, 0.75)),
            expected_type=float,
        )
    )
    max_grad_norm: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.5,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.1, 1.0)),
            expected_type=float,
        )
    )
    rollout_length: Tunable = field(
        default_factory=lambda: Tunable(
            value=2,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[2, 4, 8]),
            expected_type=int,
        )
    )
    epochs: Tunable = field(
        default_factory=lambda: Tunable(
            value=1,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[1, 5, 10]),
            expected_type=int,
        )
    )
    num_minibatches: Tunable = field(
        default_factory=lambda: Tunable(
            value=2,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[32, 64, 128, 256, 512]),
            expected_type=int,
        )
    )
    total_num_envs: Tunable = field(
        default_factory=lambda: Tunable(
            value=4,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[64, 128, 256, 512, 1024]),
            expected_type=int,
        )
    )
    standardize_advantages: Tunable = field(
        default_factory=lambda: Tunable(
            value=True,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
    decay_learning_rates: Tunable = field(
        default_factory=lambda: Tunable(
            value=True,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
    normalize_observations: Tunable = field(
        default_factory=lambda: Tunable(
            value=True,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
    sample_id: Tunable = field(
        default_factory=lambda: Tunable(
            value=-1, is_vectorized=True, is_fixed=True, expected_type=int
        )
    )
