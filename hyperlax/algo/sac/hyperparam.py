from dataclasses import dataclass, field

from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous
from hyperlax.hyperparam.tunable import Tunable


@dataclass(frozen=True)
class SACHyperparams:
    """Complete Hyperparameters for SAC, with values and specifications merged using Tunable."""

    actor_lr: Tunable = field(
        default_factory=lambda: Tunable(
            value=3e-4,
            is_vectorized=True,
            is_fixed=False,
            distribution=LogUniform(domain=(1e-5, 1e-3)),
            expected_type=float,
        )
    )
    q_lr: Tunable = field(
        default_factory=lambda: Tunable(
            value=3e-4,
            is_vectorized=True,
            is_fixed=False,
            distribution=LogUniform(domain=(1e-5, 1e-3)),
            expected_type=float,
        )
    )
    alpha_lr: Tunable = field(
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
    tau: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.005,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.001, 0.1)),
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
    autotune: Tunable = field(
        default_factory=lambda: Tunable(
            value=True,
            is_vectorized=False,
            is_fixed=False,
            distribution=Categorical(values=[True, False]),
            expected_type=bool,
        )
    )
    target_entropy_scale: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.7,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.1, 1.0)),
            expected_type=float,
        )
    )
    init_alpha: Tunable = field(
        default_factory=lambda: Tunable(
            value=1.0,
            is_vectorized=True,
            is_fixed=False,
            distribution=LogUniform(domain=(0.1, 10.0)),
            expected_type=float,
        )
    )
    total_num_envs: Tunable = field(
        default_factory=lambda: Tunable(
            value=128,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[64, 128, 256, 512, 1024]),
            expected_type=int,
        )
    )
    rollout_length: Tunable = field(
        default_factory=lambda: Tunable(
            value=2,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(
                values=[
                    2,
                    4,
                    8,
                ]
            ),
            expected_type=int,
        )
    )
    warmup_rollout_length: Tunable = field(
        default_factory=lambda: Tunable(
            value=4,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[4, 6, 8, 10]),
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
    total_buffer_size: Tunable = field(
        default_factory=lambda: Tunable(
            value=int(1e6),
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[int(1e5), int(1e6)]),
            expected_type=int,
        )
    )
    total_batch_size: Tunable = field(
        default_factory=lambda: Tunable(
            value=256,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[128, 256, 512]),
            expected_type=int,
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
