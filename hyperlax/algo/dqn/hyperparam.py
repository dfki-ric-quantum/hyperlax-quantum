from dataclasses import dataclass, field

from hyperlax.hyperparam.distributions import Categorical, LogUniform, UniformContinuous
from hyperlax.hyperparam.tunable import Tunable


@dataclass
class DQNHyperparams:
    """Complete Hyperparameters for DQN, with values and specifications merged using Tunable."""

    critic_lr: Tunable = field(
        default_factory=lambda: Tunable(
            value=7e-5,
            is_vectorized=True,
            is_fixed=False,
            distribution=LogUniform(domain=(1e-5, 1e-3)),
            expected_type=float,
        )
    )
    tau: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.005,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.001, 1.0)),
            expected_type=float,
        )
    )
    gamma: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.99,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.9, 0.9999)),
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
    training_epsilon: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.1,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.01, 0.2)),
            expected_type=float,
        )
    )
    evaluation_epsilon: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.0, is_vectorized=True, is_fixed=True, expected_type=float
        )
    )
    max_abs_reward: Tunable = field(
        default_factory=lambda: Tunable(
            value=1e9,
            is_vectorized=True,
            is_fixed=True,  # effectively disabling the effect!
            # distribution=LogUniform(domain=(100.0, 10000.0)),
            expected_type=float,
        )
    )
    huber_loss_parameter: Tunable = field(
        default_factory=lambda: Tunable(
            value=0.0,
            is_vectorized=True,
            is_fixed=False,
            distribution=UniformContinuous(domain=(0.0, 1.0)),
            expected_type=float,
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
    total_num_envs: Tunable = field(
        default_factory=lambda: Tunable(
            value=1024,
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
            distribution=Categorical(values=[2, 4, 8]),
            expected_type=int,
        )
    )
    epochs: Tunable = field(
        default_factory=lambda: Tunable(
            value=16,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[4, 6, 8, 10, 16]),
            expected_type=int,
        )
    )
    total_buffer_size: Tunable = field(
        default_factory=lambda: Tunable(
            value=int(1e5),
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[int(1e4), int(5e4), int(1e5), int(2e5)]),
            expected_type=int,
        )
    )
    total_batch_size: Tunable = field(
        default_factory=lambda: Tunable(
            value=512,
            is_vectorized=True,
            is_fixed=False,
            distribution=Categorical(values=[32, 64, 128, 256, 512]),
            expected_type=int,
        )
    )
    decay_learning_rates: Tunable = field(
        default_factory=lambda: Tunable(
            value=False,
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
    use_double_q: Tunable = field(
        default_factory=lambda: Tunable(
            value=True,
            is_vectorized=False,
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
