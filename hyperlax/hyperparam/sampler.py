"""
This module provides functionality for generating samples from various
probability distributions using different sampling methods.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Union

import jax
import numpy as np
from scipy.stats import qmc

from hyperlax.hyperparam.distributions import BaseDistribution, apply_inverse_transform
from hyperlax.hyperparam.io_utils import save_samples
from hyperlax.runner.batch_utils import find_sample_id_key

if TYPE_CHECKING:
    from hyperlax.cli import SamplingSweepConfig
    from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SamplingConfig:
    """Immutable configuration for sampling parameters."""

    distributions: dict[str, BaseDistribution]
    num_samples: int
    seed: int | None = None
    sampling_method: str = "qmc_sobol"  # ['random', 'qmc_sobol',]


class SobolMatricesFull(NamedTuple):
    """Full set of Sobol matrices including A, B, and AB with their normalized forms."""

    A_unn: dict[str, np.ndarray]
    B_unn: dict[str, np.ndarray]
    AB_unn_list: list[tuple[str, dict[str, np.ndarray]]]
    A_norm: dict[str, np.ndarray]
    B_norm: dict[str, np.ndarray]
    AB_norm_list: list[tuple[str, dict[str, np.ndarray]]]


class SobolMatricesABOmit(NamedTuple):
    """Subset of Sobol matrices including only A and B matrices."""

    A_unn: dict[str, np.ndarray]
    B_unn: dict[str, np.ndarray]
    A_norm: dict[str, np.ndarray]
    B_norm: dict[str, np.ndarray]


class IndependentSamples(NamedTuple):
    """Indepedent Samples"""

    unn: dict[str, np.ndarray]
    norm: dict[str, np.ndarray]


SamplingResult = Union[SobolMatricesFull, SobolMatricesABOmit, IndependentSamples]


def _generate_sobol_sequence_points(
    dim: int, n_samples: int, seed: int | None = None
) -> np.ndarray:
    if seed is None:
        seed = jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**32 - 1)
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    return np.asarray(sampler.random(n=n_samples))


def _generate_random_points(dim: int, n_samples: int, seed: int = 42) -> np.ndarray:
    key = jax.random.PRNGKey(seed if seed is not None else 0)
    return np.asarray(jax.random.uniform(key, (n_samples, dim)))


def _transform_samples_01_to_unnormalized(
    distributions: dict[str, BaseDistribution], samples_01: np.ndarray
) -> dict[str, np.ndarray]:
    """Transform uniform [0,1] samples to their respective distribution domains."""
    param_names = list(distributions.keys())
    assert samples_01.shape[1] == len(param_names), "Dimension mismatch"
    return {
        name: apply_inverse_transform(dist, samples_01[:, i])
        for i, (name, dist) in enumerate(distributions.items())
    }


def generate_independent_samples(
    config: SamplingConfig,
    sample_id_key: str,
) -> IndependentSamples:
    """Generate independent samples based on the provided configuration."""
    # skip e.g., __JOINT_SAMPLING__
    actual_distributions = {k: v for k, v in config.distributions.items() if not k.startswith("__")}
    dim = len(actual_distributions)
    if config.sampling_method == "qmc_sobol":
        samples_01 = _generate_sobol_sequence_points(dim, config.num_samples, config.seed)
    elif config.sampling_method == "random":
        samples_01 = _generate_random_points(dim, config.num_samples, config.seed)
    else:
        raise ValueError(f"Unsupported sampling method: {config.sampling_method}")

    unnormalized_dict = _transform_samples_01_to_unnormalized(actual_distributions, samples_01)
    normalized_dict = {
        pname: samples_01[:, i] for i, pname in enumerate(actual_distributions.keys())
    }

    sample_ids = np.arange(config.num_samples)
    unnormalized_dict[sample_id_key] = sample_ids
    normalized_dict[sample_id_key] = sample_ids

    return IndependentSamples(unn=unnormalized_dict, norm=normalized_dict)


def generate_sobol_matrices_A_B_AB(
    config: SamplingConfig, sample_id_key: str, omit_ab: bool = False
) -> SobolMatricesFull | SobolMatricesABOmit:
    """Generate Sobol matrices for sensitivity analysis."""
    distributions = {k: v for k, v in config.distributions.items() if not k.startswith("__")}
    N = config.num_samples
    seed = config.seed
    param_names = list(distributions.keys())
    D = len(param_names)

    sampler = qmc.Sobol(d=D, scramble=True, seed=seed)
    raw_01 = np.asarray(sampler.random(n=N * (2 if omit_ab else D + 2)))

    A_norm = raw_01[:N]
    B_norm = raw_01[N : 2 * N]

    A_norm_dict = {param: A_norm[:, i] for i, param in enumerate(param_names)}
    B_norm_dict = {param: B_norm[:, i] for i, param in enumerate(param_names)}

    A_unn = _transform_samples_01_to_unnormalized(distributions, A_norm)
    B_unn = _transform_samples_01_to_unnormalized(distributions, B_norm)

    sample_ids = np.arange(N)
    A_norm_dict[sample_id_key] = sample_ids
    A_unn[sample_id_key] = sample_ids
    B_norm_dict[sample_id_key] = sample_ids
    B_unn[sample_id_key] = sample_ids

    if omit_ab:
        return SobolMatricesABOmit(
            A_unn=A_unn, B_unn=B_unn, A_norm=A_norm_dict, B_norm=B_norm_dict
        )

    AB_norm_list = []
    for i in range(D):
        AB_i = A_norm.copy()
        AB_i[:, i] = B_norm[:, i]
        AB_norm_list.append(AB_i)

    AB_norm_dict_list = []
    for i, param_name in enumerate(param_names):
        ab_i_norm_dict = {param: AB_norm_list[i][:, j] for j, param in enumerate(param_names)}
        ab_i_norm_dict[sample_id_key] = sample_ids
        AB_norm_dict_list.append((param_name, ab_i_norm_dict))

    AB_unn_list = []
    for i, param_name in enumerate(param_names):
        ab_i_unn = _transform_samples_01_to_unnormalized(distributions, AB_norm_list[i])
        ab_i_unn[sample_id_key] = sample_ids
        AB_unn_list.append((param_name, ab_i_unn))

    return SobolMatricesFull(
        A_unn=A_unn,
        B_unn=B_unn,
        AB_unn_list=AB_unn_list,
        A_norm=A_norm_dict,
        B_norm=B_norm_dict,
        AB_norm_list=AB_norm_dict_list,
    )

def _apply_joint_sampling_rules(
    samples: dict[str, list], exp_config_container: "AlgoSpecificExperimentConfigContainer"
) -> dict[str, list]:
    """Post-processes samples to apply joint sampling rules defined in the hyperparam distributions."""
    dist_config = exp_config_container.hyperparam_dist_config
    if "__JOINT_SAMPLING__" not in dist_config:
        return samples

    logger.info("Applying joint sampling rules to the hyperparameter set.")
    joint_rules = dist_config["__JOINT_SAMPLING__"]
    processed_samples = samples.copy()

    for proxy_param, rule in joint_rules.items():
        if proxy_param not in processed_samples:
            logger.warning(
                f"Proxy parameter '{proxy_param}' for joint sampling was not found in the generated samples. Skipping this rule."
            )
            continue

        indices = processed_samples[proxy_param]
        targets = rule["targets"]
        choices = rule["choices"]

        # Initialize lists for target parameters
        for target_param in targets:
            processed_samples[target_param] = [None] * len(indices)

        # Populate target parameters based on sampled indices
        for i, index in enumerate(indices):
            try:
                # The index from Categorical is a string, needs to be int
                chosen_values = choices[int(index)]
                if len(chosen_values) != len(targets):
                    raise ValueError(
                        f"Mismatch between number of targets ({len(targets)}) and values in choice ({len(chosen_values)})."
                    )
                for target_param, value in zip(targets, chosen_values, strict=True):
                    processed_samples[target_param][i] = value
            except (IndexError, ValueError) as e:
                logger.error(
                    f"Error applying joint sampling rule for proxy '{proxy_param}' at sample index {i} with choice index {index}: {e}"
                )
                # Set targets to None to indicate failure for this sample
                for target_param in targets:
                    processed_samples[target_param][i] = None

    # Remove the proxy parameter from the final samples as it has served its purpose
    for proxy_param in joint_rules:
        if proxy_param in processed_samples:
            del processed_samples[proxy_param]

    logger.debug(
        "Finished applying joint sampling rules. Final sample keys: %s",
        list(processed_samples.keys()),
    )
    return processed_samples


def generate_samples(
    runner_config: "SamplingSweepConfig",
    algo_exp_info: "AlgoSpecificExperimentConfigContainer",
) -> SamplingResult:
    """Prepare samples based on experiment type and configuration."""
    sampling_config = SamplingConfig(
        distributions=algo_exp_info.hyperparam_dist_config,
        num_samples=runner_config.num_samples,
        seed=runner_config.seed,
        sampling_method=runner_config.sampling_method,
    )
    unnormalized, normalized = None, None
    sampling_res: SamplingResult

    sample_id_key = find_sample_id_key(algo_exp_info.experiment_config)

    if runner_config.experiment_type == "sobol_matrices_A_B_AB":
        sampling_res = generate_sobol_matrices_A_B_AB(
            sampling_config, sample_id_key, omit_ab=runner_config.omit_ab
        )
    elif runner_config.experiment_type == "independent_samples":
        sampling_res = generate_independent_samples(sampling_config, sample_id_key)
    else:
        raise ValueError(f"Unsupported experiment type: {runner_config.experiment_type}")

    # --- Apply joint sampling rules to the unnormalized samples BEFORE saving ---
    if isinstance(sampling_res, IndependentSamples):
        processed_unn = _apply_joint_sampling_rules(sampling_res.unn, algo_exp_info)
        sampling_res = sampling_res._replace(unn=processed_unn)
    elif isinstance(sampling_res, (SobolMatricesFull, SobolMatricesABOmit)):
        processed_A = _apply_joint_sampling_rules(sampling_res.A_unn, algo_exp_info)
        processed_B = _apply_joint_sampling_rules(sampling_res.B_unn, algo_exp_info)
        if isinstance(sampling_res, SobolMatricesFull):
            processed_AB_list = [
                (name, _apply_joint_sampling_rules(s, algo_exp_info))
                for name, s in sampling_res.AB_unn_list
            ]
            sampling_res = sampling_res._replace(
                A_unn=processed_A, B_unn=processed_B, AB_unn_list=processed_AB_list
            )
        else:  # SobolMatricesABOmit
            sampling_res = sampling_res._replace(A_unn=processed_A, B_unn=processed_B)

    # Extract the correct unnormalized samples for saving
    if isinstance(sampling_res, IndependentSamples):
        unnormalized = sampling_res.unn
        normalized = sampling_res.norm
    elif isinstance(sampling_res, SobolMatricesFull):
        unnormalized = (
            sampling_res.A_unn,
            sampling_res.B_unn,
            sampling_res.AB_unn_list,
        )
        normalized = (
            sampling_res.A_norm,
            sampling_res.B_norm,
            sampling_res.AB_norm_list,
        )
    elif isinstance(sampling_res, SobolMatricesABOmit):
        unnormalized = (sampling_res.A_unn, sampling_res.B_unn)
        normalized = (sampling_res.A_norm, sampling_res.B_norm)

    save_samples(
        runner_config.output_dir,
        unnormalized,
        normalized,
        runner_config.experiment_type,
    )
    return sampling_res
