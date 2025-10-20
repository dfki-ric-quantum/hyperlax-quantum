import logging
from pathlib import Path

import pandas as pd

from hyperlax.hyperparam.sampler import SamplingConfig, generate_independent_samples
from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer
from hyperlax.runner.batch_utils import find_sample_id_key

logger = logging.getLogger(__name__)


def generate_hyperparam_sample_and_save(
    output_file: Path,
    num_samples: int,
    sampling_method: str,
    seed: int,
    exp_config_container: AlgoSpecificExperimentConfigContainer,
) -> None:
    """
    Generates a set of hyperparameter samples and saves them to a single CSV file.
    """
    sampling_config = SamplingConfig(
        distributions=exp_config_container.hyperparam_dist_config,
        num_samples=num_samples,
        seed=seed,
        sampling_method=sampling_method,
    )
    logger.info(f"Generating {num_samples} samples with method '{sampling_method}'...")

    sample_id_key = find_sample_id_key(exp_config_container.experiment_config)

    sampling_result = generate_independent_samples(sampling_config, sample_id_key)
    samples_dict = sampling_result.unn

    try:
        df = pd.DataFrame.from_dict(samples_dict)
    except ValueError as e:
        logger.error(
            f"Failed to create DataFrame from generated samples. This can happen if sample lists have inconsistent lengths. Error: {e}"
        )
        # Add a preview of the sample dictionary keys and lengths for debugging
        for key, value in samples_dict.items():
            logger.error(f"  - Key: '{key}', Length: {len(value)}")
        raise

    # 5. Reorder columns for better readability, placing 'sample_id' first.
    if "sample_id" in df.columns:
        cols = ["sample_id"] + [col for col in df.columns if col != "sample_id"]
        df = df[cols]

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(
            f"Successfully saved {len(df)} hyperparameter samples to: {output_file.resolve()}"
        )
    except Exception as e:
        logger.error(f"Failed to write samples to file '{output_file.resolve()}': {e}")
        raise
