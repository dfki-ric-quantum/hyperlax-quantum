import logging

import pytest

from hyperlax.runner.base_types import AlgoSpecificExperimentConfigContainer
from hyperlax.runner.sampling import _apply_joint_sampling_rules


# A dummy object to satisfy the type hint for the container in tests
class MockConfig:
    pass


@pytest.fixture
def mock_exp_container(request):
    """Fixture to provide a mock experiment container with a specific distribution config."""
    dist_config = request.param if hasattr(request, "param") else {}
    return AlgoSpecificExperimentConfigContainer(
        algo_name="mock_algo",
        algo_main_fn=lambda: None,
        experiment_config=MockConfig(),
        hyperparam_dist_config=dist_config,
        hyperparams_container_spec=None,
    )


# --- Test Cases ---


@pytest.mark.parametrize("mock_exp_container", [{}], indirect=True)
def test_joint_sampling_no_op(mock_exp_container):
    """Tests that the function does nothing if __JOINT_SAMPLING__ is not in the config."""
    samples = {"param1": [0.1, 0.2], "param2": [10, 20]}
    original_samples = samples.copy()
    processed_samples = _apply_joint_sampling_rules(samples, mock_exp_container)
    assert processed_samples == original_samples


@pytest.mark.parametrize("mock_exp_container", [{"__JOINT_SAMPLING__": {}}], indirect=True)
def test_joint_sampling_empty_samples(mock_exp_container):
    """Tests that the function handles an empty samples dictionary correctly."""
    samples = {}
    processed_samples = _apply_joint_sampling_rules(samples, mock_exp_container)
    assert processed_samples == {}


def test_joint_sampling_basic_success():
    """Tests a single, valid joint sampling rule."""
    rule = {
        "__JOINT_SAMPLING__": {
            "proxy.arch_choice": {
                "targets": ["target.hidden_dim", "target.num_nodes"],
                "choices": [[32, 5], [64, 6], [128, 7]],
            }
        }
    }
    container = AlgoSpecificExperimentConfigContainer(None, None, None, rule, None)
    samples = {
        "proxy.arch_choice": ["0", "2", "1"],  # Categorical samples are strings
        "independent.param": [0.5, 0.6, 0.7],
    }

    processed_samples = _apply_joint_sampling_rules(samples, container)

    assert "proxy.arch_choice" not in processed_samples
    assert "target.hidden_dim" in processed_samples
    assert "target.num_nodes" in processed_samples
    assert "independent.param" in processed_samples

    assert processed_samples["target.hidden_dim"] == [32, 128, 64]
    assert processed_samples["target.num_nodes"] == [5, 7, 6]
    assert processed_samples["independent.param"] == [0.5, 0.6, 0.7]


def test_joint_sampling_multiple_rules():
    """Tests that multiple joint sampling rules are applied correctly."""
    rules = {
        "__JOINT_SAMPLING__": {
            "proxy.arch_choice": {
                "targets": ["target.hidden_dim", "target.num_nodes"],
                "choices": [[32, 5], [64, 6]],
            },
            "proxy.optim_choice": {
                "targets": ["target.lr", "target.beta"],
                "choices": [[1e-3, 0.9], [5e-4, 0.99]],
            },
        }
    }
    container = AlgoSpecificExperimentConfigContainer(None, None, None, rules, None)
    samples = {"proxy.arch_choice": ["1", "0"], "proxy.optim_choice": ["0", "1"]}

    processed_samples = _apply_joint_sampling_rules(samples, container)

    assert "proxy.arch_choice" not in processed_samples
    assert "proxy.optim_choice" not in processed_samples
    assert processed_samples["target.hidden_dim"] == [64, 32]
    assert processed_samples["target.num_nodes"] == [6, 5]
    assert processed_samples["target.lr"] == [1e-3, 5e-4]
    assert processed_samples["target.beta"] == [0.9, 0.99]


def test_joint_sampling_missing_proxy_in_samples(caplog):
    """Tests that a rule is skipped if its proxy parameter is not in the samples."""
    rule = {
        "__JOINT_SAMPLING__": {
            "proxy.arch_choice": {
                "targets": ["target.hidden_dim", "target.num_nodes"],
                "choices": [[32, 5]],
            }
        }
    }
    container = AlgoSpecificExperimentConfigContainer(None, None, None, rule, None)
    samples = {"independent.param": [0.5, 0.6]}
    original_samples = samples.copy()

    with caplog.at_level(logging.WARNING):
        processed_samples = _apply_joint_sampling_rules(samples, container)

    assert processed_samples == original_samples
    assert "Proxy parameter 'proxy.arch_choice' for joint sampling was not found" in caplog.text


def test_joint_sampling_out_of_bounds_index(caplog):
    """Tests behavior when a sampled proxy index is out of bounds for the choices."""
    rule = {
        "__JOINT_SAMPLING__": {
            "proxy.arch_choice": {
                "targets": ["target.hidden_dim", "target.num_nodes"],
                "choices": [[32, 5]],  # Only one choice at index 0
            }
        }
    }
    container = AlgoSpecificExperimentConfigContainer(None, None, None, rule, None)
    samples = {"proxy.arch_choice": ["0", "1"]}  # Index '1' is out of bounds

    with caplog.at_level(logging.ERROR):
        processed_samples = _apply_joint_sampling_rules(samples, container)

    assert processed_samples["target.hidden_dim"] == [32, None]
    assert processed_samples["target.num_nodes"] == [5, None]
    assert "Error applying joint sampling rule" in caplog.text
    assert "list index out of range" in caplog.text


def test_joint_sampling_mismatched_choice_length(caplog):
    """Tests error handling when a choice tuple has a different length than the targets list."""
    rule = {
        "__JOINT_SAMPLING__": {
            "proxy.arch_choice": {
                "targets": ["target.hidden_dim", "target.num_nodes"],  # Expects 2 values
                "choices": [[32]],  # Provides only 1 value
            }
        }
    }
    container = AlgoSpecificExperimentConfigContainer(None, None, None, rule, None)
    samples = {"proxy.arch_choice": ["0"]}

    with caplog.at_level(logging.ERROR):
        processed_samples = _apply_joint_sampling_rules(samples, container)

    assert processed_samples["target.hidden_dim"] == [None]
    assert processed_samples["target.num_nodes"] == [None]
    assert "Error applying joint sampling rule" in caplog.text
    assert "Mismatch between number of targets (2) and values in choice (1)" in caplog.text
