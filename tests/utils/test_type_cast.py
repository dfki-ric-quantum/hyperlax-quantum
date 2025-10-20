
import jax.numpy as jnp
import numpy as np
import pytest

from hyperlax.utils.type_cast import cast_value_to_expected_type


# --- Basic Type Casting ---
def test_cast_noop():
    assert cast_value_to_expected_type(5, int) == 5
    assert cast_value_to_expected_type(5.5, float) == 5.5


def test_cast_string_to_numeric():
    assert cast_value_to_expected_type("10", int) == 10
    assert cast_value_to_expected_type("10.5", float) == 10.5
    with pytest.raises(TypeError):
        cast_value_to_expected_type("not_a_number", int)


def test_cast_numeric_to_numeric():
    assert cast_value_to_expected_type(10.5, int) == 10
    assert cast_value_to_expected_type(10, float) == 10.0


def test_cast_array_to_scalar():
    assert cast_value_to_expected_type(jnp.array(5), int) == 5
    assert cast_value_to_expected_type(np.array(5.5), float) == 5.5
    with pytest.raises(TypeError):
        cast_value_to_expected_type(jnp.array([1, 2]), int)


# --- Boolean Casting ---
@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        (1, True),
        (1.0, True),
        (jnp.array(1.0), True),
        (np.array(1), True),
        (0, False),
        (0.0, False),
        (jnp.array(0.0), False),
        (np.array(0), False),
    ],
)
def test_cast_to_bool_success(value, expected):
    assert cast_value_to_expected_type(value, bool) is expected


@pytest.mark.parametrize(
    "value",
    [
        "t",
        "f",
        "yes",
        "no",
        "",
        2,
        -1,
        0.5,
        jnp.array([1, 0]),
        np.array(2.0),
    ],
)
def test_cast_to_bool_failure(value):
    with pytest.raises(TypeError):
        cast_value_to_expected_type(value, bool)


# --- List Casting ---
def test_cast_string_to_list():
    assert cast_value_to_expected_type("[1, 2, 3]", list[int]) == [1, 2, 3]
    assert cast_value_to_expected_type("['a', 'b']", list[str]) == ["a", "b"]
    assert cast_value_to_expected_type("[1.0, 2, '3']", list) == [1.0, 2, "3"]


def test_cast_string_to_list_failure():
    with pytest.raises(TypeError):
        cast_value_to_expected_type("1, 2, 3", list[int])
    with pytest.raises(TypeError):
        cast_value_to_expected_type("{'a': 1}", list[int])  # not a list


def test_cast_sequence_to_list():
    assert cast_value_to_expected_type((1, 2, 3), list[int]) == [1, 2, 3]
    assert cast_value_to_expected_type(np.array([1.1, 2.2]), list[float]) == [1.1, 2.2]
    assert cast_value_to_expected_type(np.array(["1", "2"]), list[str]) == ["1", "2"]


def test_cast_list_with_element_casting():
    assert cast_value_to_expected_type(["1", 2.0, 3], list[int]) == [1, 2, 3]
    assert cast_value_to_expected_type(["1.1", 2, "3.3"], list[float]) == [1.1, 2.0, 3.3]
    assert cast_value_to_expected_type([True, 0], list[bool]) == [True, False]


def test_cast_list_failure_on_element():
    with pytest.raises(TypeError):
        cast_value_to_expected_type(["1", "two", "3"], list[int])
