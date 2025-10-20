import ast
from typing import Any

import jax.numpy as jnp
import numpy as np


def cast_value_to_expected_type(value: Any, expected_type: type) -> Any:
    if type(value) is expected_type:
        return value

    try:
        # Handle list[int] types specifically (or other specific list types)
        if (
            hasattr(expected_type, "__origin__") and expected_type.__origin__ == list
        ) or expected_type is list:
            if isinstance(value, str):
                # Try to parse string representation of a list
                try:
                    parsed_value = ast.literal_eval(value)
                    if not isinstance(parsed_value, list):  # Ensure it's a list after eval
                        raise TypeError(f"String '{value}' did not evaluate to a list.")
                except (ValueError, SyntaxError) as e_ast:
                    raise TypeError(
                        f"Cannot parse string '{value}' as a list for expected type {expected_type}: {e_ast}"
                    )
            elif isinstance(
                value, (list, tuple, np.ndarray, jnp.ndarray)
            ):  # Handle various sequence types
                parsed_value = list(value)  # Convert to Python list first
            else:
                raise TypeError(
                    f"Cannot convert value of type {type(value)} to list for expected type {expected_type}"
                )

            # Now, cast elements within the list
            element_type = (
                expected_type.__args__[0]
                if hasattr(expected_type, "__args__") and expected_type.__args__
                else Any
            )
            if element_type is Any:  # No specific element type, return as is
                return parsed_value
            else:
                # Recursively cast each element
                return [cast_value_to_expected_type(i, element_type) for i in parsed_value]

        # Handle booleans from various input types
        elif expected_type == bool:
            if isinstance(value, str):
                lower_value = value.lower()
                if lower_value == "true":
                    return True
                if lower_value == "false":
                    return False
                # Try to interpret as numeric if 'true'/'false' fails
                try:
                    value_as_num = float(value)
                except ValueError:
                    raise TypeError(
                        f"String '{value}' is not 'true', 'false', or a valid number for boolean cast."
                    )
            elif isinstance(value, (jnp.ndarray, np.ndarray)):
                if value.size == 1:
                    value_as_num = value.item()  # Extract scalar
                else:
                    raise TypeError(f"Cannot cast multi-element array {value} to single boolean.")
            elif isinstance(value, (int, float, np.integer, np.floating)):
                value_as_num = value
            else:  # Other unexpected types
                raise TypeError(f"Cannot cast value of type {type(value)} to boolean.")

            # Numeric to boolean conversion (1.0/1 -> True, 0.0/0 -> False)
            if isinstance(value_as_num, (int, float)):
                if abs(value_as_num - 1.0) < 1e-9 or value_as_num == 1:
                    return True
                if abs(value_as_num - 0.0) < 1e-9 or value_as_num == 0:
                    return False
            raise TypeError(
                f"Numeric value {value_as_num} (from original {value}) is not 0/1 for boolean cast."
            )

        # General casting for other types (int, float, etc.)
        else:
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                if value.size == 1:
                    return expected_type(value.item())
                else:
                    raise TypeError(
                        f"Cannot cast multi-element array {value} to scalar type {expected_type}"
                    )
            return expected_type(value)

    except (ValueError, TypeError, SyntaxError) as e:
        # Return None or raise a more specific error.
        # For this case, returning None is what caused the issue in the log.
        # It's better to raise an error if casting truly fails, but the boolean case needs to succeed.
        # The boolean logic above should now handle the 0.0/1.0 cases before this catch.
        # However, if other unexpected issues arise, this catch is still here.
        # For the reported issue, the problem was that the boolean logic didn't correctly handle
        # numeric inputs like JAX arrays of 0.0/1.0.
        raise TypeError(
            f"Failed to cast value '{value}' (type: {type(value)}) to expected type {expected_type}: {e}"
        )
