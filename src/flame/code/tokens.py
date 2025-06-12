"""
This module provides backward compatibility for code that still imports from flame.code.tokens.
The original tokens.py file was removed as part of a refactoring effort.
"""

import warnings
from typing import List


def tokens(api_model_string: str) -> List[str]:
    """
    This function is maintained for backward compatibility but is deprecated.

    Args:
        api_model_string (str): The API model string (not used)

    Returns:
        List[str]: An empty list - stop tokens are now handled automatically by LiteLLM

    Raises:
        DeprecationWarning: To indicate this function should not be used
    """
    warnings.warn(
        "The tokens function is deprecated. Do not pass stop tokens with LiteLLM.",
        DeprecationWarning,
        stacklevel=2,
    )
    return []
