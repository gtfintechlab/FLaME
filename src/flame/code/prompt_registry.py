"""
DEPRECATED: This module has been moved to flame.code.prompts.registry

This module now simply re-exports the contents of flame.code.prompts.registry
for backward compatibility. It will be removed in a future release.

Please update your imports to use the new location:
from flame.code.prompts import get_prompt, PromptFormat, register_prompt

Instead of:
from flame.code.prompt_registry import get_prompt, PromptFormat
"""

import warnings

# Issue deprecation warning immediately
warnings.filterwarnings("always", category=DeprecationWarning)
warnings.warn(
    "prompt_registry.py is deprecated and will be removed in a future version. "
    "Use flame.code.prompts.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all components from new location

# For backward compatibility, expose the registry
