# FLaME Prompt System Reorganization

## Original Pain Points

The prompt system in FLaME had several issues that made maintenance difficult:

1. **Duplicated Functions**: The same prompts were implemented in multiple files with subtle differences
2. **Inconsistent Naming**: Function names didn't follow a clear pattern (e.g., `numclaim_prompt` vs `fpb_zeroshot_prompt`)
3. **No Central Registry**: Each file had its own `prompt_map` for lookups
4. **Poor Discoverability**: No easy way to see which prompts were available for a task
5. **Inconsistent Organization**: Prompts were scattered across 5+ files with no clear pattern
6. **No Type Hints or Documentation**: Many functions lacked proper documentation or type hints

## Our Approach

We took a three-phased approach to improve the prompt system:

1. **Immediate Cleanup**: Fix critical issues with minimum disruption to existing code
   - Remove duplicate functions
   - Add imports for backward compatibility
   - Fix inconsistent naming
   - Restore missing functionality

2. **Medium-term Reorganization**: Create a better structure while maintaining compatibility
   - Implement a package-based structure
   - Create a central registry with enumerated formats
   - Add proper type hints and docstrings
   - Create backward compatibility layers
   - Write comprehensive tests

3. **Long-term Structure**: Plan for future improvements
   - Task-based organization
   - Consistent API for all prompts
   - Better error handling
   - Improved documentation

## Accomplishments

We've successfully completed the immediate cleanup and most of the medium-term reorganization:

1. **New Package Structure**:
   - Created `src/flame/code/prompts/` package
   - Implemented `registry.py` with centralized registry
   - Created format-specific modules: `base.py`, `zeroshot.py`, `fewshot.py`
   - Added comprehensive docstrings and type hints

2. **Registry System**:
   - Implemented `PromptFormat` enum for categorizing prompts
   - Created registry decorator for easy function registration
   - Added `get_prompt()` function for unified access to all prompts
   - Added fallback behavior for format preferences

3. **Backward Compatibility**:
   - Updated legacy files to import from new package
   - Added deprecation warnings to guide future migrations
   - Maintained function aliases for renamed functions
   - Created tests to verify compatibility

4. **Testing**:
   - Added tests for the new package structure
   - Added tests for backward compatibility
   - Ensured all tests pass with the new system

## Benefits of the New Structure

1. **Reduced Duplication**: Functions now have a single canonical implementation
2. **Consistent Naming**: Clear naming conventions like `task_zeroshot_prompt`
3. **Better Discoverability**: Central registry makes it easy to see all available prompts
4. **Type Safety**: Added type hints for better IDE support and error checking
5. **Documentation**: Added docstrings to explain function purpose and parameters
6. **Extensibility**: Easy to add new prompt formats or variants
7. **Maintainability**: Easier to update and improve prompts in one place

## Recommended Next Steps

1. **Continue Migration**: Move remaining prompts to the appropriate files
   - Migrate more zero-shot prompts to `zeroshot.py`
   - Implement key few-shot prompts in `fewshot.py`
   - Move default prompts to `base.py`

2. **Standardize Parameters**:
   - Ensure consistent parameter names across functions
   - Add proper error handling for special cases
   - Standardize return types and formats

3. **Update Task Files**:
   - Update task files to use the new registry API
   - Replace direct imports with `get_prompt()` function calls
   - Add documentation about the preferred usage pattern

4. **Documentation**:
   - Create user guide for the prompt system
   - Add examples of how to use the registry
   - Document the migration path for any remaining legacy code

## Usage Example

```python
# Old way - direct import from specific file
from flame.code.prompts_zeroshot import fpb_zeroshot_prompt
result = fpb_zeroshot_prompt(input_text)

# New way - use registry
from flame.code.prompts import get_prompt, PromptFormat
prompt_func = get_prompt("fpb", PromptFormat.ZERO_SHOT)
result = prompt_func(input_text)
```

This new approach offers a more flexible, maintainable, and discoverable prompt system that will make it easier to add and improve prompts in the future.