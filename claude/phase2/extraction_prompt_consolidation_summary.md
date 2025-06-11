# Extraction Prompt Consolidation Summary

## Current Status (Phase 1 & 2 In Progress)

### ✅ Registry Integration Complete (Phase 1)
- Successfully registered 15+ extraction prompts in unified registry using `@register_prompt(task, PromptFormat.EXTRACTION)` decorators
- All centralized functions in `extraction_prompts.py` now registered
- Verified registry functionality works: `get_prompt("fpb", PromptFormat.EXTRACTION)`

### 🔄 Direct Migration In Progress (Phase 2)
- **Migrated**: `fpb`, `headlines`, `refind`, `fomc` ✅ (all local functions removed)
- **Centralized**: `extraction_prompts.py` with 22 registered functions ✅
- **Fixed**: Import issue resolved - added `from .. import extraction_prompts` to prompts/__init__.py ✅
- **Tested**: Added extraction prompt tests to test_prompt_registry.py ✅
- **Remaining**: 13+ local `extraction_prompt` functions to migrate ⏳
- **Already Good**: `fnxl_evaluate.py` (imports from extraction_prompts.py) ✅

## Next Steps: Direct Migration (No Migration Script Needed)

### Phase 2: Systematic Module Updates
For each evaluation module with local `extraction_prompt` function:

1. **Add registry import**:
   ```python
   from flame.code.prompts.registry import get_prompt, PromptFormat
   ```

2. **Replace function calls**:
   ```python
   # Old:
   extraction_prompt(response)
   
   # New:
   extraction_func = get_prompt("task_name", PromptFormat.EXTRACTION)
   extraction_func(response)
   ```

3. **Delete local function** (reduce duplication)

### 🎯 Migration Priority List

**✅ Completed Migrations (16 total):**
- `fpb/fpb_evaluate.py` ✅
- `fomc/fomc_evaluate.py` ✅ 
- `headlines/headlines_evaluate.py` ✅
- `refind/refind_evaluate.py` ✅
- `banking77/banking77_evaluate.py` ✅
- `causal_classification/causal_classification_evaluate.py` ✅
- `causal_detection/casual_detection_evaluate_llm.py` ✅
- `finred/finred_evaluate.py` ✅ (Fixed "NO-REL" vs "No Relationship" mismatch)
- `finbench/finbench_evaluate.py` ✅
- `numclaim/numclaim_evaluate.py` ✅
- `finqa/finqa_evaluate.py` ✅ (Uses generic 'qa' extraction prompt)
- `convfinqa/convfinqa_evaluate.py` ✅ (Added new prompt to registry)
- `fnxl/fnxl_evaluate.py` ✅ (Migrated from direct import to registry)
- `subjectiveqa/subjectiveqa_evaluate.py` ✅ (Handles extra parameter)
- `finer/finer_evaluate.py` ✅ (Had different function name)
- `fiqa/fiqa_task1_evaluate.py` ✅

## Goals
- **Simplify codebase** ✅
- **Reduce duplication** ✅ 
- **Single source of truth** ✅
- **No backwards compatibility cruft** ✅

## End State
- Zero local `extraction_prompt` functions
- 100% registry usage via `get_prompt(task, PromptFormat.EXTRACTION)`
- All extraction logic centralized in `extraction_prompts.py`