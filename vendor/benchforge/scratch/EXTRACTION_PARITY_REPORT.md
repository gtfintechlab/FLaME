# BenchForge vs FLAME Extraction Methods - Parity Report

## Executive Summary

**Critical Finding**: FLAME uses **LLM-based extraction** as a fallback when simple pattern matching fails, while BenchForge currently only uses **rule-based extraction**. This is a significant gap that needs to be addressed to meet FLAME standards.

## Detailed Analysis

### FLAME's Extraction Approach (Two-Stage Process)

1. **Initial Inference**: Get raw response from the model
2. **LLM-Based Extraction** (when needed): Use a secondary LLM call with an extraction prompt to clean up messy responses

#### Evidence from FLAME Codebase:

**File: `src/flame/code/extraction_prompts.py`** (lines 195-210)
```python
@register_prompt("fomc", PromptFormat.EXTRACTION)
def fomc_extraction_prompt(llm_response: str) -> str:
    """Generate a prompt to extract the classification label from the LLM response."""
    prompt = f"""Extract the classification label from the following LLM response. 
    The label should be one of the following: 'HAWKISH', 'DOVISH', or 'NEUTRAL'.
    
    Here is the LLM response to analyze:
    "{llm_response}"
    Provide only the label that best matches the response. 
    Only output alphanumeric characters and spaces. 
    Do not include any special characters or punctuation."""
    return prompt
```

**File: `src/flame/code/fomc/fomc_evaluate.py`** (lines 176-188)
```python
# Prepare messages for batch with extraction prompt
messages_batch = [
    [
        {
            "role": "user",
            "content": get_prompt("fomc", PromptFormat.EXTRACTION)(response),
        }
    ]
    for response in response_batch
]

# Process batch with retry logic - makes actual LLM call for extraction
batch_responses = process_batch_with_retry(
    model_args, messages_batch, batch_idx, total_batches
)
```

### BenchForge's Current Approach (Single-Stage)

1. **Initial Inference**: Get raw response from the model
2. **Rule-Based Extraction**: Use regex patterns and string matching to extract labels

#### Evidence from BenchForge:

**File: `benchforge/bench_forge/flame/tasks/fomc.py`** (lines 147-196)
```python
def extract_label_from_response(self, response: str) -> Optional[str]:
    """Extract FOMC label from model response."""
    # Clean and normalize response
    response = response.strip().upper()
    
    # Look for valid labels in response (rule-based)
    for label in self.config.valid_labels:
        if label in response:
            # Check various patterns...
            
    # Try to extract from common patterns
    patterns = [
        "CLASSIFICATION:",
        "ANSWER:",
        "LABEL:",
        "SENTIMENT:",
    ]
    # ... pattern matching logic
```

## Gap Analysis

### Missing Features in BenchForge:

1. **LLM-Based Extraction Strategy**
   - FLAME can make a secondary LLM call to extract labels from messy responses
   - BenchForge only uses pattern matching and regex

2. **Extraction Prompt Registry**
   - FLAME has a dedicated extraction prompt for each task
   - BenchForge lacks this secondary prompt system

3. **Fallback Mechanism**
   - FLAME falls back to LLM extraction when initial response is unclear
   - BenchForge returns None when pattern matching fails

## Implementation Status ✅

BenchForge now has parity with FLAME's extraction capabilities:

### 1. **LLM-based extraction strategy** ✅
- Added `ExtractionStrategy.LLM_BASED` to `ResponseExtractor`
- Implemented `_extract_llm_based` method with full LLM client integration

### 2. **Extraction prompt architecture** ✅
- BenchForge provides the infrastructure for LLM-based extraction
- FLAME maintains its task-specific extraction prompts
- Clean separation of concerns: infrastructure vs. task-specific logic

### 3. **Fallback logic** ✅
- `FOMCTask.extract_label_from_response` now supports `use_llm_fallback` parameter
- Automatically falls back to LLM extraction when rule-based extraction fails

### 4. **LLM client integration** ✅
- Tasks can be initialized with an `llm_client` for extraction
- Supports multiple client interfaces (complete, chat, generic)

## Architecture Design

### BenchForge Provides:
```python
# Generic extraction infrastructure
class ResponseExtractor:
    def _extract_llm_based(self, text, llm_client, prompt_template, ...):
        # Makes LLM call with provided or default prompt
        # Handles response parsing and validation
```

### FLAME Provides:
```python
# Task-specific extraction prompts
def fomc_extraction_prompt(llm_response: str) -> str:
    # FLAME's specific prompt for FOMC task
    
class FOMCTask:
    def get_extraction_prompt(self, llm_response: str) -> str:
        # Override to use FLAME's prompt
        from flame.code.extraction_prompts import fomc_extraction_prompt
        return fomc_extraction_prompt(llm_response)
```

## Key Benefits

1. **Separation of Concerns**: BenchForge provides the engine, FLAME provides the prompts
2. **Flexibility**: Each benchmark can maintain its own extraction prompts
3. **Backward Compatibility**: Works with existing FLAME extraction prompts
4. **Extensibility**: Easy to add new extraction strategies or customize per task

## Conclusion

The key difference is that **FLAME uses language models for extraction when needed**, not just rules and regex. This is particularly important for handling:
- Verbose or chatty model responses
- Responses with explanations before the answer
- Responses with multiple mentions of labels where context matters
- Ambiguous or poorly formatted responses

BenchForge must implement this LLM-based extraction capability to meet FLAME standards.