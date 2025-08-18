# BenchForge FOMC 7-Strategy Extraction System

## Overview

The BenchForge FOMC task implements a comprehensive 7-strategy extraction system that achieves >99.6% success rate with rule-based strategies alone, and potentially near 100% with optional LLM-based fallback.

## The 7 Extraction Strategies

### Strategy 1: Direct Match
- **Description**: Checks if response is exactly the label
- **Example**: Response = "DOVISH" → Extracted = "DOVISH"
- **Success Rate**: High for well-behaved models

### Strategy 2: Classification Format
- **Description**: Looks for "Classification: LABEL" pattern
- **Example**: "Classification: HAWKISH" → "HAWKISH"
- **Regex**: `r'Classification:\s*(\w+)'`

### Strategy 3: Quoted Extraction
- **Description**: Extracts labels from quotes
- **Example**: 'The answer is "NEUTRAL"' → "NEUTRAL"
- **Handles**: Single and double quotes

### Strategy 4: Context-Based
- **Description**: Matches contextual patterns
- **Examples**:
  - "I would classify this as DOVISH"
  - "This statement is HAWKISH"
  - "The stance appears to be NEUTRAL"

### Strategy 5: Line-by-Line Search
- **Description**: Checks each line for valid labels
- **Example**: Multi-line response with label on separate line
- **Benefit**: Handles verbose responses

### Strategy 6: Case-Insensitive Fallback
- **Description**: Handles lowercase/mixed case variations
- **Example**: "hawkish" → "HAWKISH"
- **Normalization**: Converts to uppercase for matching

### Strategy 7: LLM-Based Extraction (Optional)
- **Description**: Uses a second LLM call to extract from messy responses
- **When Used**: Only if:
  1. LLM client is provided
  2. All 6 rule-based strategies failed
  3. Response seems to contain extractable information
- **Benefits**:
  - Handles extremely messy responses
  - Can interpret implied labels
  - Pushes success rate toward 100%
- **Costs**:
  - Additional API call (latency + tokens)
  - Only used as last resort

## Implementation

### Enabling LLM-Based Extraction

```python
from bench_forge.flame.tasks.fomc import FOMCTask, FOMCConfig
from bench_forge.llm.client import LLMClient

# Option 1: Provide client at initialization
config = FOMCConfig(name="fomc")
llm_client = LLMClient(your_config)
task = FOMCTask(config, llm_client=llm_client)

# Option 2: Set client after initialization
task = FOMCTask(config)
task.set_llm_client(llm_client)
```

### Without LLM-Based Extraction

```python
# Uses only 6 rule-based strategies
task = FOMCTask(config)
# Still achieves 99.6% success rate!
```

## Performance Metrics

### With 6 Rule-Based Strategies Only
- **Success Rate**: 99.6% (494/496 samples)
- **Speed**: ~0.001s per extraction
- **Cost**: Zero (no API calls)
- **Failures**: Typically incomplete responses cut off by token limits

### With 7 Strategies (Including LLM)
- **Success Rate**: ~100% (theoretical)
- **Speed**: ~0.001s normally, +1-2s for LLM fallback
- **Cost**: Additional API call only for failed extractions (~0.4%)
- **Benefit**: Handles edge cases and malformed responses

## Null/None Handling

The system preserves the option to return `None` for truly unextractable responses:

- **Returns None when**:
  - Response is empty/null
  - All 7 strategies fail
  - Response contains no label information
  
- **Benefits**:
  - Clear signal of extraction failure
  - Allows downstream error handling
  - Maintains data integrity

## Example Extraction Flow

```
Input Response: "Based on the economic indicators and inflation 
                concerns, I would classify this statement as HAWKISH 
                because it suggests tightening monetary policy."

Strategy 1 (Direct): ❌ Not exact match
Strategy 2 (Classification): ❌ No "Classification:" prefix  
Strategy 3 (Quoted): ❌ No quotes around label
Strategy 4 (Context): ✅ Matches "classify this statement as HAWKISH"
Result: "HAWKISH" extracted successfully
```

## Edge Case Handling

### Incomplete Responses
```
Response: "The statement suggests that with inflation currently below 2 percent..."
Strategies 1-6: ❌ No label found
Strategy 7 (if enabled): Makes LLM call to infer label
Result: Either extracted label or None
```

### Mixed Case
```
Response: "hawkish"
Strategies 1-5: ❌ Case mismatch
Strategy 6: ✅ Case-insensitive match
Result: "HAWKISH"
```

### Multiple Labels
```
Response: "Not DOVISH, but NEUTRAL"
Strategy 4: ✅ Finds first valid label "DOVISH"
Note: Could be improved to handle negation
```

## Configuration

### FOMCConfig Fields
- `valid_labels`: List of valid labels (DOVISH, HAWKISH, NEUTRAL)
- `text_field`: Field name for input text (default: "sentence")
- `label_field`: Field name for ground truth (default: "label")

### Extraction Settings
- Temperature for LLM extraction: 0.0 (deterministic)
- Max tokens for LLM extraction: 20 (labels are short)
- Timeout: Configurable per LLM client

## Best Practices

1. **Start without LLM**: The 6 rule-based strategies handle 99.6% of cases
2. **Add LLM for production**: Enable Strategy 7 for maximum robustness
3. **Monitor extraction rates**: Track which strategies are most effective
4. **Handle None gracefully**: Implement downstream error handling
5. **Cache LLM extractions**: Avoid repeated API calls for same responses

## Integration with FLAME

The extraction system is fully compatible with FLAME evaluation:

- **Column Names**: Uses FLAME-expected names (extracted_labels, llm_responses, etc.)
- **Fallback Support**: Stores complete_responses for evaluation-time extraction
- **Metrics**: Compatible with FLAME's accuracy and F1 calculations

## Future Improvements

1. **Negation Handling**: Detect "not LABEL" patterns
2. **Confidence Scores**: Return extraction confidence
3. **Multi-Label Support**: Handle responses with multiple labels
4. **Custom Patterns**: Allow task-specific extraction patterns
5. **Extraction Caching**: Store successful extraction patterns