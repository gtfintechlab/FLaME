# Comprehensive Task Review Plan

## Objective
Systematically review all 24 inference tasks and 22 evaluation tasks to ensure they are working correctly after our changes.

## Review Strategy

For each task, we will check:
1. **Imports**: Task can be imported without errors
2. **Prompts**: Prompts are registered and working
3. **Data Loading**: Dataset loading works (or is properly mocked)
4. **Inference**: Inference function executes without errors
5. **Evaluation**: Evaluation function executes without errors (if applicable)
6. **Integration**: Task works in multi-task scenarios

## Task Review Checklist

### Group 1: Core Financial Tasks (HIGH PRIORITY)
- [x] **fomc** - Federal Open Market Committee sentiment analysis ✓
  - [x] Check inference ✓
  - [x] Check evaluation ✓
  - [x] Verify prompts ✓
- [x] **numclaim** - Numerical claim classification ✓
  - [x] Check inference ✓
  - [x] Check evaluation ✓
  - [x] Verify prompts ✓
- [x] **fpb** - Financial Phrase Bank ✓
  - [x] Check inference ✓
  - [x] Check evaluation ✓
  - [x] Verify prompts ✓
- [x] **finentity** - Financial entity extraction ✓
  - [x] Check inference ✓
  - [x] Check evaluation ✓
  - [x] Verify prompts ✓
- [x] **finer** - Financial named entity recognition ✓
  - [x] Check inference ✓
  - [x] Check evaluation ✓
  - [x] Verify prompts ✓

### Group 2: Question Answering Tasks
- [ ] **finqa** - Financial question answering
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **convfinqa** - Conversational financial QA
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **tatqa** - Table and text question answering
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **subjectiveqa** - Subjective question answering
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **econlogicqa** - Economic logic QA
  - [ ] Check inference only (no evaluation)
  - [ ] Verify prompts

### Group 3: Summarization Tasks
- [ ] **ectsum** - Earnings call transcript summarization
  - [ ] Check inference
  - [ ] Check evaluation (with BERTScore)
  - [ ] Verify prompts
- [ ] **edtsum** - Event-driven temporal summarization
  - [ ] Check inference
  - [ ] Check evaluation (with BERTScore)
  - [ ] Verify prompts

### Group 4: Classification Tasks
- [ ] **causal_classification** - Causal relationship classification
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **causal_detection** - Causal relation detection
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Note: filename typo issue
- [ ] **banking77** - Banking intent classification
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **headlines** - News headlines classification
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts

### Group 5: Specialized Tasks
- [ ] **fnxl** - Financial document extraction
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **finred** - Financial relation extraction
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **refind** - Financial document relation extraction
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **fiqa_task1** - FiQA sentiment analysis
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **fiqa_task2** - FiQA aspect-based sentiment
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts

### Group 6: Benchmark Tasks
- [ ] **bizbench** - Business benchmark
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **finbench** - Financial benchmark
  - [ ] Check inference
  - [ ] Check evaluation
  - [ ] Verify prompts
- [ ] **mmlu** - Massive Multitask Language Understanding
  - [ ] Check inference (heavy deps - may skip)
  - [ ] Check evaluation
  - [ ] Verify prompts

## Review Process

### For Each Task:
1. **Import Test**
   ```python
   from flame.code.<task>.<task>_inference import <task>_inference
   from flame.code.<task>.<task>_evaluate import <task>_evaluate  # if exists
   ```

2. **Prompt Test**
   ```python
   from flame.code.prompts import get_prompt, PromptFormat
   prompt = get_prompt("<task>", PromptFormat.ZERO_SHOT)
   assert prompt is not None
   ```

3. **Quick Function Test**
   - Check function signatures
   - Review any special requirements
   - Note any issues or dependencies

4. **Integration Test**
   ```bash
   uv run python main.py --tasks <task> --mode inference --model dummy
   ```

## Tracking Progress

We'll go through each group systematically, testing and fixing issues as we find them. 

### Priority Order:
1. Core Financial Tasks (most important)
2. Question Answering Tasks
3. Classification Tasks
4. Summarization Tasks (check BERTScore issues)
5. Specialized Tasks
6. Benchmark Tasks (lowest priority)

## Expected Issues to Watch For:

1. **Import errors** - Missing dependencies or circular imports
2. **Prompt registration** - Some tasks might not have prompts properly registered
3. **Data loading** - Some tasks might try to download large datasets
4. **API dependencies** - Tasks that require specific API keys or services
5. **Evaluation metrics** - Some evaluation functions might have specific requirements
6. **File path issues** - Hardcoded paths that don't work in all environments

## Success Criteria

Each task should:
- Import without errors
- Have registered prompts
- Run inference without crashing (with dummy data)
- Run evaluation without crashing (with dummy data)
- Work in multi-task scenarios
- Have clear error messages for missing dependencies