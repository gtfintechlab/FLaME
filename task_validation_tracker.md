# FLaME Task Validation Tracker

This document tracks the progress of running live inference and evaluation for all tasks in the FLaME framework.

## Overview
- Total Tasks: 24
- Model: together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct
- Date Started: January 24, 2025

## Status Legend
- ⬜ Not Started
- 🔄 In Progress
- ✅ Completed
- ❌ Failed
- ⚠️ Issues/Warnings

## Task Progress

### 1. banking77
- **Config File**: `configs/banking77.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 2. bizbench
- **Config File**: `configs/bizbench.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 3. causal_classification
- **Config File**: `configs/causal_classification.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 4. causal_detection
- **Config File**: `configs/causal_detection.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 5. convfinqa
- **Config File**: `configs/convfinqa.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 6. econlogicqa
- **Config File**: `configs/econlogicqa.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: No evaluation function in task_registry
- **Results Path**: 
- **Evaluation Path**: N/A

### 7. ectsum
- **Config File**: `configs/ectsum.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 8. edtsum
- **Config File**: `configs/edtsum.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 9. finbench
- **Config File**: `configs/finbench.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 10. finentity
- **Config File**: `configs/finentity.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 11. finer
- **Config File**: `configs/finer.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 12. finqa
- **Config File**: `configs/finqa.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 13. finred
- **Config File**: `configs/finred.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: No evaluation function in task_registry
- **Results Path**: 
- **Evaluation Path**: N/A

### 14. fiqa_task1
- **Config File**: `configs/fiqa_task1.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 15. fiqa_task2
- **Config File**: `configs/fiqa_task2.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 16. fnxl
- **Config File**: `configs/fnxl.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 17. fomc
- **Config File**: `configs/fomc.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 18. fpb
- **Config File**: `configs/fpb.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 19. headlines
- **Config File**: `configs/headlines.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 20. mmlu
- **Config File**: `configs/mmlu.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 21. numclaim
- **Config File**: `configs/numclaim.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 22. refind
- **Config File**: `configs/refind.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 23. subjectiveqa
- **Config File**: `configs/subjectiveqa.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

### 24. tatqa
- **Config File**: `configs/tatqa.yaml`
- **Inference Status**: ⬜ Not Started
- **Evaluation Status**: ⬜ Not Started
- **Notes**: 
- **Results Path**: 
- **Evaluation Path**: 

## Summary Statistics
- **Total Tasks**: 24
- **Tasks with Both Inference & Evaluation**: 22
- **Tasks with Inference Only**: 2 (econlogicqa, finred)
- **Completed Inference**: 0/24 (0%)
- **Completed Evaluation**: 0/22 (0%)

## Command Templates

### Inference Command
```bash
uv run python main.py --config configs/<task>.yaml --mode inference --dataset <task> --model "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"
```

### Evaluation Command
```bash
uv run python main.py --config configs/<task>.yaml --mode evaluate --dataset <task> --file_name "<results_file_path>"
```

## Notes and Observations
- econlogicqa and finred do not have evaluation functions registered in task_registry.py
- Each task will generate results in `results/<task>/` directory
- Evaluation results will be saved in `evaluations/<task>/` directory
- We should monitor API usage and rate limits during the validation process
- Consider running tasks in batches to avoid overwhelming the API

## Next Steps
1. Confirm API keys are properly configured
2. Start with a small task to validate the process
3. Monitor for any errors or issues
4. Document any task-specific requirements or quirks
5. Update this tracker as we progress through each task