# FERRArI: Financial Economics Reasoning Refinement for Artificial Intelligence

Glenn Matlin, Kaushik Arcot, Thaneesh B. Krishnasamy, Neeraj Menon Suresh Kumar

Corresponding author: `glennmatlin [at] gatech.edu`

## Tasks

1. **FOMC**: Federal Reserve statement classification (Hawkish/Dovish/Neutral)
1. **EconLogicQA**: Ordering economic events based on logical sequences
1. **FiQA**: Financial sentiment analysis and opinion-based QA
   - Task 1: Target-specific sentiment analysis
   - Task 2: Financial opinion question answering
1. **MMLU**: Massive Multitask Language Understanding (Economics focus)
1. **BizBench**: Numerical question answering on SEC filings

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## Usage

The pipeline is split into two stages: inference and evaluation.

### 1. Inference Stage

Generate model responses for any supported task:

```bash
# Using config file
python main.py --config configs/task_name.yaml --mode inference

# Or with explicit arguments
python main.py \
    --mode inference \
    --dataset [bizbench|econlogicqa|fiqa_task1|fiqa_task2|fomc|mmlu] \
    --model together_ai/meta-llama/Llama-2-7b \
    --batch_size 10 \
    --temperature 0.0 \
    --top_p 0.9
```

Task-specific arguments:
- For MMLU:
  ```bash
  --mmlu-subjects econometrics high_school_macroeconomics \
  --mmlu-split test \
  --mmlu-num-few-shot 5
  ```

### 2. Evaluation Stage

Evaluate the model's responses:

```bash
python main.py \
    --mode evaluate \
    --dataset [bizbench|econlogicqa|fiqa_task1|fiqa_task2|fomc|mmlu] \
    --model together_ai/meta-llama/Llama-2-7b \
    --file_name path/to/inference_results.csv
```

## Task Details

### BizBench
- Purpose: Extract numerical answers from SEC filings
- Input: Question and SEC filing context
- Output: Numerical answer without units

### EconLogicQA
- Purpose: Order economic events logically
- Input: Question and 4 events
- Output: Ordered sequence of events with explanation

### FiQA
#### Task 1: Sentiment Analysis
- Purpose: Target-specific financial sentiment analysis
- Input: Financial text
- Output: Sentiment scores (-1 to 1) for identified targets

#### Task 2: Opinion QA
- Purpose: Answer opinion-based financial questions
- Input: Financial question
- Output: Answer based on financial opinions and analysis

### FOMC
- Purpose: Classify Federal Reserve statements
- Input: FOMC statement
- Output: HAWKISH/DOVISH/NEUTRAL classification

### MMLU (Economics Focus)
- Purpose: Test model's economics knowledge
- Input: Multiple-choice questions
- Output: Answer with explanation
- Supported subjects: Economics, Finance, Accounting, etc.

## Configuration

Each task has a corresponding config file in `configs/`:
- `bizbench.yaml`
- `econlogicqa.yaml`
- `fiqa.yaml`
- `fomc.yaml`
- `mmlu.yaml`

Configure:
- Model parameters (temperature, top_p, etc.)
- Task-specific settings
- Batch size and other inference settings

## Output Structure

### Inference Results
```
output/results/
├── bizbench/
├── econlogicqa/
├── fiqa/
│   ├── task1/
│   └── task2/
├── fomc/
└── mmlu/
```

### Evaluation Results
```
output/evaluation/
├── bizbench/
├── econlogicqa/
├── fiqa/
│   ├── task1/
│   └── task2/
├── fomc/
└── mmlu/
```

Each directory contains:
- `inference_{model}_{date}.csv`: Raw model responses
- `evaluation_{model}_{date}.csv`: Detailed results
- `evaluation_{model}_{date}_metrics.csv`: Task-specific metrics

## Logging

Logs are saved to `logs/` with task-specific log files:
- `bizbench_[inference|evaluation].log`
- `econlogicqa_[inference|evaluation].log`
- `fiqa_task[1|2]_[inference|evaluation].log`
- `fomc_[inference|evaluation].log`
- `mmlu_[inference|evaluation].log`
