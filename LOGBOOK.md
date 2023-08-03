# Research Logbook
***SUBMITTAL DEADLINE: DECEMBER 15TH 2023 (OR) JANUARY 15TH 2024***
***FINAL RESULTS DEADLINE: NOVEMBER 15TH deadline to work with agam***

october 15th results on dolly, falon, llama across quantization levels, with specific sentences we fail on

AUGUST 11th: llama 2 results on 4 tasks and different quantization levels

## TODOs

### Inbox

### High Impact, High Urgency
* [ ] Move my fork to the private lab repo
* [ ] GET LLAMA 2 WORKING - only do work with llama, falcon, dolly

### High Impact, Low Urgency
* [ ] Run the models on the sentences being done on less agreement (50/66/75)
* [ ] Get cluster access for GPUs (ETA?)
* [ ] Formulate clear research questions about how quantization causes failures on outliers in the genLLM results

i.e. If we quantize the model does it do worse on specific sentences in this dataset and which kind does quantizated models fail on outliers? if the quantmodel can only get the common stuff then its not as good. what kind of sentences do quantized models do worse on?

### Low Impact, High Urgency
* [ ] Get the data for numclaim from Agam (its not public yet)

### Low Impact, Low Urgency
* [ ] For dolly, give a control flow to separate the way "finer_ord" is handled compared to the other tasks
* [ ] Use `get_financial_phrasebank()` to download the dataset from HuggingFace for the project
* [ ] Replace all the dashes with underscores e.g. FPB-sentiment-analysis-allagree with FPB_sentiment_analysis_allagree
* [ ] Convert data used for "finer_ord" from `.csv.` to `.xlsx` so I can use `pd.read_xlsx` instead

## Asks and Answers

### Asks

### Answers
* [X] Discuss how `instruct_pipeline.py` works ... I could be wrong but it seems like it is pushing through all 453 at once not batching/one-at-a-time? // Actually I think it is doing it one-at-a-time using the Pipeline since there is not a DataLoader provided.
ANSWER: yea its one by one but its needed because we dont want to prompt in order and taint the results

* [X] Is the numclaim data is just the same as the sentiment analysis data ... there is no data folder for numclaims
ANSWER: NO IT IS DIFFERENT

* [X] What is the deal with `lab-manual-split-combine-test` for FOMC data, 'test.csv' for finder_ord
ANSWER: check the papers original for the data


## Useful Links

### Torch
- https://pytorch.org/docs/stable/notes/cuda.html

### Transformers
- https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
- https://huggingface.co/docs/transformers/perf_infer_gpu_one
- https://huggingface.co/docs/transformers/main_classes/quantization

### Quantization
- https://github.com/TimDettmers/bitsandbytes
- https://github.com/huggingface/peft
- https://github.com/artidoro/qlora

## Journal

### Monday, 2023-07-31

#### Actions

#### Notes:
 - focus on memory gpu utilization with quantization
 - parameters reduction can be done by pruning
 - Agam's paper
   - few shot learning (sampling sentences for few shot)
     - how do i sample few shot from different regions
     - region aware sampling
     - HOW DO WE DO FEW SHOT LEARNING IN THE KNOWLEDGE OF FINANCE
     - keyword aware sampling how does this LLM perform
   - what is the number of shots we need?
   - how can we pick the best shots?
   - goal is to use zero/few shot on fomc data like trillion dollar words paper (correlate with CPI data on hawk/dove)
     - hawk/dove isnt a task its just to build a measure

### Friday, 2023-07-28
- Could using `.batch_decode()` instead of `.decode()` in the instruction pipeline speed things up?
- Runtimes are found in the original research. There were 453 sentences in the test set for sentiment classification. Reported inference time was ~7s each so $453 \times 7 \times \frac{1}{60} =$ 52.85 minutes for completion -- which is similar to what I was getting!

## Thursday, 2023-07-27
- turned off device mapping for inference and instead set to use one GPU for now
- dolly model inference is working with quantization and batching
- utils.py broken down into cuda_utils and model_utils