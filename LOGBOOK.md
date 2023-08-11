https://github.com/facebookresearch/llama-recipes/tree/main/inference

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

  
## Research Questions

- How is performance on financial tasks affected by pruning or parameters reduction
- few shot learning (sampling sentences for few shot)
  - how do i sample few shot from different regions
  - region aware sampling
  - HOW DO WE DO FEW SHOT LEARNING IN THE KNOWLEDGE OF FINANCE
  - keyword aware sampling how does this LLM perform
- what is the number of shots we need?
- how can we pick the best shots?
- goal is to use zero/few shot on fomc data like trillion dollar words paper (correlate with CPI data on hawk/dove)
  - hawk/dove isnt a task its just to build a measure

- normal float 4
- nested quantization
- mixed precision
- Play with llm_int8_threshold
- skip quantization of specific modules\layers
- instruction based fine tuning with quantized models

## Journal

### Tuesday, 2023-08-08

### Llama2
```bash
python src/scripts/download_llama.py -m meta-llama/Llama-2-7b-chat-hf -hf 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 ; \
python src/scripts/download_llama.py -m meta-llama/Llama-2-13b-chat-hf -hf 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 ; \
python src/scripts/download_llama.py -m meta-llama/Llama-2-70b-chat-hf -hf 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
```

```bash
python src/scripts/run_llama.py -q fp16 -t sentiment_analysis -m meta-llama/Llama-2-7b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q fp16 -t sentiment_analysis -m meta-llama/Llama-2-13b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q fp16 -t sentiment_analysis -m meta-llama/Llama-2-70b-chat-hf -hf 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

python src/scripts/run_llama.py -q bf16 -t sentiment_analysis -m meta-llama/Llama-2-7b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q bf16 -t sentiment_analysis -m meta-llama/Llama-2-13b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q bf16 -t sentiment_analysis -m meta-llama/Llama-2-70b-chat-hf -hf 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

python src/scripts/run_llama.py -q int8 -t sentiment_analysis -m meta-llama/Llama-2-7b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q int8 -t sentiment_analysis -m meta-llama/Llama-2-13b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q int8 -t sentiment_analysis -m meta-llama/Llama-2-70b-chat-hf -hf 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

python src/scripts/run_llama.py -q fp4 -t sentiment_analysis -m meta-llama/Llama-2-7b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q fp4 -t sentiment_analysis -m meta-llama/Llama-2-13b-chat-hf -hf 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 ; \
python src/scripts/run_llama.py -q fp4 -t sentiment_analysis -m meta-llama/Llama-2-70b-chat-hf -hf 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
```

```
-q fp16 -m meta-llama/Llama-2-7b-chat-hf [DONE on 10.138.66.5]
-q fp16 -m meta-llama/Llama-2-13b-chat-hf [DONE on 10.138.66.5]
-q fp16 -m meta-llama/Llama-2-70b-chat-hf [INCOMPLETE on 10.138.66.5]

-q bf16 -m meta-llama/Llama-2-7b-chat-hf [DONE on 10.138.66.6]
-q bf16 -m meta-llama/Llama-2-13b-chat-hf [DONE on 10.138.66.6]
-q bf16 -m meta-llama/Llama-2-70b-chat-hf [INCOMPLETE on 10.138.66.5]

-q int8 -m meta-llama/Llama-2-7b-chat-hf [DONE on 10.138.66.6]
-q int8 -m meta-llama/Llama-2-13b-chat-hf [WIP on 10.138.66.6]
-q int8 -m meta-llama/Llama-2-70b-chat-hf [QUEUED on 10.138.66.6]

-q fp4 -m meta-llama/Llama-2-7b-chat-hf [DONE on 10.138.66.6]
-q fp4 -m meta-llama/Llama-2-13b-chat-hf [DONE on 10.138.66.6]
-q fp4 -m meta-llama/Llama-2-70b-chat-hf [DONE on 10.138.66.6]
```

python src/scripts/run_llama.py -q fp4 -t sentiment_analysis -m meta-llama/Llama-2-7b-chat-hf
python src/scripts/run_llama.py -q fp4 -t sentiment_analysis -m meta-llama/Llama-2-13b-chat-hf
python src/scripts/run_llama.py -q fp4 -t sentiment_analysis -m meta-llama/Llama-2-70b-chat-hf


#### Prompt Template

##### Special Tokens in Llama2 Foundation Model
- `<s>`: the beginning of the entire sequence.
  - NOTE: When using the `LlamaTokenizer()` from HuggingFace, the `<s>` token is automatically added to the beginning of the sequence.
- `<<SYS>>\n`: the beginning of the system message.
- `\n<</SYS>>\n\n`: the end of the system message.
- `[INST]`: the beginning of some instructions.
- `[/INST]`: the end of some instructions.

##### Examples of Llama2's Prompt Template

Template:
"""
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
"""

Example 1:
"""
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

There's a llama in my garden. What should I do? [/INST]
"""

Example 2:
"""
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always do...

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

How many Llamas are we from skynet? [/INST]
"""

##### Alternative to Llama2's Prompt Template
Alternative Template:
"""
<<SYS>>
{{ system_prompt }}
<</SYS>>

[INST] {{instruction_prompt}} [/INST] {{ user_message }}
"""

Alternative Example 1:
"""
<<SYS>>
You are a helpful, respectful and honest assistant. Always do...

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

[INST] Remember you are an assistant [/INST] User: How many Llamas are we from skynet?
"""

## Useful Links

### Torch
- https://pytorch.org/docs/stable/notes/cuda.html

### Transformers
- https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
- https://huggingface.co/docs/transformers/perf_infer_gpu_one
- https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/pipelines#transformers.pipeline

### Accelerate
- https://huggingface.co/docs/accelerate/usage_guides/big_modeling
- https://huggingface.co/docs/accelerate/package_reference/big_modeling
- https://huggingface.co/docs/accelerate/usage_guides/training_zoo
- https://github.com/huggingface/accelerate/blob/v0.21.0/src/accelerate/big_modeling.py#L394
### Quantization
- https://huggingface.co/docs/transformers/main_classes/quantization
- https://huggingface.co/TheBloke/Llama-2-70B-fp16
#### BNB
- https://huggingface.co/blog/4bit-transformers-bitsandbytes
- https://github.com/TimDettmers/bitsandbytes
#### QLORA
- https://github.com/artidoro/qlora
#### GPTQ
- https://github.com/PanQiWei/AutoGPTQ
- https://huggingface.co/TheBloke/Llama-2-70B-GPTQ
#### other quantized llms
- https://huggingface.co/TheBloke

### PEFT
- https://github.com/huggingface/peft

### Llama
- https://huggingface.co/docs/transformers/main/model_doc/llama2
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama
- https://github.com/huggingface/transformers/tree/v4.31.0/src/transformers/pipelines
- https://gist.github.com/viniciusarruda/ef463e9e04e2a221710a72d978d604c3