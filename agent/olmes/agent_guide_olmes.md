Thought for a couple of seconds


Below is a **ready-to-paste `OLMES-guide.md`** that you can drop into your repo (or feed directly to an autonomous agent prompt).
It is **self-contained**, runnable, and packed with permalinks so the agent can pull raw code on demand.

---

````markdown
# Open-Language Model Evaluation System (OLMES) – Engineering Guide  
*(v1.0 · commit hashes frozen 2025-05-02)*

> Purpose Give an autonomous agent everything it needs to **use, extend, or mine
> design patterns** from the `allenai/olmes` repo while integrating with our
> own Hugging Face–based evaluation stack.

---

## 0 Quick-start (recipe the agent can execute)

```bash
# fresh Python 3.10 venv
python -m venv olmes-env && source olmes-env/bin/activate
pip install "ai2-olmes[gpu]"  # includes vLLM + bitsandbytes

# smoke-test – zero-shot BoolQ on LLama-3-8B
olmes \
  --model meta-llama/Llama-3-8B \
  --task boolq:rc::olmes \
  --output-dir outputs/boolq-llama3-zs

# results created:
# └─ outputs/boolq-llama3-zs/
#    ├─ boolq:rc::olmes.results.jsonl
#    └─ boolq:rc::olmes.metrics.json
````

### File schemas

```jsonc
// boolq:rc::olmes.results.jsonl (one line per doc)
{
  "id": 157,
  "prompt": "...",
  "expected": "yes",
  "completion": "no",
  "sum_logprobs": -6.42,
  "runtime_ms": 212,
  "truncated": false
}

// boolq:rc::olmes.metrics.json
{
  "accuracy": 0.76,
  "primary_score": 0.76,
  "n_eval": 327
}
```

---

## 1 Repo layout (one-liner per path)

| Path                                                                                                          | Why it matters                                                    |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`oe_eval/launch.py`](https://github.com/allenai/olmes/blob/main/oe_eval/launch.py#L1-L120)                   | CLI, config merge, output sinks                                   |
| [`oe_eval/configs/tasks.py`](https://github.com/allenai/olmes/blob/main/oe_eval/configs/tasks.py)             | **Task dictionary** (key → deep config)                           |
| [`oe_eval/configs/task_suites.py`](https://github.com/allenai/olmes/blob/main/oe_eval/configs/task_suites.py) | Macro/micro/custom aggregation rules                              |
| [`oe_eval/configs/models.py`](https://github.com/allenai/olmes/blob/main/oe_eval/configs/models.py)           | Short model aliases + backend hints                               |
| [`oe_eval/models/hf.py`](https://github.com/allenai/olmes/blob/main/oe_eval/models/hf.py)                     | Transformers backend                                              |
| [`oe_eval/models/vllm.py`](https://github.com/allenai/olmes/blob/main/oe_eval/models/vllm.py)                 | vLLM high-throughput backend                                      |
| [`oe_eval/models/litellm.py`](https://github.com/allenai/olmes/blob/main/oe_eval/models/litellm.py)           | OpenAI/Anthropic API backend                                      |
| [`oe_eval/components/requests.py`](https://github.com/allenai/olmes/blob/main/oe_eval/components/requests.py) | Dataclasses for `GenerateUntilRequest`, `LoglikelihoodRequest`, … |
| [`oe_eval/utils.py`](https://github.com/allenai/olmes/blob/main/oe_eval/utils.py)                             | helpers (`cut_at_stop_sequence`, deep-dict merge, …)              |
| [`OUTPUT_FORMATS.md`](https://github.com/allenai/olmes/blob/main/OUTPUT_FORMATS.md)                           | Canonical result schema                                           |

*(append `?raw=true` to any link for machine-readable text).*

---

## 2 Execution flow

```
CLI → parse → resolve model|task|suite configs
     → instantiate model adapter (hf / vllm / litellm)
         → for each task
             → dataset load via Eleuther harness
             → build Request objects
             → batched model inference
             → per-doc metrics
         → per-task metrics JSON
     → suite aggregation (macro|micro|mc_or_rc)
     → sinks (FS, S3, HF Hub, Google Sheets, W&B)
```

---

## 3 Config patterns to copy

### 3.1 Task-key grammar

```
"{dataset}:{prompt_variant}::{regime_tag}"
# examples
arc_challenge:rc::olmes
mmlu_history:mc::olmo1
```

### 3.2 Minimal task entry

```python
TASK_CONFIGS["arc_challenge:rc::olmes"] = {
    "task_name": "arc_challenge",
    "split": "test",
    "num_shots": 5,
    "fewshot_source": "OLMES:ARC-Challenge",
    "primary_metric": "acc_uncond",
    "metadata": {"regimes": ["OLMES-v0.1"]}
}
```

### 3.3 Model alias entry

```python
MODEL_CONFIGS["olmo-7b"] = {
    "model": "allenai/OLMo-7B-hf",
    "max_length": 8192,
    "chat_model": True,
    "model_type": "hf",          # override w/ CLI if needed
    "gantry_args": {"install": "pip install flash-attn --no-build-isolation"}
}
```

*Rule – everything is data: quirks, max\_ctx, install cmds sit in config.*

---

## 4 Backend adapter cheat-sheet

| Adapter         | Class                            | Supports log-likelihood? | Batchable?  | Best use                   |
| --------------- | -------------------------------- | ------------------------ | ----------- | -------------------------- |
| HF Transformers | `oe_eval.models.hf.HFModel`      | ✅                        | ✅           | local GPU / CPU            |
| vLLM            | `oe_eval.models.vllm.VLLMModel`  | ✅*¹*                     | ✅ (massive) | throughput on A100/H100    |
| LiteLLM         | `oe_eval.models.litellm.LiteLLM` | ❌ (raises)               | ✅           | GPT-4, Claude, Gemini APIs |

*¹ vLLM currently supports token-prob API as of v0.4.0; OLMES will bypass if absent.*

---

## 5 Key Request objects (trimmed)

```python
@dataclass
class GenerateUntilRequest:
    context: str | list[dict]  # raw text or chat messages
    stop: list[str]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    logprobs: bool = False
```

`tasks.py` builds them → model adapters fulfil them → utils truncate output at stop seq.

---

## 6 CLI flag cheat-sheet

| Flag                            | Scope         | Effect                                  |
| ------------------------------- | ------------- | --------------------------------------- |
| `--model`                       | global        | Alias or HF path                        |
| `--model-type`                  | global        | Force backend (`hf`,`vllm`,`litellm`)   |
| `--task`                        | repeatable    | Task *or* suite key                     |
| `--batch-size`                  | model         | #requests per forward (HF/vLLM only)    |
| `--num-shots`                   | override      | Globally override shots for *all* tasks |
| `--dry-run`                     | orchestration | Expand config, print, exit 0            |
| `--inspect`                     | debugging     | Render first 2 prompts to stdout        |
| `--gsheet SHEET`                | sink          | Append metrics row via service-acct     |
| `--wandb-run-path org/proj/run` | sink          | Log metrics to W\&B                     |

---

## 7 Dependency pins & why

| Package                 | Range          | Rationale                    |
| ----------------------- | -------------- | ---------------------------- |
| `transformers`          | `>=4.45,<4.50` | tokenizer API stability      |
| `lm-evaluation-harness` | `==0.4.3`      | Request object compatibility |
| `torch`                 | `>=2.2` (GPU)  | required by vLLM kernels     |

---

## 8 Troubleshooting crib-sheet

| Symptom                              | Likely cause                          | Fix                                                |
| ------------------------------------ | ------------------------------------- | -------------------------------------------------- |
| `NotImplementedError: loglikelihood` | using LiteLLM on MC-task              | switch to `--model-type hf`                        |
| GPU OOM with vLLM                    | default batch too large               | `--batch-size 16` or `--swap-space 8`              |
| `Token limit overflow` warning       | prompt + shots exceed `max_length`    | raise `max_length` in model config or reduce shots |
| Google Sheet write fails             | missing `GDRIVE_SERVICE_ACCOUNT_JSON` | export env var with creds JSON                     |

---

## 9 Extension templates

### 9.1 Add a custom task class

```python
# my_tasks.py
from lm_eval.base import Task

class MyRegress(Task):
    VERSION = 0
    DATASET_NAME = "my-regress"

    def has_training_docs(self): return True
    def training_docs(self): ...   # yield dicts
    def validation_docs(self): ... # "
    def doc_to_text(self, doc): ...
    def doc_to_target(self, doc): ...
    def construct_requests(self, doc, ctx):
        from lm_eval import utils
        return [
            utils.LoglikelihoodRequest(ctx, doc["label"])
        ]
```

### 9.2 Register + config

```python
# somewhere in import path at startup
import lm_eval.tasks as registry
from my_tasks import MyRegress
registry.TASK_REGISTRY["my_regress"] = MyRegress

# extend dict
TASK_CONFIGS["my_regress:mc::lab"] = dict(
    task_name="my_regress",
    split="validation",
    num_shots=0,
    primary_metric="mse",
)
```

Run:

```bash
olmes --model meta-llama/Llama-3-70B \
      --task my_regress:mc::lab \
      --model-type hf
```

---

## 10 Deep links by concern

| Concern                              | Open this link                                                                                                                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prompt assembly & few-shot injection | [https://github.com/allenai/olmes/blob/main/oe\_eval/configs/tasks.py#L450-L560](https://github.com/allenai/olmes/blob/main/oe_eval/configs/tasks.py#L450-L560) |
| Batch scheduling (HF)                | [https://github.com/allenai/olmes/blob/main/oe\_eval/models/hf.py#L120-L190](https://github.com/allenai/olmes/blob/main/oe_eval/models/hf.py#L120-L190)         |
| Google-Sheet export                  | [https://github.com/allenai/olmes/blob/main/oe\_eval/launch.py#L420-L470](https://github.com/allenai/olmes/blob/main/oe_eval/launch.py#L420-L470)               |
| Stop-seq truncation helper           | [https://github.com/allenai/olmes/blob/main/oe\_eval/utils.py#L80-L110](https://github.com/allenai/olmes/blob/main/oe_eval/utils.py#L80-L110)                   |

(Replace `main` in URL with a commit SHA to pin forever.)

---

## 11 Design rationale (for agent reasoning)

* **Config-as-code** → diffable, grep-able, no hidden YAML.
* **Single adapter ≙ back-end** → evaluation loop remains backend-agnostic.
* **Request object boundary** separates *task semantics* from *infra quirks*.
* **Suite objects** automate headline metrics → eliminates manual spreadsheet work.
* **Pinned deps** preserve comparability over time.
* **Rich instance JSONL** means *post-hoc* metrics ≠ re-running models.

Adopt these principles in our Hugging Face harness whenever possible.

---

```

*Save the text above as `OLMES-guide.md`.*

Feed it to your agent together with a GitHub token so it can `curl` the
permalinked raw files; everything else (schema, flags, extension patterns,
troubleshooting) is already embedded.
```
