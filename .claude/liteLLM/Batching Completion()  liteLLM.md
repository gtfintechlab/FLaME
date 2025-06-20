---
created: 2025-05-26T16:02:00 (UTC -04:00)
tags: []
source: https://docs.litellm.ai/docs/routing#max-parallel-requests-async
author: 
---

# Batching Completion() | liteLLM

> ## Excerpt
> LiteLLM allows you to:

---
LiteLLM allows you to:

-   Send many completion calls to 1 model
-   Send 1 completion call to many models: Return Fastest Response
-   Send 1 completion call to many models: Return All Responses

## Send multiple completion calls to 1 model[](https://docs.litellm.ai/docs/routing#send-multiple-completion-calls-to-1-model "Direct link to Send multiple completion calls to 1 model")

In the batch\_completion method, you provide a list of `messages` where each sub-list of messages is passed to `litellm.completion()`, allowing you to process multiple prompts efficiently in a single API call.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_batch_completion.ipynb)

### Example Code[](https://docs.litellm.ai/docs/routing#example-code "Direct link to Example Code")

```
import litellmimport osfrom litellm import batch_completionos.environ['ANTHROPIC_API_KEY'] = ""responses = batch_completion(    model="claude-2",    messages = [        [            {                "role": "user",                "content": "good morning? "            }        ],        [            {                "role": "user",                "content": "what's the time? "            }        ]    ])
```

## Send 1 completion call to many models: Return Fastest Response[](https://docs.litellm.ai/docs/routing#send-1-completion-call-to-many-models-return-fastest-response "Direct link to Send 1 completion call to many models: Return Fastest Response")

This makes parallel calls to the specified `models` and returns the first response

Use this to reduce latency

-   SDK
-   PROXY

### Example Code[](https://docs.litellm.ai/docs/routing#example-code-1 "Direct link to Example Code")

```
import litellmimport osfrom litellm import batch_completion_modelsos.environ['ANTHROPIC_API_KEY'] = ""os.environ['OPENAI_API_KEY'] = ""os.environ['COHERE_API_KEY'] = ""response = batch_completion_models(    models=["gpt-3.5-turbo", "claude-instant-1.2", "command-nightly"],     messages=[{"role": "user", "content": "Hey, how's it going"}])print(result)
```

### Output[](https://docs.litellm.ai/docs/routing#output "Direct link to Output")

Returns the first response in OpenAI format. Cancels other LLM API calls.

```
{  "object": "chat.completion",  "choices": [    {      "finish_reason": "stop",      "index": 0,      "message": {        "content": " I'm doing well, thanks for asking! I'm an AI assistant created by Anthropic to be helpful, harmless, and honest.",        "role": "assistant",        "logprobs": null      }    }  ],  "id": "chatcmpl-23273eed-e351-41be-a492-bafcf5cf3274",  "created": 1695154628.2076092,  "model": "command-nightly",  "usage": {    "prompt_tokens": 6,    "completion_tokens": 14,    "total_tokens": 20  }}
```

## Send 1 completion call to many models: Return All Responses[](https://docs.litellm.ai/docs/routing#send-1-completion-call-to-many-models-return-all-responses "Direct link to Send 1 completion call to many models: Return All Responses")

This makes parallel calls to the specified models and returns all responses

Use this to process requests concurrently and get responses from multiple models.

### Example Code[](https://docs.litellm.ai/docs/routing#example-code-2 "Direct link to Example Code")

```
import litellmimport osfrom litellm import batch_completion_models_all_responsesos.environ['ANTHROPIC_API_KEY'] = ""os.environ['OPENAI_API_KEY'] = ""os.environ['COHERE_API_KEY'] = ""responses = batch_completion_models_all_responses(    models=["gpt-3.5-turbo", "claude-instant-1.2", "command-nightly"],     messages=[{"role": "user", "content": "Hey, how's it going"}])print(responses)
```

### Output[](https://docs.litellm.ai/docs/routing#output-1 "Direct link to Output")

```
[<ModelResponse chat.completion id=chatcmpl-e673ec8e-4e8f-4c9e-bf26-bf9fa7ee52b9 at 0x103a62160> JSON: {  "object": "chat.completion",  "choices": [    {      "finish_reason": "stop_sequence",      "index": 0,      "message": {        "content": " It's going well, thank you for asking! How about you?",        "role": "assistant",        "logprobs": null      }    }  ],  "id": "chatcmpl-e673ec8e-4e8f-4c9e-bf26-bf9fa7ee52b9",  "created": 1695222060.917964,  "model": "claude-instant-1.2",  "usage": {    "prompt_tokens": 14,    "completion_tokens": 9,    "total_tokens": 23  }}, <ModelResponse chat.completion id=chatcmpl-ab6c5bd3-b5d9-4711-9697-e28d9fb8a53c at 0x103a62b60> JSON: {  "object": "chat.completion",  "choices": [    {      "finish_reason": "stop",      "index": 0,      "message": {        "content": " It's going well, thank you for asking! How about you?",        "role": "assistant",        "logprobs": null      }    }  ],  "id": "chatcmpl-ab6c5bd3-b5d9-4711-9697-e28d9fb8a53c",  "created": 1695222061.0445492,  "model": "command-nightly",  "usage": {    "prompt_tokens": 6,    "completion_tokens": 14,    "total_tokens": 20  }}, <OpenAIObject chat.completion id=chatcmpl-80szFnKHzCxObW0RqCMw1hWW1Icrq at 0x102dd6430> JSON: {  "id": "chatcmpl-80szFnKHzCxObW0RqCMw1hWW1Icrq",  "object": "chat.completion",  "created": 1695222061,  "model": "gpt-3.5-turbo-0613",  "choices": [    {      "index": 0,      "message": {        "role": "assistant",        "content": "Hello! I'm an AI language model, so I don't have feelings, but I'm here to assist you with any questions or tasks you might have. How can I help you today?"      },      "finish_reason": "stop"    }  ],  "usage": {    "prompt_tokens": 13,    "completion_tokens": 39,    "total_tokens": 52  }}]
```
