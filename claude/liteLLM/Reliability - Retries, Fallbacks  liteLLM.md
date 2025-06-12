---
created: 2025-05-26T16:02:08 (UTC -04:00)
tags: []
source: https://docs.litellm.ai/docs/routing#max-parallel-requests-async
author: 
---

# Reliability - Retries, Fallbacks | liteLLM

> ## Excerpt
> LiteLLM helps prevent failed requests in 2 ways:

---
LiteLLM helps prevent failed requests in 2 ways:

-   Retries
-   Fallbacks: Context Window + General

## Helper utils[](https://docs.litellm.ai/docs/routing#helper-utils "Direct link to Helper utils")

LiteLLM supports the following functions for reliability:

-   `litellm.longer_context_model_fallback_dict`: Dictionary which has a mapping for those models which have larger equivalents
-   `num_retries`: use tenacity retries
-   `completion()` with fallbacks: switch between models/keys/api bases in case of errors.

## Retry failed requests[](https://docs.litellm.ai/docs/routing#retry-failed-requests "Direct link to Retry failed requests")

Call it in completion like this `completion(..num_retries=2)`.

Here's a quick look at how you can use it:

```
from litellm import completionuser_message = "Hello, whats the weather in San Francisco??"messages = [{"content": user_message, "role": "user"}]# normal call response = completion(            model="gpt-3.5-turbo",            messages=messages,            num_retries=2        )
```

## Fallbacks (SDK)[](https://docs.litellm.ai/docs/routing#fallbacks-sdk "Direct link to Fallbacks (SDK)")

### Context Window Fallbacks (SDK)[](https://docs.litellm.ai/docs/routing#context-window-fallbacks-sdk "Direct link to Context Window Fallbacks (SDK)")

```
from litellm import completionfallback_dict = {"gpt-3.5-turbo": "gpt-3.5-turbo-16k"}messages = [{"content": "how does a court case get to the Supreme Court?" * 500, "role": "user"}]completion(model="gpt-3.5-turbo", messages=messages, context_window_fallback_dict=fallback_dict)
```

### Fallbacks - Switch Models/API Keys/API Bases (SDK)[](https://docs.litellm.ai/docs/routing#fallbacks---switch-modelsapi-keysapi-bases-sdk "Direct link to Fallbacks - Switch Models/API Keys/API Bases (SDK)")

LLM APIs can be unstable, completion() with fallbacks ensures you'll always get a response from your calls

#### Usage[](https://docs.litellm.ai/docs/routing#usage "Direct link to Usage")

To use fallback models with `completion()`, specify a list of models in the `fallbacks` parameter.

The `fallbacks` list should include the primary model you want to use, followed by additional models that can be used as backups in case the primary model fails to provide a response.

#### switch models[](https://docs.litellm.ai/docs/routing#switch-models "Direct link to switch models")

```
response = completion(model="bad-model", messages=messages,     fallbacks=["gpt-3.5-turbo" "command-nightly"])
```

#### switch api keys/bases (E.g. azure deployment)[](https://docs.litellm.ai/docs/routing#switch-api-keysbases-eg-azure-deployment "Direct link to switch api keys/bases (E.g. azure deployment)")

Switch between different keys for the same azure deployment, or use another deployment as well.

```
api_key="bad-key"response = completion(model="azure/gpt-4", messages=messages, api_key=api_key,    fallbacks=[{"api_key": "good-key-1"}, {"api_key": "good-key-2", "api_base": "good-api-base-2"}])
```

[Check out this section for implementation details](https://docs.litellm.ai/docs/routing#fallbacks-1)

## Implementation Details (SDK)[](https://docs.litellm.ai/docs/routing#implementation-details-sdk "Direct link to Implementation Details (SDK)")

### Fallbacks[](https://docs.litellm.ai/docs/routing#fallbacks "Direct link to Fallbacks")

#### Output from calls[](https://docs.litellm.ai/docs/routing#output-from-calls "Direct link to Output from calls")

```
Completion with 'bad-model': got exception Unable to map your input to a model. Check your input - {'model': 'bad-model'completion call gpt-3.5-turbo{  "id": "chatcmpl-7qTmVRuO3m3gIBg4aTmAumV1TmQhB",  "object": "chat.completion",  "created": 1692741891,  "model": "gpt-3.5-turbo-0613",  "choices": [    {      "index": 0,      "message": {        "role": "assistant",        "content": "I apologize, but as an AI, I do not have the capability to provide real-time weather updates. However, you can easily check the current weather in San Francisco by using a search engine or checking a weather website or app."      },      "finish_reason": "stop"    }  ],  "usage": {    "prompt_tokens": 16,    "completion_tokens": 46,    "total_tokens": 62  }}
```

#### How does fallbacks work[](https://docs.litellm.ai/docs/routing#how-does-fallbacks-work "Direct link to How does fallbacks work")

When you pass `fallbacks` to `completion`, it makes the first `completion` call using the primary model specified as `model` in `completion(model=model)`. If the primary model fails or encounters an error, it automatically tries the `fallbacks` models in the specified order. This ensures a response even if the primary model is unavailable.

#### Key components of Model Fallbacks implementation:[](https://docs.litellm.ai/docs/routing#key-components-of-model-fallbacks-implementation "Direct link to Key components of Model Fallbacks implementation:")

-   Looping through `fallbacks`
-   Cool-Downs for rate-limited models

#### Looping through `fallbacks`[](https://docs.litellm.ai/docs/routing#looping-through-fallbacks "Direct link to looping-through-fallbacks")

Allow `45seconds` for each request. In the 45s this function tries calling the primary model set as `model`. If model fails it loops through the backup `fallbacks` models and attempts to get a response in the allocated `45s` time set here:

```
while response == None and time.time() - start_time < 45:        for model in fallbacks:
```

#### Cool-Downs for rate-limited models[](https://docs.litellm.ai/docs/routing#cool-downs-for-rate-limited-models "Direct link to Cool-Downs for rate-limited models")

If a model API call leads to an error - allow it to cooldown for `60s`

```
except Exception as e:  print(f"got exception {e} for model {model}")  rate_limited_models.add(model)  model_expiration_times[model] = (      time.time() + 60  )  # cool down this selected model  pass
```

Before making an LLM API call we check if the selected model is in `rate_limited_models`, if so skip making the API call

```
if (  model in rate_limited_models):  # check if model is currently cooling down  if (      model_expiration_times.get(model)      and time.time() >= model_expiration_times[model]  ):      rate_limited_models.remove(          model      )  # check if it's been 60s of cool down and remove model  else:      continue  # skip model
```

#### Full code of completion with fallbacks()[](https://docs.litellm.ai/docs/routing#full-code-of-completion-with-fallbacks "Direct link to Full code of completion with fallbacks()")

```
    response = None    rate_limited_models = set()    model_expiration_times = {}    start_time = time.time()    fallbacks = [kwargs["model"]] + kwargs["fallbacks"]    del kwargs["fallbacks"]  # remove fallbacks so it's not recursive    while response == None and time.time() - start_time < 45:        for model in fallbacks:            # loop thru all models            try:                if (                    model in rate_limited_models                ):  # check if model is currently cooling down                    if (                        model_expiration_times.get(model)                        and time.time() >= model_expiration_times[model]                    ):                        rate_limited_models.remove(                            model                        )  # check if it's been 60s of cool down and remove model                    else:                        continue  # skip model                # delete model from kwargs if it exists                if kwargs.get("model"):                    del kwargs["model"]                print("making completion call", model)                response = litellm.completion(**kwargs, model=model)                if response != None:                    return response            except Exception as e:                print(f"got exception {e} for model {model}")                rate_limited_models.add(model)                model_expiration_times[model] = (                    time.time() + 60                )  # cool down this selected model                pass    return response
```
