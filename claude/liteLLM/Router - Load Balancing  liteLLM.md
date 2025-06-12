---
created: 2025-05-26T16:00:02 (UTC -04:00)
tags: []
source: https://docs.litellm.ai/docs/routing#max-parallel-requests-async
author: 
---

# Router - Load Balancing | liteLLM

> ## Excerpt
> LiteLLM manages:

---
LiteLLM manages:

-   Load-balance across multiple deployments (e.g. Azure/OpenAI)
-   Prioritizing important requests to ensure they don't fail (i.e. Queueing)
-   Basic reliability logic - cooldowns, fallbacks, timeouts and retries (fixed + exponential backoff) across multiple deployments/providers.

In production, litellm supports using Redis as a way to track cooldown server and usage (managing tpm/rpm limits).

## Load Balancing[](https://docs.litellm.ai/docs/routing#load-balancing "Direct link to Load Balancing")

(s/o [@paulpierre](https://www.linkedin.com/in/paulpierre/) and [sweep proxy](https://docs.sweep.dev/blogs/openai-proxy) for their contributions to this implementation) [**See Code**](https://github.com/BerriAI/litellm/blob/main/litellm/router.py)

### Quick Start[](https://docs.litellm.ai/docs/routing#quick-start "Direct link to Quick Start")

Loadbalance across multiple [azure](https://docs.litellm.ai/docs/providers/azure)/[bedrock](https://docs.litellm.ai/docs/providers/bedrock)/[provider](https://docs.litellm.ai/docs/providers/) deployments. LiteLLM will handle retrying in different regions if a call fails.

-   SDK
-   PROXY

```
from litellm import Routermodel_list = [{ # list of model deployments     "model_name": "gpt-3.5-turbo", # model alias -> loadbalance between models with same `model_name`    "litellm_params": { # params for litellm completion/embedding call         "model": "azure/chatgpt-v-2", # actual model name        "api_key": os.getenv("AZURE_API_KEY"),        "api_version": os.getenv("AZURE_API_VERSION"),        "api_base": os.getenv("AZURE_API_BASE")    }}, {    "model_name": "gpt-3.5-turbo",     "litellm_params": { # params for litellm completion/embedding call         "model": "azure/chatgpt-functioncalling",         "api_key": os.getenv("AZURE_API_KEY"),        "api_version": os.getenv("AZURE_API_VERSION"),        "api_base": os.getenv("AZURE_API_BASE")    }}, {    "model_name": "gpt-3.5-turbo",     "litellm_params": { # params for litellm completion/embedding call         "model": "gpt-3.5-turbo",         "api_key": os.getenv("OPENAI_API_KEY"),    }}, {    "model_name": "gpt-4",     "litellm_params": { # params for litellm completion/embedding call         "model": "azure/gpt-4",         "api_key": os.getenv("AZURE_API_KEY"),        "api_base": os.getenv("AZURE_API_BASE"),        "api_version": os.getenv("AZURE_API_VERSION"),    }}, {    "model_name": "gpt-4",     "litellm_params": { # params for litellm completion/embedding call         "model": "gpt-4",         "api_key": os.getenv("OPENAI_API_KEY"),    }},]router = Router(model_list=model_list)# openai.ChatCompletion.create replacement# requests with model="gpt-3.5-turbo" will pick a deployment where model_name="gpt-3.5-turbo"response = await router.acompletion(model="gpt-3.5-turbo",                 messages=[{"role": "user", "content": "Hey, how's it going?"}])print(response)# openai.ChatCompletion.create replacement# requests with model="gpt-4" will pick a deployment where model_name="gpt-4"response = await router.acompletion(model="gpt-4",                 messages=[{"role": "user", "content": "Hey, how's it going?"}])print(response)
```

### Available Endpoints[](https://docs.litellm.ai/docs/routing#available-endpoints "Direct link to Available Endpoints")

-   `router.completion()` - chat completions endpoint to call 100+ LLMs
-   `router.acompletion()` - async chat completion calls
-   `router.embedding()` - embedding endpoint for Azure, OpenAI, Huggingface endpoints
-   `router.aembedding()` - async embeddings calls
-   `router.text_completion()` - completion calls in the old OpenAI `/v1/completions` endpoint format
-   `router.atext_completion()` - async text completion calls
-   `router.image_generation()` - completion calls in OpenAI `/v1/images/generations` endpoint format
-   `router.aimage_generation()` - async image generation calls

## Advanced - Routing Strategies ⭐️[](https://docs.litellm.ai/docs/routing#advanced---routing-strategies-%EF%B8%8F "Direct link to Advanced - Routing Strategies ⭐️")

#### Routing Strategies - Weighted Pick, Rate Limit Aware, Least Busy, Latency Based, Cost Based[](https://docs.litellm.ai/docs/routing#routing-strategies---weighted-pick-rate-limit-aware-least-busy-latency-based-cost-based "Direct link to Routing Strategies - Weighted Pick, Rate Limit Aware, Least Busy, Latency Based, Cost Based")

Router provides 4 strategies for routing your calls across multiple deployments:

-   Rate-Limit Aware v2 (ASYNC)
-   Latency-Based
-   (Default) Weighted Pick (Async)
-   Rate-Limit Aware
-   Least-Busy
-   Custom Routing Strategy
-   Lowest Cost Routing (Async)

**🎉 NEW** This is an async implementation of usage-based-routing.

**Filters out deployment if tpm/rpm limit exceeded** - If you pass in the deployment's tpm/rpm limits.

Routes to **deployment with lowest TPM usage** for that minute.

In production, we use Redis to track usage (TPM/RPM) across multiple deployments. This implementation uses **async redis calls** (redis.incr and redis.mget).

For Azure, [you get 6 RPM per 1000 TPM](https://stackoverflow.com/questions/77368844/what-is-the-request-per-minute-rate-limit-for-azure-openai-models-for-gpt-3-5-tu)

-   sdk
-   proxy

```
from litellm import Router model_list = [{ # list of model deployments     "model_name": "gpt-3.5-turbo", # model alias     "litellm_params": { # params for litellm completion/embedding call         "model": "azure/chatgpt-v-2", # actual model name        "api_key": os.getenv("AZURE_API_KEY"),        "api_version": os.getenv("AZURE_API_VERSION"),        "api_base": os.getenv("AZURE_API_BASE")        "tpm": 100000,        "rpm": 10000,    }, }, {    "model_name": "gpt-3.5-turbo",     "litellm_params": { # params for litellm completion/embedding call         "model": "azure/chatgpt-functioncalling",         "api_key": os.getenv("AZURE_API_KEY"),        "api_version": os.getenv("AZURE_API_VERSION"),        "api_base": os.getenv("AZURE_API_BASE")        "tpm": 100000,        "rpm": 1000,    },}, {    "model_name": "gpt-3.5-turbo",     "litellm_params": { # params for litellm completion/embedding call         "model": "gpt-3.5-turbo",         "api_key": os.getenv("OPENAI_API_KEY"),        "tpm": 100000,        "rpm": 1000,    },}]router = Router(model_list=model_list,                 redis_host=os.environ["REDIS_HOST"],                 redis_password=os.environ["REDIS_PASSWORD"],                 redis_port=os.environ["REDIS_PORT"],                 routing_strategy="usage-based-routing-v2" # 👈 KEY CHANGE                enable_pre_call_checks=True, # enables router rate limits for concurrent calls                )response = await router.acompletion(model="gpt-3.5-turbo",                 messages=[{"role": "user", "content": "Hey, how's it going?"}]print(response)
```

## Basic Reliability[](https://docs.litellm.ai/docs/routing#basic-reliability "Direct link to Basic Reliability")

### Weighted Deployments[](https://docs.litellm.ai/docs/routing#weighted-deployments "Direct link to Weighted Deployments")

Set `weight` on a deployment to pick one deployment more often than others.

This works across **simple-shuffle** routing strategy (this is the default, if no routing strategy is selected).

-   SDK
-   PROXY

```
from litellm import Router model_list = [    {        "model_name": "o1",        "litellm_params": {            "model": "o1-preview",             "api_key": os.getenv("OPENAI_API_KEY"),             "weight": 1        },    },    {        "model_name": "o1",        "litellm_params": {            "model": "o1-preview",             "api_key": os.getenv("OPENAI_API_KEY"),             "weight": 2 # 👈 PICK THIS DEPLOYMENT 2x MORE OFTEN THAN o1-preview        },    },]router = Router(model_list=model_list, routing_strategy="cost-based-routing")response = await router.acompletion(    model="gpt-3.5-turbo",     messages=[{"role": "user", "content": "Hey, how's it going?"}])print(response)
```

### Max Parallel Requests (ASYNC)[](https://docs.litellm.ai/docs/routing#max-parallel-requests-async "Direct link to Max Parallel Requests (ASYNC)")

Used in semaphore for async requests on router. Limit the max concurrent calls made to a deployment. Useful in high-traffic scenarios.

If tpm/rpm is set, and no max parallel request limit given, we use the RPM or calculated RPM (tpm/1000/6) as the max parallel request limit.

```
from litellm import Router model_list = [{    "model_name": "gpt-4",    "litellm_params": {        "model": "azure/gpt-4",        ...        "max_parallel_requests": 10 # 👈 SET PER DEPLOYMENT    }}]### OR ### router = Router(model_list=model_list, default_max_parallel_requests=20) # 👈 SET DEFAULT MAX PARALLEL REQUESTS # deployment max parallel requests > default max parallel requests
```

[**See Code**](https://github.com/BerriAI/litellm/blob/a978f2d8813c04dad34802cb95e0a0e35a3324bc/litellm/utils.py#L5605)

### Cooldowns[](https://docs.litellm.ai/docs/routing#cooldowns "Direct link to Cooldowns")

Set the limit for how many calls a model is allowed to fail in a minute, before being cooled down for a minute.

-   SDK
-   PROXY

```
from litellm import Routermodel_list = [{...}]router = Router(model_list=model_list,                 allowed_fails=1,      # cooldown model if it fails > 1 call in a minute.                 cooldown_time=100    # cooldown the deployment for 100 seconds if it num_fails > allowed_fails        )user_message = "Hello, whats the weather in San Francisco??"messages = [{"content": user_message, "role": "user"}]# normal call response = router.completion(model="gpt-3.5-turbo", messages=messages)print(f"response: {response}")
```

**Expected Response**

```
No deployments available for selected model, Try again in 60 seconds. Passed model=claude-3-5-sonnet. pre-call-checks=False, allowed_model_region=n/a.
```

#### **Disable cooldowns**[](https://docs.litellm.ai/docs/routing#disable-cooldowns "Direct link to disable-cooldowns")

-   SDK
-   PROXY

```
from litellm import Router router = Router(..., disable_cooldowns=True)
```

### Retries[](https://docs.litellm.ai/docs/routing#retries "Direct link to Retries")

For both async + sync functions, we support retrying failed requests.

For RateLimitError we implement exponential backoffs

For generic errors, we retry immediately

Here's a quick look at how we can set `num_retries = 3`:

```
from litellm import Routermodel_list = [{...}]router = Router(model_list=model_list,                  num_retries=3)user_message = "Hello, whats the weather in San Francisco??"messages = [{"content": user_message, "role": "user"}]# normal call response = router.completion(model="gpt-3.5-turbo", messages=messages)print(f"response: {response}")
```

We also support setting minimum time to wait before retrying a failed request. This is via the `retry_after` param.

```
from litellm import Routermodel_list = [{...}]router = Router(model_list=model_list,                  num_retries=3, retry_after=5) # waits min 5s before retrying requestuser_message = "Hello, whats the weather in San Francisco??"messages = [{"content": user_message, "role": "user"}]# normal call response = router.completion(model="gpt-3.5-turbo", messages=messages)print(f"response: {response}")
```

### \[Advanced\]: Custom Retries, Cooldowns based on Error Type[](https://docs.litellm.ai/docs/routing#advanced-custom-retries-cooldowns-based-on-error-type "Direct link to advanced-custom-retries-cooldowns-based-on-error-type")

-   Use `RetryPolicy` if you want to set a `num_retries` based on the Exception received
-   Use `AllowedFailsPolicy` to set a custom number of `allowed_fails`/minute before cooling down a deployment

[**See All Exception Types**](https://github.com/BerriAI/litellm/blob/ccda616f2f881375d4e8586c76fe4662909a7d22/litellm/types/router.py#L436)

-   SDK
-   PROXY

Example:

```
retry_policy = RetryPolicy(    ContentPolicyViolationErrorRetries=3,         # run 3 retries for ContentPolicyViolationErrors    AuthenticationErrorRetries=0,                 # run 0 retries for AuthenticationErrorRetries)allowed_fails_policy = AllowedFailsPolicy(    ContentPolicyViolationErrorAllowedFails=1000, # Allow 1000 ContentPolicyViolationError before cooling down a deployment    RateLimitErrorAllowedFails=100,               # Allow 100 RateLimitErrors before cooling down a deployment)
```

Example Usage

```
from litellm.router import RetryPolicy, AllowedFailsPolicyretry_policy = RetryPolicy(    ContentPolicyViolationErrorRetries=3,         # run 3 retries for ContentPolicyViolationErrors    AuthenticationErrorRetries=0,                 # run 0 retries for AuthenticationErrorRetries    BadRequestErrorRetries=1,    TimeoutErrorRetries=2,    RateLimitErrorRetries=3,)allowed_fails_policy = AllowedFailsPolicy(    ContentPolicyViolationErrorAllowedFails=1000, # Allow 1000 ContentPolicyViolationError before cooling down a deployment    RateLimitErrorAllowedFails=100,               # Allow 100 RateLimitErrors before cooling down a deployment)router = litellm.Router(    model_list=[        {            "model_name": "gpt-3.5-turbo",  # openai model name            "litellm_params": {  # params for litellm completion/embedding call                "model": "azure/chatgpt-v-2",                "api_key": os.getenv("AZURE_API_KEY"),                "api_version": os.getenv("AZURE_API_VERSION"),                "api_base": os.getenv("AZURE_API_BASE"),            },        },        {            "model_name": "bad-model",  # openai model name            "litellm_params": {  # params for litellm completion/embedding call                "model": "azure/chatgpt-v-2",                "api_key": "bad-key",                "api_version": os.getenv("AZURE_API_VERSION"),                "api_base": os.getenv("AZURE_API_BASE"),            },        },    ],    retry_policy=retry_policy,    allowed_fails_policy=allowed_fails_policy,)response = await router.acompletion(    model=model,    messages=messages,)
```

### Caching[](https://docs.litellm.ai/docs/routing#caching "Direct link to Caching")

In production, we recommend using a Redis cache. For quickly testing things locally, we also support simple in-memory caching.

**In-memory Cache**

```
router = Router(model_list=model_list,                 cache_responses=True)print(response)
```

**Redis Cache**

```
router = Router(model_list=model_list,                 redis_host=os.getenv("REDIS_HOST"),                 redis_password=os.getenv("REDIS_PASSWORD"),                 redis_port=os.getenv("REDIS_PORT"),                cache_responses=True)print(response)
```

**Pass in Redis URL, additional kwargs**

```
router = Router(model_list: Optional[list] = None,                 ## CACHING ##                  redis_url=os.getenv("REDIS_URL")",                 cache_kwargs= {}, # additional kwargs to pass to RedisCache (see caching.py)                 cache_responses=True)
```

## Pre-Call Checks (Context Window, EU-Regions)[](https://docs.litellm.ai/docs/routing#pre-call-checks-context-window-eu-regions "Direct link to Pre-Call Checks (Context Window, EU-Regions)")

Enable pre-call checks to filter out:

1.  deployments with context window limit < messages for a call.
2.  deployments outside of eu-region

-   SDK
-   Proxy

**1\. Enable pre-call checks**

```
from litellm import Router # ...router = Router(model_list=model_list, enable_pre_call_checks=True) # 👈 Set to True
```

**2\. Set Model List**

For context window checks on azure deployments, set the base model. Pick the base model from [this list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), all the azure models start with `azure/`.

For 'eu-region' filtering, Set 'region\_name' of deployment.

**Note:** We automatically infer region\_name for Vertex AI, Bedrock, and IBM WatsonxAI based on your litellm params. For Azure, set `litellm.enable_preview = True`.

[**See Code**](https://github.com/BerriAI/litellm/blob/d33e49411d6503cb634f9652873160cd534dec96/litellm/router.py#L2958)

```
model_list = [            {                "model_name": "gpt-3.5-turbo", # model group name                "litellm_params": {  # params for litellm completion/embedding call                    "model": "azure/chatgpt-v-2",                    "api_key": os.getenv("AZURE_API_KEY"),                    "api_version": os.getenv("AZURE_API_VERSION"),                    "api_base": os.getenv("AZURE_API_BASE"),                    "region_name": "eu" # 👈 SET 'EU' REGION NAME                    "base_model": "azure/gpt-35-turbo", # 👈 (Azure-only) SET BASE MODEL                },            },            {                "model_name": "gpt-3.5-turbo", # model group name                "litellm_params": {  # params for litellm completion/embedding call                    "model": "gpt-3.5-turbo-1106",                    "api_key": os.getenv("OPENAI_API_KEY"),                },            },            {                "model_name": "gemini-pro",                "litellm_params: {                    "model": "vertex_ai/gemini-pro-1.5",                     "vertex_project": "adroit-crow-1234",                    "vertex_location": "us-east1" # 👈 AUTOMATICALLY INFERS 'region_name'                }            }        ]router = Router(model_list=model_list, enable_pre_call_checks=True) 
```

**3\. Test it!**

-   Context Window Check
-   EU Region Check

```
"""- Give a gpt-3.5-turbo model group with different context windows (4k vs. 16k)- Send a 5k prompt- Assert it works"""from litellm import Routerimport osmodel_list = [    {        "model_name": "gpt-3.5-turbo",  # model group name        "litellm_params": {  # params for litellm completion/embedding call            "model": "azure/chatgpt-v-2",            "api_key": os.getenv("AZURE_API_KEY"),            "api_version": os.getenv("AZURE_API_VERSION"),            "api_base": os.getenv("AZURE_API_BASE"),            "base_model": "azure/gpt-35-turbo",        },        "model_info": {            "base_model": "azure/gpt-35-turbo",         }    },    {        "model_name": "gpt-3.5-turbo",  # model group name        "litellm_params": {  # params for litellm completion/embedding call            "model": "gpt-3.5-turbo-1106",            "api_key": os.getenv("OPENAI_API_KEY"),        },    },]router = Router(model_list=model_list, enable_pre_call_checks=True) text = "What is the meaning of 42?" * 5000response = router.completion(    model="gpt-3.5-turbo",    messages=[        {"role": "system", "content": text},        {"role": "user", "content": "Who was Alexander?"},    ],)print(f"response: {response}")
```

## Caching across model groups[](https://docs.litellm.ai/docs/routing#caching-across-model-groups "Direct link to Caching across model groups")

If you want to cache across 2 different model groups (e.g. azure deployments, and openai), use caching groups.

```
import litellm, asyncio, timefrom litellm import Router # set os envos.environ["OPENAI_API_KEY"] = ""os.environ["AZURE_API_KEY"] = ""os.environ["AZURE_API_BASE"] = ""os.environ["AZURE_API_VERSION"] = ""async def test_acompletion_caching_on_router_caching_groups():     # tests acompletion + caching on router     try:        litellm.set_verbose = True        model_list = [            {                "model_name": "openai-gpt-3.5-turbo",                "litellm_params": {                    "model": "gpt-3.5-turbo-0613",                    "api_key": os.getenv("OPENAI_API_KEY"),                },            },            {                "model_name": "azure-gpt-3.5-turbo",                "litellm_params": {                    "model": "azure/chatgpt-v-2",                    "api_key": os.getenv("AZURE_API_KEY"),                    "api_base": os.getenv("AZURE_API_BASE"),                    "api_version": os.getenv("AZURE_API_VERSION")                },            }        ]        messages = [            {"role": "user", "content": f"write a one sentence poem {time.time()}?"}        ]        start_time = time.time()        router = Router(model_list=model_list,                 cache_responses=True,                 caching_groups=[("openai-gpt-3.5-turbo", "azure-gpt-3.5-turbo")])        response1 = await router.acompletion(model="openai-gpt-3.5-turbo", messages=messages, temperature=1)        print(f"response1: {response1}")        await asyncio.sleep(1) # add cache is async, async sleep for cache to get set        response2 = await router.acompletion(model="azure-gpt-3.5-turbo", messages=messages, temperature=1)        assert response1.id == response2.id        assert len(response1.choices[0].message.content) > 0        assert response1.choices[0].message.content == response2.choices[0].message.content    except Exception as e:        traceback.print_exc()asyncio.run(test_acompletion_caching_on_router_caching_groups())
```

## Alerting 🚨[](https://docs.litellm.ai/docs/routing#alerting- "Direct link to Alerting 🚨")

Send alerts to slack / your webhook url for the following events

-   LLM API Exceptions
-   Slow LLM Responses

Get a slack webhook url from [https://api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks)

#### Usage[](https://docs.litellm.ai/docs/routing#usage "Direct link to Usage")

Initialize an `AlertingConfig` and pass it to `litellm.Router`. The following code will trigger an alert because `api_key=bad-key` which is invalid

```
from litellm.router import AlertingConfigimport litellmimport osrouter = litellm.Router(    model_list=[        {            "model_name": "gpt-3.5-turbo",            "litellm_params": {                "model": "gpt-3.5-turbo",                "api_key": "bad_key",            },        }    ],    alerting_config= AlertingConfig(        alerting_threshold=10,                        # threshold for slow / hanging llm responses (in seconds). Defaults to 300 seconds        webhook_url= os.getenv("SLACK_WEBHOOK_URL")   # webhook you want to send alerts to    ),)try:    await router.acompletion(        model="gpt-3.5-turbo",        messages=[{"role": "user", "content": "Hey, how's it going?"}],    )except:    pass
```

## Track cost for Azure Deployments[](https://docs.litellm.ai/docs/routing#track-cost-for-azure-deployments "Direct link to Track cost for Azure Deployments")

**Problem**: Azure returns `gpt-4` in the response when `azure/gpt-4-1106-preview` is used. This leads to inaccurate cost tracking

**Solution** ✅ : Set `model_info["base_model"]` on your router init so litellm uses the correct model for calculating azure cost

Step 1. Router Setup

```
from litellm import Routermodel_list = [    { # list of model deployments         "model_name": "gpt-4-preview", # model alias         "litellm_params": { # params for litellm completion/embedding call             "model": "azure/chatgpt-v-2", # actual model name            "api_key": os.getenv("AZURE_API_KEY"),            "api_version": os.getenv("AZURE_API_VERSION"),            "api_base": os.getenv("AZURE_API_BASE")        },        "model_info": {            "base_model": "azure/gpt-4-1106-preview" # azure/gpt-4-1106-preview will be used for cost tracking, ensure this exists in litellm model_prices_and_context_window.json        }    },     {        "model_name": "gpt-4-32k",         "litellm_params": { # params for litellm completion/embedding call             "model": "azure/chatgpt-functioncalling",             "api_key": os.getenv("AZURE_API_KEY"),            "api_version": os.getenv("AZURE_API_VERSION"),            "api_base": os.getenv("AZURE_API_BASE")        },        "model_info": {            "base_model": "azure/gpt-4-32k" # azure/gpt-4-32k will be used for cost tracking, ensure this exists in litellm model_prices_and_context_window.json        }    }]router = Router(model_list=model_list)
```

Step 2. Access `response_cost` in the custom callback, **litellm calculates the response cost for you**

```
import litellmfrom litellm.integrations.custom_logger import CustomLoggerclass MyCustomHandler(CustomLogger):            def log_success_event(self, kwargs, response_obj, start_time, end_time):         print(f"On Success")        response_cost = kwargs.get("response_cost")        print("response_cost=", response_cost)customHandler = MyCustomHandler()litellm.callbacks = [customHandler]# router completion callresponse = router.completion(    model="gpt-4-32k",     messages=[{ "role": "user", "content": "Hi who are you"}])
```

#### Default litellm.completion/embedding params[](https://docs.litellm.ai/docs/routing#default-litellmcompletionembedding-params "Direct link to Default litellm.completion/embedding params")

You can also set default params for litellm completion/embedding calls. Here's how to do that:

```
from litellm import Routerfallback_dict = {"gpt-3.5-turbo": "gpt-3.5-turbo-16k"}router = Router(model_list=model_list,                 default_litellm_params={"context_window_fallback_dict": fallback_dict})user_message = "Hello, whats the weather in San Francisco??"messages = [{"content": user_message, "role": "user"}]# normal call response = router.completion(model="gpt-3.5-turbo", messages=messages)print(f"response: {response}")
```

## Custom Callbacks - Track API Key, API Endpoint, Model Used[](https://docs.litellm.ai/docs/routing#custom-callbacks---track-api-key-api-endpoint-model-used "Direct link to Custom Callbacks - Track API Key, API Endpoint, Model Used")

If you need to track the api\_key, api endpoint, model, custom\_llm\_provider used for each completion call, you can setup a [custom callback](https://docs.litellm.ai/docs/observability/custom_callback)

### Usage[](https://docs.litellm.ai/docs/routing#usage-1 "Direct link to Usage")

```
import litellmfrom litellm.integrations.custom_logger import CustomLoggerclass MyCustomHandler(CustomLogger):            def log_success_event(self, kwargs, response_obj, start_time, end_time):         print(f"On Success")        print("kwargs=", kwargs)        litellm_params= kwargs.get("litellm_params")        api_key = litellm_params.get("api_key")        api_base = litellm_params.get("api_base")        custom_llm_provider= litellm_params.get("custom_llm_provider")        response_cost = kwargs.get("response_cost")        # print the values        print("api_key=", api_key)        print("api_base=", api_base)        print("custom_llm_provider=", custom_llm_provider)        print("response_cost=", response_cost)    def log_failure_event(self, kwargs, response_obj, start_time, end_time):         print(f"On Failure")        print("kwargs=")customHandler = MyCustomHandler()litellm.callbacks = [customHandler]# Init Routerrouter = Router(model_list=model_list, routing_strategy="simple-shuffle")# router completion callresponse = router.completion(    model="gpt-3.5-turbo",     messages=[{ "role": "user", "content": "Hi who are you"}])
```

## Deploy Router[](https://docs.litellm.ai/docs/routing#deploy-router "Direct link to Deploy Router")

If you want a server to load balance across different LLM APIs, use our [LiteLLM Proxy Server](https://docs.litellm.ai/docs/simple_proxy#load-balancing---multiple-instances-of-1-model)

## Debugging Router[](https://docs.litellm.ai/docs/routing#debugging-router "Direct link to Debugging Router")

### Basic Debugging[](https://docs.litellm.ai/docs/routing#basic-debugging "Direct link to Basic Debugging")

Set `Router(set_verbose=True)`

```
from litellm import Routerrouter = Router(    model_list=model_list,    set_verbose=True)
```

### Detailed Debugging[](https://docs.litellm.ai/docs/routing#detailed-debugging "Direct link to Detailed Debugging")

Set `Router(set_verbose=True,debug_level="DEBUG")`

```
from litellm import Routerrouter = Router(    model_list=model_list,    set_verbose=True,    debug_level="DEBUG"  # defaults to INFO)
```

### Very Detailed Debugging[](https://docs.litellm.ai/docs/routing#very-detailed-debugging "Direct link to Very Detailed Debugging")

Set `litellm.set_verbose=True` and `Router(set_verbose=True,debug_level="DEBUG")`

```
from litellm import Routerimport litellmlitellm.set_verbose = Truerouter = Router(    model_list=model_list,    set_verbose=True,    debug_level="DEBUG"  # defaults to INFO)
```

## Router General Settings[](https://docs.litellm.ai/docs/routing#router-general-settings "Direct link to Router General Settings")

### Usage[](https://docs.litellm.ai/docs/routing#usage-2 "Direct link to Usage")

```
router = Router(model_list=..., router_general_settings=RouterGeneralSettings(async_only_mode=True))
```

### Spec[](https://docs.litellm.ai/docs/routing#spec "Direct link to Spec")

```
class RouterGeneralSettings(BaseModel):    async_only_mode: bool = Field(        default=False    )  # this will only initialize async clients. Good for memory utils    pass_through_all_models: bool = Field(        default=False    )  # if passed a model not llm_router model list, pass through the request to litellm.acompletion/embedding
```
