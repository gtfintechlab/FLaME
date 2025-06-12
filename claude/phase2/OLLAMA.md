# Ollama Integration Guide

This document provides comprehensive guidance for working with Ollama on your machine, both directly and through LiteLLM for OpenAI-compatible API access.

## Table of Contents
1. [Direct Ollama Setup](#direct-ollama-setup)
2. [LiteLLM Integration](#litellm-integration)
3. [Model Management](#model-management)
4. [Configuration Examples](#configuration-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Direct Ollama Setup

### Basic Ollama Server Configuration

**Default API Endpoint:**
```
http://127.0.0.1:11434
```

**Starting Ollama Server:**
```bash
# Start the Ollama server (run in separate terminal)
ollama serve

# Verify server is running
curl http://127.0.0.1:11434/api/tags
```

**Available API Endpoints:**
- `/api/tags` - List available models
- `/api/generate` - Generate completions
- `/api/chat` - Chat completions
- `/api/pull` - Download models
- `/api/show` - Show model information

### Tested Working Configuration

Based on our FERRArI integration testing:

**Server URL:** `http://127.0.0.1:11434`
**Timeout:** 120 seconds (recommended for financial reasoning tasks)
**Max Tokens:** 512 (configurable, tested successfully)
**Temperature:** 0.0 (for reproducible results)

**Example HTTP Request:**
```bash
curl -X POST http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:1.5b",
    "prompt": "What is 2+2?",
    "stream": false,
    "options": {
      "temperature": 0.0,
      "num_predict": 512
    }
  }'
```

## LiteLLM Integration

### Installation

```bash
pip install 'litellm[proxy]'
```

### Basic LiteLLM Usage with Ollama

**Direct Python Integration:**
```python
from litellm import completion

response = completion(
    model="ollama/qwen2.5:1.5b",
    messages=[{"content": "What is 2+2?", "role": "user"}],
    api_base="http://localhost:11434"
)
print(response.choices[0].message.content)
```

**Async Support:**
```python
import asyncio
from litellm import acompletion

async def get_response():
    response = await acompletion(
        model="ollama/qwen2.5:1.5b",
        messages=[{"content": "Financial question here", "role": "user"}],
        api_base="http://localhost:11434"
    )
    return response

response = asyncio.run(get_response())
```

### LiteLLM Proxy Server Setup

**Starting the Proxy:**
```bash
# Start LiteLLM proxy for Ollama models
litellm --model ollama/qwen2.5:1.5b

# With custom port
litellm --model ollama/qwen2.5:1.5b --port 8000

# With debug logging
litellm --model ollama/qwen2.5:1.5b --debug
```

**Proxy Configuration File (config.yaml):**
```yaml
model_list:
  - model_name: qwen2.5-1.5b
    litellm_params:
      model: ollama/qwen2.5:1.5b
      api_base: http://localhost:11434
  
  - model_name: tinyllama
    litellm_params:
      model: ollama/tinyllama
      api_base: http://localhost:11434
  
  - model_name: mistral-7b
    litellm_params:
      model: ollama/mistral:7b-instruct
      api_base: http://localhost:11434

general_settings:
  master_key: your-secret-key  # Optional: for API key protection
```

**Using Configuration File:**
```bash
litellm --config config.yaml
```

### OpenAI-Compatible API Access

Once LiteLLM proxy is running (default: `http://0.0.0.0:4000`):

**Using OpenAI Python SDK:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-secret-key",  # Optional if no master_key set
    base_url="http://localhost:4000"
)

response = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[
        {"role": "user", "content": "Calculate profit margin for revenue $1000M and expenses $800M"}
    ],
    temperature=0.0,
    max_tokens=512
)

print(response.choices[0].message.content)
```

**Using curl:**
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "qwen2.5-1.5b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

## Model Management

### Recommended Models for Different Use Cases

**Lightweight Testing:**
```bash
# Ultra-lightweight for quick testing (~700MB)
ollama pull tinyllama

# Small but capable for development (~900MB)
ollama pull qwen2.5:1.5b
```

**Financial Reasoning (Tested in FERRArI):**
```bash
# Primary development model - excellent for financial calculations
ollama pull qwen2.5:1.5b

# Advanced reasoning for complex financial scenarios
ollama pull mistral:7b-instruct
```

**Production Models:**
```bash
# Larger models for production use
ollama pull qwen2.5:7b
ollama pull llama2:13b
ollama pull mistral:7b-instruct
```

### Model Information Commands

**List Installed Models:**
```bash
ollama list
```

**Show Model Details:**
```bash
ollama show qwen2.5:1.5b
```

**Remove Models:**
```bash
ollama rm tinyllama
```

**Model Storage Location:**
- Linux: `~/.ollama/models`
- macOS: `~/.ollama/models`
- Windows: `%USERPROFILE%\.ollama\models`

## Configuration Examples

### FERRArI Integration Configuration

Based on our successful testing:

```python
# Ollama Engine Configuration
ollama_config = {
    "base_url": "http://127.0.0.1:11434",
    "timeout": 120.0,
    "max_retries": 3,
    "stream": False,
    "keep_alive": "5m"
}

# Model Information
model_info = {
    "name": "qwen2.5:1.5b",
    "family": "qwen",
    "size_category": "small",
    "parameters_b": 1.5,
    "tensor_parallel": 1,
    "gpu_memory_gb": 2.0,
    "context_length": 32768
}

# Financial Reasoning Prompt Template
prompt_template = """You are a financial analyst. Solve this step-by-step.

Question: {question}

Please calculate the answer step by step and provide your response in this exact format:
Therefore, the answer is <begin_final_answer> FINAL_ANSWER <end_final_answer> with derivation <begin_derivation> CALCULATION <end_derivation>.

Requirements:
- FINAL_ANSWER must be numeric only (no %, $, or extra text)
- Express percentages as numbers (e.g., 25 for 25%)
- Show your calculation clearly in CALCULATION
- Be precise with arithmetic"""
```

### LiteLLM Production Configuration

```yaml
# production-config.yaml
model_list:
  # Development models
  - model_name: qwen-dev
    litellm_params:
      model: ollama/qwen2.5:1.5b
      api_base: http://localhost:11434
      
  # Production models  
  - model_name: qwen-prod
    litellm_params:
      model: ollama/qwen2.5:7b
      api_base: http://localhost:11434
      
  # Specialized models
  - model_name: mistral-reasoning
    litellm_params:
      model: ollama/mistral:7b-instruct
      api_base: http://localhost:11434

router_settings:
  routing_strategy: "simple-shuffle"  # Load balance between models
  
general_settings:
  master_key: ${LITELLM_MASTER_KEY}
  database_url: "postgresql://..."  # For logging and analytics
  
logging:
  - provider: "supabase"
    table_name: "model_requests"
```

## Best Practices

### Development Workflow

1. **Start with Lightweight Models:**
   ```bash
   ollama pull tinyllama  # Quick testing
   ollama pull qwen2.5:1.5b  # Development
   ```

2. **Use LiteLLM for Standardization:**
   - Provides OpenAI-compatible API
   - Easy to switch between local/remote models
   - Better integration with existing tools

3. **Model Selection Guidelines:**
   - **tinyllama**: Initial testing, proof of concepts
   - **qwen2.5:1.5b**: Primary development, financial reasoning
   - **mistral:7b-instruct**: Advanced reasoning, production testing
   - **qwen2.5:7b+**: Production deployments

### Performance Optimization

**Memory Management:**
```bash
# Keep models loaded for specific time
ollama run qwen2.5:1.5b --keep-alive 10m

# Force unload models to free memory
ollama stop qwen2.5:1.5b
```

**Concurrent Usage:**
- Ollama processes requests sequentially by default
- Use LiteLLM proxy for better concurrency handling
- Consider multiple Ollama instances for high throughput

### Security Considerations

**API Access Control:**
```yaml
# Use LiteLLM proxy with authentication
general_settings:
  master_key: strong-random-key
  allowed_origins: ["https://yourdomain.com"]
```

**Network Configuration:**
```bash
# Bind to specific interface
OLLAMA_HOST=127.0.0.1:11434 ollama serve

# Use environment variables for sensitive config
export OLLAMA_API_BASE=http://127.0.0.1:11434
export LITELLM_MASTER_KEY=your-secret-key
```

## Troubleshooting

### Common Issues

**1. Ollama Server Not Responding:**
```bash
# Check if server is running
ps aux | grep ollama

# Check port availability
netstat -tlnp | grep 11434

# Restart server
pkill ollama
ollama serve
```

**2. Model Not Found:**
```bash
# List available models
ollama list

# Pull missing model
ollama pull qwen2.5:1.5b

# Verify model exists
curl http://127.0.0.1:11434/api/tags
```

**3. LiteLLM Connection Issues:**
```bash
# Test direct Ollama connection first
curl -X POST http://127.0.0.1:11434/api/generate \
  -d '{"model": "qwen2.5:1.5b", "prompt": "test"}'

# Start LiteLLM with debug logging
litellm --model ollama/qwen2.5:1.5b --debug
```

**4. Memory Issues:**
```bash
# Check available memory
free -h

# Monitor Ollama memory usage
top -p $(pgrep ollama)

# Use smaller models if memory constrained
ollama pull tinyllama
```

### Debug Commands

**Ollama Logs:**
```bash
# View Ollama server logs
journalctl -u ollama -f

# Or check system logs
tail -f /var/log/ollama.log
```

**LiteLLM Debugging:**
```bash
# Enable detailed debugging
export LITELLM_LOG=DEBUG
litellm --model ollama/qwen2.5:1.5b --detailed_debug

# Test specific endpoints
curl -X GET http://localhost:4000/health
curl -X GET http://localhost:4000/models
```

### Performance Monitoring

**Response Time Testing:**
```python
import time
from litellm import completion

start_time = time.time()
response = completion(
    model="ollama/qwen2.5:1.5b",
    messages=[{"content": "Quick test", "role": "user"}],
    api_base="http://localhost:11434"
)
elapsed_time = time.time() - start_time
print(f"Response time: {elapsed_time:.2f} seconds")
```

**Token Usage Monitoring:**
```python
# Track token usage through LiteLLM
response = completion(
    model="ollama/qwen2.5:1.5b", 
    messages=[{"content": "Test message", "role": "user"}]
)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
```

## Integration with Other Projects

### For Similar Financial AI Projects:

1. **Use LiteLLM Proxy Approach:**
   - Provides standardized OpenAI-compatible API
   - Easy model switching and load balancing
   - Better monitoring and logging capabilities

2. **Recommended Architecture:**
   ```
   Your Application → LiteLLM Proxy → Ollama Server → Local Models
   ```

3. **Configuration Template:**
   ```yaml
   model_list:
     - model_name: financial-reasoning-small
       litellm_params:
         model: ollama/qwen2.5:1.5b
         api_base: http://localhost:11434
     
     - model_name: financial-reasoning-large  
       litellm_params:
         model: ollama/qwen2.5:7b
         api_base: http://localhost:11434
   ```

4. **Benefits:**
   - Local inference (no external API costs)
   - Privacy and data security
   - Consistent response times
   - Easy model experimentation
   - OpenAI SDK compatibility

This setup provides a robust foundation for local AI model inference that can scale from development to production while maintaining compatibility with standard AI/ML tooling.