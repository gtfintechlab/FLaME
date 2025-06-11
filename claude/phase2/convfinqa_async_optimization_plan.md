# ConvFinQA Async Optimization Plan - Final Version

## 1. Architecture Overview

### 1.1 Core Requirements
- **Multi-turn conversations**: Each conversation requires 5 sequential API calls
- **Dependencies**: Q2 depends on A1, Q3 depends on A2, etc.
- **Scale**: ~1000+ conversations in test set
- **Goal**: Maximize throughput while maintaining reliability

### 1.2 LiteLLM Features to Leverage
- **Router with load balancing** for distributed processing
- **Async operations** (`acompletion`) for concurrent conversations
- **Retry mechanisms** with exponential backoff
- **Fallback strategies** for resilience
- **Redis integration** for production usage tracking
- **Rate limit aware routing** with usage-based-routing-v2

## 2. Implementation Strategy

### 2.1 Enhanced Conversation State Management

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time
import json

@dataclass
class ConversationState:
    """Enhanced state tracking with LiteLLM integration and checkpointing."""
    id: str
    context: str
    questions: List[str]
    answers: List[str] = field(default_factory=list)
    current_turn: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # LiteLLM specific tracking
    model_used: List[str] = field(default_factory=list)  # Track which model served each turn
    api_latencies: List[float] = field(default_factory=list)  # Track latency per turn
    retry_counts: List[int] = field(default_factory=list)  # Track retries per turn
    error: Optional[str] = None
    
    # Checkpointing for resumability
    last_checkpoint: Optional[Dict] = None
    checkpoint_path: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return len(self.answers) == len(self.questions) or self.error is not None
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def get_next_prompt(self) -> str:
        """Build prompt with full conversation history."""
        prompt = self.context + "\n"
        for i in range(self.current_turn):
            prompt += f"\nQ{i+1}: {self.questions[i]}\nA{i+1}: {self.answers[i]}"
        prompt += f"\nQ{self.current_turn+1}: {self.questions[self.current_turn]}\nA{self.current_turn+1}:"
        return prompt
    
    def to_checkpoint(self) -> Dict:
        """Serialize state for checkpointing."""
        return {
            "id": self.id,
            "current_turn": self.current_turn,
            "answers": self.answers,
            "model_used": self.model_used,
            "api_latencies": self.api_latencies,
            "retry_counts": self.retry_counts,
            "checkpoint_time": time.time()
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint: Dict, original: 'ConversationState') -> 'ConversationState':
        """Restore from checkpoint."""
        restored = cls(
            id=original.id,
            context=original.context,
            questions=original.questions
        )
        restored.current_turn = checkpoint["current_turn"]
        restored.answers = checkpoint["answers"]
        restored.model_used = checkpoint["model_used"]
        restored.api_latencies = checkpoint["api_latencies"]
        restored.retry_counts = checkpoint["retry_counts"]
        restored.last_checkpoint = checkpoint
        return restored
```

### 2.2 Production-Ready Router Configuration

```python
from litellm import Router
import os

class ConvFinQARouter:
    """Production router configuration with multiple deployments."""
    
    def __init__(self, args):
        # Multiple model deployments for load distribution and failover
        self.model_list = [
            {
                "model_name": "llama-scout",
                "litellm_params": {
                    "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "api_key": os.getenv("TOGETHER_API_KEY"),
                    "max_parallel_requests": 20,
                    "tpm": 100000,
                    "rpm": 60,
                },
                "weight": 2  # Primary deployment
            },
            {
                "model_name": "llama-scout",
                "litellm_params": {
                    "model": "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "api_key": os.getenv("TOGETHER_API_KEY_2"),
                    "max_parallel_requests": 15,
                    "tpm": 80000,
                    "rpm": 50,
                },
                "weight": 1  # Secondary deployment
            },
            # Fallback deployment with different provider
            {
                "model_name": "llama-scout-fallback",
                "litellm_params": {
                    "model": "openai/gpt-4",  # Fallback to different provider
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_parallel_requests": 10,
                },
                "weight": 0.5  # Lower priority fallback
            }
        ]
        
        # Initialize router with production features
        self.router = Router(
            model_list=self.model_list,
            routing_strategy="usage-based-routing-v2",
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=os.getenv("REDIS_PORT"),
            
            # Reliability settings
            num_retries=3,
            retry_after=2,
            allowed_fails=3,
            cooldown_time=60,
            
            # Performance settings
            enable_pre_call_checks=True,
            cache_responses=True,
            cache_kwargs={"ttl": 3600},  # 1 hour cache
            
            # Fallback configuration
            fallbacks=["llama-scout", "llama-scout-fallback"],
            context_window_fallback_dict={
                "llama-scout": "llama-scout-fallback"
            },
            
            # Default parameters
            default_litellm_params={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "timeout": 30,
            }
        )
```

### 2.3 Async Conversation Processor with Monitoring

```python
import asyncio
from typing import Dict, Any
import aiofiles
import json

class ConversationMetrics:
    """Track detailed metrics for conversations."""
    def __init__(self):
        self.start_time = time.time()
        self.api_calls = 0
        self.successful_turns = 0
        self.failed_turns = 0
        self.total_tokens = 0
        self.model_distribution = {}
        
    def record_turn(self, model: str, success: bool, tokens: int = 0):
        self.api_calls += 1
        if success:
            self.successful_turns += 1
        else:
            self.failed_turns += 1
        self.total_tokens += tokens
        self.model_distribution[model] = self.model_distribution.get(model, 0) + 1

class AsyncConversationProcessor:
    """Processes multi-turn conversations with comprehensive monitoring."""
    
    def __init__(self, router: ConvFinQARouter, args, checkpoint_dir: str = "./checkpoints"):
        self.router = router
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Dynamic concurrency control
        self.max_concurrent_conversations = min(
            args.batch_size // 5,  # Each conversation needs 5 API calls
            50  # Hard limit
        )
        self.conversation_semaphore = asyncio.Semaphore(self.max_concurrent_conversations)
        
        # Per-turn concurrency (total API calls in flight)
        self.api_semaphore = asyncio.Semaphore(args.batch_size)
        
        # Metrics tracking
        self.metrics = ConversationMetrics()
        
    async def checkpoint_conversation(self, conv_state: ConversationState):
        """Async checkpoint saving."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{conv_state.id}_checkpoint.json"
        )
        async with aiofiles.open(checkpoint_path, 'w') as f:
            await f.write(json.dumps(conv_state.to_checkpoint()))
        conv_state.checkpoint_path = checkpoint_path
        
    async def restore_from_checkpoint(self, conv_state: ConversationState) -> ConversationState:
        """Restore conversation from checkpoint if exists."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{conv_state.id}_checkpoint.json"
        )
        if os.path.exists(checkpoint_path):
            async with aiofiles.open(checkpoint_path, 'r') as f:
                checkpoint = json.loads(await f.read())
            return ConversationState.from_checkpoint(checkpoint, conv_state)
        return conv_state
        
    async def process_single_turn(
        self, 
        conv_state: ConversationState,
        prompt: str,
        turn_num: int
    ) -> str:
        """Process a single turn with comprehensive error handling."""
        
        async with self.api_semaphore:
            start_time = time.time()
            
            try:
                # Use router for intelligent routing
                response = await self.router.router.acompletion(
                    model="llama-scout",
                    messages=[{"role": "user", "content": prompt}],
                    metadata={
                        "conversation_id": conv_state.id,
                        "turn": turn_num,
                    }
                )
                
                # Track metrics
                latency = time.time() - start_time
                conv_state.api_latencies.append(latency)
                conv_state.model_used.append(response.model)
                conv_state.retry_counts.append(0)
                
                # Record in global metrics
                self.metrics.record_turn(
                    response.model, 
                    True, 
                    response.usage.total_tokens
                )
                
                # Extract answer
                answer = response.choices[0].message.content
                return answer
                
            except Exception as e:
                logger.error(f"Turn {turn_num} failed for conv {conv_state.id}: {str(e)}")
                self.metrics.record_turn("failed", False)
                conv_state.retry_counts.append(-1)
                raise
                
    async def process_conversation(self, conv_state: ConversationState) -> ConversationState:
        """Process entire conversation with checkpointing and monitoring."""
        
        async with self.conversation_semaphore:
            try:
                # Try to restore from checkpoint
                conv_state = await self.restore_from_checkpoint(conv_state)
                
                while conv_state.current_turn < len(conv_state.questions):
                    # Build prompt with conversation history
                    prompt = conv_state.get_next_prompt()
                    
                    # Process turn with exponential backoff
                    for attempt in range(3):
                        try:
                            answer = await self.process_single_turn(
                                conv_state, 
                                prompt, 
                                conv_state.current_turn
                            )
                            conv_state.answers.append(answer)
                            conv_state.current_turn += 1
                            
                            # Checkpoint after each successful turn
                            await self.checkpoint_conversation(conv_state)
                            break
                            
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                conv_state.error = str(e)
                                raise
                            await asyncio.sleep(2 ** attempt)
                
                conv_state.end_time = time.time()
                
                # Clean up checkpoint on success
                if conv_state.checkpoint_path and os.path.exists(conv_state.checkpoint_path):
                    os.remove(conv_state.checkpoint_path)
                    
                return conv_state
                
            except Exception as e:
                conv_state.error = str(e)
                conv_state.end_time = time.time()
                return conv_state
```

### 2.4 Advanced Batch Processing with Performance Monitoring

```python
class PerformanceMonitor:
    """Real-time performance monitoring."""
    def __init__(self):
        self.start_time = None
        self.checkpoints = []
        
    def start(self):
        self.start_time = time.time()
        
    def checkpoint(self, name: str, metadata: Dict[str, Any] = None):
        self.checkpoints.append({
            "name": name,
            "time": time.time() - self.start_time,
            "metadata": metadata or {}
        })
        
    def get_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        return {
            "total_time": total_time,
            "checkpoints": self.checkpoints
        }

class ConvFinQABatchProcessor:
    """Advanced batch processing with monitoring and optimization."""
    
    def __init__(self, router: ConvFinQARouter, args):
        self.router = router
        self.processor = AsyncConversationProcessor(router, args)
        self.args = args
        self.monitor = PerformanceMonitor()
        
    async def process_with_pipelining(self, conversations: List[ConversationState]):
        """Process conversations with intelligent scheduling and monitoring."""
        
        self.monitor.start()
        
        # Group by conversation length for optimal scheduling
        short_convs = [c for c in conversations if len(c.questions) <= 3]
        long_convs = [c for c in conversations if len(c.questions) > 3]
        
        self.monitor.checkpoint("grouping_complete", {
            "short_conversations": len(short_convs),
            "long_conversations": len(long_convs)
        })
        
        # Initialize task tracking
        all_tasks = []
        task_to_conv = {}  # Map tasks to conversations for tracking
        
        # Start long conversations first (they take longer)
        for conv in long_convs[:self.processor.max_concurrent_conversations // 2]:
            task = asyncio.create_task(self.processor.process_conversation(conv))
            all_tasks.append(task)
            task_to_conv[task] = conv
            
        # Fill with short conversations
        remaining_slots = self.processor.max_concurrent_conversations - len(all_tasks)
        for conv in short_convs[:remaining_slots]:
            task = asyncio.create_task(self.processor.process_conversation(conv))
            all_tasks.append(task)
            task_to_conv[task] = conv
            
        # Queue remaining conversations
        remaining = short_convs[remaining_slots:] + long_convs[self.processor.max_concurrent_conversations // 2:]
        remaining_queue = asyncio.Queue()
        for conv in remaining:
            await remaining_queue.put(conv)
            
        self.monitor.checkpoint("initial_tasks_started", {
            "active_tasks": len(all_tasks),
            "queued_conversations": remaining_queue.qsize()
        })
        
        # Process with dynamic scheduling
        completed = []
        pending = set(all_tasks)
        
        with tqdm(total=len(conversations), desc="Processing conversations") as pbar:
            while pending or not remaining_queue.empty():
                # Wait for any task to complete
                done, pending = await asyncio.wait(
                    pending, 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    result = await task
                    completed.append(result)
                    pbar.update(1)
                    
                    # Log completion
                    conv = task_to_conv[task]
                    logger.debug(
                        f"Completed {conv.id}: "
                        f"{len(result.answers)}/{len(result.questions)} turns, "
                        f"duration: {result.duration:.1f}s"
                    )
                    
                    # Start new conversation if available
                    if not remaining_queue.empty():
                        next_conv = await remaining_queue.get()
                        new_task = asyncio.create_task(
                            self.processor.process_conversation(next_conv)
                        )
                        pending.add(new_task)
                        task_to_conv[new_task] = next_conv
                        
        self.monitor.checkpoint("all_conversations_complete", {
            "total_completed": len(completed),
            "successful": sum(1 for c in completed if c.error is None)
        })
        
        return completed
```

### 2.5 Production-Ready Main Function

```python
async def convfinqa_inference_optimized(args):
    """Production-ready ConvFinQA inference with comprehensive features."""
    
    # Initialize components
    router = ConvFinQARouter(args)
    batch_processor = ConvFinQABatchProcessor(router, args)
    
    # Load and prepare dataset
    logger.info("Loading ConvFinQA dataset...")
    dataset = safe_load_dataset("gtfintechlab/convfinqa", trust_remote_code=True)
    
    # Prepare conversation states with full data extraction
    conversations = []
    for idx, entry in enumerate(dataset["test"]):
        # Extract all questions (up to 5)
        questions = []
        for i in range(5):
            q_key = f"question_{i}"
            if q_key in entry and entry[q_key]:
                questions.append(str(entry[q_key]))
        
        # Build comprehensive context
        pre_text = " ".join(entry["pre_text"]) if entry["pre_text"] else ""
        post_text = " ".join(entry["post_text"]) if entry["post_text"] else ""
        
        # Format table data
        table_text = ""
        if entry.get("table_ori"):
            table_rows = []
            for row in entry["table_ori"]:
                table_rows.append(" | ".join(map(str, row)))
            table_text = "\n".join(table_rows)
        
        # Combine all context
        context = f"""Context:
{pre_text}

Table:
{table_text}

{post_text}"""
        
        # Create conversation state
        conv_state = ConversationState(
            id=f"conv_{idx:04d}",
            context=context.strip(),
            questions=questions
        )
        conversations.append(conv_state)
    
    total_questions = sum(len(c.questions) for c in conversations)
    logger.info(f"Prepared {len(conversations)} conversations with {total_questions} total questions")
    
    # Process all conversations
    start_time = time.time()
    
    try:
        results = await batch_processor.process_with_pipelining(conversations)
        
        # Generate comprehensive metrics
        total_time = time.time() - start_time
        performance_summary = batch_processor.monitor.get_summary()
        
        # Calculate detailed metrics
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        total_api_calls = sum(len(r.answers) for r in results)
        
        # API latency statistics
        all_latencies = []
        for r in successful:
            all_latencies.extend(r.api_latencies)
        
        avg_latency = np.mean(all_latencies) if all_latencies else 0
        p50_latency = np.percentile(all_latencies, 50) if all_latencies else 0
        p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0
        
        # Model usage distribution
        model_usage = {}
        for r in results:
            for model in r.model_used:
                model_usage[model] = model_usage.get(model, 0) + 1
        
        # Log comprehensive results
        logger.info("=" * 50)
        logger.info("ConvFinQA Inference Complete")
        logger.info("=" * 50)
        logger.info(f"Total conversations: {len(conversations)}")
        logger.info(f"Successful: {len(successful)} ({len(successful)/len(conversations)*100:.1f}%)")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average time per conversation: {total_time/len(conversations):.2f}s")
        logger.info(f"Total API calls: {total_api_calls}")
        logger.info(f"API calls/second: {total_api_calls/total_time:.1f}")
        logger.info(f"Average API latency: {avg_latency:.3f}s (P50: {p50_latency:.3f}s, P95: {p95_latency:.3f}s)")
        logger.info(f"Model distribution: {model_usage}")
        logger.info(f"Global metrics: {batch_processor.processor.metrics.__dict__}")
        
        # Build comprehensive results DataFrame
        df_data = []
        for conv in results:
            # Extract individual answers for each question
            for q_idx, question in enumerate(conv.questions):
                answer = conv.answers[q_idx] if q_idx < len(conv.answers) else None
                actual_answer_key = f"answer_{q_idx}"
                actual_answer = dataset["test"][int(conv.id.split("_")[1])].get(actual_answer_key)
                
                df_data.append({
                    "conversation_id": conv.id,
                    "question_index": q_idx,
                    "context": conv.context,
                    "question": question,
                    "response": answer,
                    "actual_label": actual_answer,
                    "model_used": conv.model_used[q_idx] if q_idx < len(conv.model_used) else None,
                    "api_latency": conv.api_latencies[q_idx] if q_idx < len(conv.api_latencies) else None,
                    "retry_count": conv.retry_counts[q_idx] if q_idx < len(conv.retry_counts) else None,
                    "conversation_duration": conv.duration,
                    "error": conv.error if q_idx == len(conv.answers) else None
                })
        
        df = pd.DataFrame(df_data)
        
        # Add metadata
        df.attrs['metrics'] = {
            'total_time': total_time,
            'successful_conversations': len(successful),
            'failed_conversations': len(failed),
            'total_api_calls': total_api_calls,
            'average_latency': avg_latency,
            'model_distribution': model_usage,
            'performance_checkpoints': performance_summary
        }
        
        return df
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise
    finally:
        # Cleanup
        await router.router.close()
        
        # Save any remaining checkpoints for recovery
        checkpoint_files = os.listdir(batch_processor.processor.checkpoint_dir)
        if checkpoint_files:
            logger.warning(f"Found {len(checkpoint_files)} checkpoint files - some conversations may have failed")
```

## 3. Key Features of Final Implementation

### 3.1 **Comprehensive State Management**
- Full conversation state with checkpointing
- Detailed metric tracking per turn
- Serialization for fault tolerance

### 3.2 **Production Router Configuration**
- Multiple deployments with different API keys
- Cross-provider fallback (Together AI → OpenAI)
- Redis-based usage tracking
- Smart routing with usage-based-routing-v2

### 3.3 **Advanced Processing Features**
- Checkpoint/restore for resumability
- Dynamic conversation scheduling
- Real-time performance monitoring
- Comprehensive error handling

### 3.4 **Detailed Metrics and Monitoring**
- Per-turn latency tracking
- Model distribution analysis
- Success/failure rates
- Performance checkpoints

## 4. Expected Performance

### Baseline (Sequential):
- 1000 conversations × 5 turns × 1s = 5000 seconds (83 minutes)

### Optimized Performance:
- **Conservative (20 concurrent)**: ~250 seconds (20x speedup)
- **Moderate (50 concurrent)**: ~100 seconds (50x speedup)  
- **Aggressive (100 concurrent with Router)**: ~50 seconds (100x speedup)

### Additional Benefits:
- **Fault tolerance**: Can resume from checkpoints
- **Load distribution**: Spreads load across multiple API keys
- **Automatic failover**: Falls back to different providers
- **Detailed insights**: Comprehensive metrics for optimization

## 5. Implementation Phases

### Phase 1: Basic Async
1. Implement `ConversationState` class
2. Create basic async processor
3. Test with small dataset

### Phase 2: LiteLLM Router Integration
1. Set up Router with multiple deployments
2. Implement retry and fallback logic
3. Add Redis for production metrics

### Phase 3: Advanced Optimizations
1. Implement conversation pipelining
2. Add checkpointing for resumability
3. Deploy monitoring and alerting

### Phase 4: Production Hardening
1. Error handling and recovery
2. Performance tuning
3. Documentation and testing

This final plan combines the best aspects of both iterations, providing a production-ready implementation that leverages all of LiteLLM's advanced features for maximum performance and reliability.