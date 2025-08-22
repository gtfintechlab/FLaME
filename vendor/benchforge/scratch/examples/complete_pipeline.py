"""Complete pipeline example demonstrating BenchForge features.

This example shows professional-grade data management, configuration,
and inference pipeline with all components working together.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# BenchForge imports
from bench_forge import (
    # Configuration
    BenchForgeConfig,
    get_config,
    set_config,
    setup_logging,
    # Data Management
    LoaderConfig,
    HuggingFaceLoader,
    load_dataset,
    # Processing
    ProcessorConfig,
    TextProcessor,
    DatasetProcessor,
    SplitConfig,
    DataSplitter,
    CacheConfig,
    CacheManager,
    ResponseCache,
    # Validation
    InputValidator,
    OutputValidator,
    ParallelConfig,
    ParallelExecutor,
    AsyncExecutor,
    parallel_map,
)


def setup_environment():
    """Setup BenchForge environment with professional configuration."""
    print("\n" + "=" * 80)
    print("PHASE 3: DATA & CONFIG - COMPLETE PIPELINE EXAMPLE")
    print("=" * 80)

    # Configure logging
    setup_logging(level="INFO", colored=True, log_file=Path("benchforge_phase3.log"))

    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 3 pipeline demonstration")

    # Setup configuration
    config = BenchForgeConfig(
        project_name="benchforge_phase3_demo",
        version="0.3.0",
        environment="development",
        debug_mode=True,
        # Data settings
        cache_dir="./cache/phase3",
        output_dir="./outputs/phase3",
        data_dir="./data",
        # Processing settings
        default_batch_size=20,
        max_parallel_workers=4,
        # Caching
        enable_cache=True,
        cache_ttl=3600,
        response_cache_enabled=True,
        # Validation
        validate_inputs=True,
        validate_outputs=True,
        # Performance
        profile_enabled=True,
    )

    # Set global configuration
    set_config(config)

    print("\n✅ Environment configured successfully")
    print(f"   Project: {config.project_name}")
    print(f"   Version: {config.version}")
    print(f"   Environment: {config.environment}")

    return config, logger


def demonstrate_data_loading(logger):
    """Demonstrate professional data loading with multiple formats."""
    print("\n" + "-" * 80)
    print("1. DATA LOADING DEMONSTRATION")
    print("-" * 80)

    # Configure loader
    loader_config = LoaderConfig(
        cache_dir=Path("./cache/datasets"),
        validate_on_load=True,
        max_retries=3,
    )

    # Example 1: HuggingFace dataset loading
    print("\n📁 Loading HuggingFace dataset with caching and validation...")
    try:
        HuggingFaceLoader(loader_config)

        # Mock dataset for demonstration
        dataset = [
            {"text": "This is a positive example.", "label": 1},
            {"text": "This is a negative example.", "label": 0},
            {"text": "Another positive text.", "label": 1},
            {"text": "More negative content.", "label": 0},
            {"text": "Neutral statement here.", "label": 2},
        ] * 20  # Create 100 samples

        print(f"   Loaded {len(dataset)} samples")
        logger.info(f"Dataset loaded: {len(dataset)} samples")

    except Exception as e:
        logger.warning(f"HuggingFace loading skipped (mock): {e}")
        dataset = create_mock_dataset(100)

    # Example 2: Factory pattern loading
    print("\n🏭 Using loader factory for automatic format detection...")

    # Save mock data for loading
    import json

    json_path = Path("./temp_data.json")
    with open(json_path, "w") as f:
        json.dump(dataset[:10], f)

    loaded_data = load_dataset(str(json_path), loader_type="json")
    print(f"   Factory loaded {len(loaded_data)} samples from JSON")

    # Cleanup
    json_path.unlink()

    return dataset


def demonstrate_data_processing(dataset, logger):
    """Demonstrate data processing with text cleaning and transformations."""
    print("\n" + "-" * 80)
    print("2. DATA PROCESSING DEMONSTRATION")
    print("-" * 80)

    # Configure processor
    processor_config = ProcessorConfig(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=False,  # Keep for this demo
        normalize_whitespace=True,
        max_length=100,
    )

    # Text processing
    print("\n🔧 Processing text data with cleaning and normalization...")
    text_processor = TextProcessor(processor_config)

    # Process sample texts
    sample_texts = [item["text"] for item in dataset[:5]]
    processed_texts = []

    for text in sample_texts:
        processed = text_processor.process(text)
        processed_texts.append(processed)
        print(f"   Original: {text[:50]}...")
        print(f"   Processed: {processed[:50]}...")
        print()

    # Dataset processing
    print("📊 Processing entire dataset with transformations...")
    DatasetProcessor(processor_config)

    # Process with statistics
    processed_dataset = []
    for item in dataset:
        processed_item = {
            "text": text_processor.process(item["text"]),
            "label": item["label"],
            "original_length": len(item["text"]),
        }
        processed_dataset.append(processed_item)

    stats = text_processor.get_stats()
    print("\n   Processing Statistics:")
    print(f"   - Texts processed: {stats['texts_processed']}")
    print(f"   - Chars removed: {stats['chars_removed']}")
    print(f"   - Texts truncated: {stats['texts_truncated']}")

    logger.info(f"Processed {len(processed_dataset)} samples")

    return processed_dataset


def demonstrate_data_splitting(dataset, logger):
    """Demonstrate data splitting with stratification."""
    print("\n" + "-" * 80)
    print("3. DATA SPLITTING DEMONSTRATION")
    print("-" * 80)

    # Configure splitter
    split_config = SplitConfig(
        random_state=42,
        shuffle=True,
        stratify="label",
        validate_splits=True,
        verbose=True,
    )

    splitter = DataSplitter(split_config)

    # Example 1: Train/Val/Test split
    print("\n✂️ Splitting data into train/validation/test sets...")

    splits = {
        "train": 0.7,
        "validation": 0.15,
        "test": 0.15,
    }

    split_result = splitter.split(
        dataset,
        splits,
        stratify="label",  # Stratify by label
    )

    print("\n   Split Results:")
    for name, data in split_result.items():
        print(
            f"   - {name}: {len(data)} samples ({len(data) / len(dataset) * 100:.1f}%)"
        )

    # Example 2: K-fold cross-validation
    print("\n🔄 Creating 5-fold cross-validation splits...")

    folds = splitter.k_fold_split(dataset, n_folds=5)

    print(f"   Created {len(folds)} folds:")
    for i, (train, val) in enumerate(folds):
        print(f"   - Fold {i + 1}: Train={len(train)}, Val={len(val)}")

    # Get statistics
    stats = splitter.get_stats()
    print("\n   Splitter Statistics:")
    print(f"   - Total splits created: {stats['splits_created']}")
    print(f"   - Total samples split: {stats['total_samples_split']}")
    print(f"   - Stratified splits: {stats['stratified_splits']}")

    logger.info("Data splitting completed successfully")

    return split_result


def demonstrate_caching(logger):
    """Demonstrate caching system with TTL and namespaces."""
    print("\n" + "-" * 80)
    print("4. CACHING DEMONSTRATION")
    print("-" * 80)

    # Configure cache
    cache_config = CacheConfig(
        cache_dir=Path("./cache/demo"),
        default_ttl=300,  # 5 minutes
        max_size_mb=100,
        enable_stats=True,
        compression=True,
    )

    cache_manager = CacheManager(cache_config)

    # Example 1: Basic caching
    print("\n💾 Demonstrating basic caching with TTL...")

    # Cache some data
    cache_manager.set(
        "model_output_1",
        {"prediction": "positive", "score": 0.95},
        namespace="predictions",
    )
    cache_manager.set(
        "model_output_2",
        {"prediction": "negative", "score": 0.87},
        namespace="predictions",
    )

    # Retrieve cached data
    cached = cache_manager.get("model_output_1", namespace="predictions")
    print(f"   Retrieved from cache: {cached}")

    # Example 2: Response caching for LLM
    print("\n🤖 Demonstrating LLM response caching...")

    response_cache = ResponseCache(cache_manager)

    # Simulate caching responses
    prompt = "Analyze the sentiment of: 'This product is amazing!'"
    model = "gpt-3.5-turbo"

    # Check cache
    cached_response = response_cache.get_response(prompt, model, temperature=0.0)
    if cached_response:
        print(f"   Cache HIT: {cached_response}")
    else:
        print("   Cache MISS - would call LLM here")
        # Simulate LLM response
        response = "Positive sentiment (confidence: 0.95)"
        response_cache.cache_response(prompt, model, response, temperature=0.0)
        print("   Cached response for future use")

    # Example 3: Cache statistics
    print("\n📊 Cache Statistics:")
    stats = cache_manager.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    # Cleanup specific namespace
    cache_manager.clear(namespace="predictions")

    logger.info("Caching demonstration completed")


def demonstrate_validation(dataset, logger):
    """Demonstrate comprehensive validation system."""
    print("\n" + "-" * 80)
    print("5. VALIDATION DEMONSTRATION")
    print("-" * 80)

    # Create validators
    input_validator = InputValidator(strict=True)
    output_validator = OutputValidator()

    # Example 1: Dataset validation
    print("\n✅ Validating dataset format and content...")

    result = input_validator.validate_dataset(
        dataset, min_size=10, max_size=1000, required_fields=["text", "label"]
    )

    if result.is_valid:
        print("   Dataset validation PASSED")
    else:
        print(f"   Dataset validation FAILED: {result.errors}")

    print(f"   Metadata: {result.metadata}")

    # Example 2: Prompt validation
    print("\n📝 Validating prompt format...")

    test_prompt = """You are a helpful assistant.
    
    Task: Analyze the following text for sentiment.
    Text: {text}
    
    Provide your analysis in JSON format.
    """

    prompt_result = input_validator.validate_prompt(
        test_prompt, min_length=10, max_length=10000
    )

    print(f"   Prompt validation: {'PASSED' if prompt_result.is_valid else 'FAILED'}")
    print(f"   - Length: {prompt_result.metadata['length']} chars")
    print(f"   - Words: {prompt_result.metadata['word_count']}")
    print(f"   - Lines: {prompt_result.metadata['line_count']}")

    # Example 3: Output validation
    print("\n📊 Validating model outputs...")

    # Mock results DataFrame
    import pandas as pd

    results_df = pd.DataFrame(
        {
            "input": [item["text"] for item in dataset[:5]],
            "prompt": [test_prompt] * 5,
            "raw_response": ["positive"] * 3 + ["negative"] * 2,
            "extracted_response": ["positive"] * 3 + ["negative"] * 2,
        }
    )

    output_result = output_validator.validate_results(results_df)

    if output_result.is_valid:
        print("   Output validation PASSED")
        print(f"   Shape: {output_result.metadata['shape']}")
    else:
        print(f"   Output validation FAILED: {output_result.errors}")

    # Get validator statistics
    print("\n📈 Validator Statistics:")
    stats = input_validator.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    logger.info("Validation demonstration completed")


def demonstrate_parallel_execution(dataset, logger):
    """Demonstrate parallel and async execution capabilities."""
    print("\n" + "-" * 80)
    print("6. PARALLEL EXECUTION DEMONSTRATION")
    print("-" * 80)

    # Configure parallel executor
    parallel_config = ParallelConfig(
        max_workers=4,
        executor_type="thread",
        preserve_order=True,
        progress_callback=lambda curr, total: print(
            f"   Progress: {curr}/{total}", end="\r"
        ),
    )

    # Example 1: Parallel text processing
    print("\n⚡ Processing texts in parallel...")

    def process_text_item(item):
        """Process a single text item."""
        import time

        time.sleep(0.01)  # Simulate processing
        return {
            "original": item["text"],
            "processed": item["text"].lower().strip(),
            "length": len(item["text"]),
        }

    with ParallelExecutor(parallel_config) as executor:
        result = executor.map(process_text_item, dataset[:20])

        if result.all_successful:
            print(f"\n   ✅ Processed {len(result.successful)} items successfully")
            print(f"   Duration: {result.duration:.2f}s")
            print(f"   Success rate: {result.success_rate * 100:.1f}%")
        else:
            print(f"\n   ⚠️ Some items failed: {len(result.failed)}")

    # Example 2: Batch processing
    print("\n📦 Batch processing with parallel execution...")

    def process_batch(batch):
        """Process a batch of items."""
        return [{"id": i, "result": "processed"} for i, _ in enumerate(batch)]

    with ParallelExecutor(parallel_config) as executor:
        batch_result = executor.batch_process(
            process_batch, dataset[:50], batch_size=10
        )

        print(f"   Processed {len(batch_result.successful)} items in batches")
        print(f"   Duration: {batch_result.duration:.2f}s")

    # Example 3: Async execution (demonstration)
    print("\n🔄 Demonstrating async execution...")

    async def async_demo():
        """Async execution demonstration."""
        async_executor = AsyncExecutor(max_concurrent=3)

        # Create async tasks
        async def process_async(item):
            await asyncio.sleep(0.01)
            return f"Processed: {item}"

        tasks = [process_async(i) for i in range(10)]
        results = await async_executor.gather(*tasks)

        print(f"   Async processed {len(results)} items")

        # Get statistics
        stats = async_executor.get_stats()
        print(f"   Async stats: {stats}")

    # Run async demo
    asyncio.run(async_demo())

    # Get parallel executor statistics
    print("\n📊 Parallel Execution Statistics:")
    with ParallelExecutor(parallel_config) as executor:
        stats = executor.get_stats()
        for key, value in stats.items():
            print(f"   - {key}: {value}")

    logger.info("Parallel execution demonstration completed")


def demonstrate_complete_pipeline(logger):
    """Demonstrate complete integrated pipeline."""
    print("\n" + "=" * 80)
    print("7. COMPLETE INTEGRATED PIPELINE")
    print("=" * 80)

    print("\n🚀 Running complete pipeline with all Phase 3 components...")

    # Step 1: Load configuration
    config = get_config()
    print(f"\n1️⃣ Configuration loaded: {config.project_name} v{config.version}")

    # Step 2: Setup cache
    cache_manager = CacheManager()
    print("2️⃣ Cache manager initialized")

    # Step 3: Load data with caching
    cache_key = "demo_dataset"
    dataset = cache_manager.get(cache_key)

    if dataset is None:
        print("3️⃣ Loading fresh dataset...")
        dataset = create_mock_dataset(200)
        cache_manager.set(cache_key, dataset, ttl=300)
    else:
        print("3️⃣ Dataset loaded from cache")

    # Step 4: Validate data
    validator = InputValidator()
    validation = validator.validate_dataset(dataset, required_fields=["text", "label"])
    validation.raise_if_invalid()
    print("4️⃣ Dataset validation passed")

    # Step 5: Process data in parallel
    print("5️⃣ Processing data in parallel...")

    def process_item(item):
        processor = TextProcessor()
        return {**item, "processed_text": processor.process(item["text"])}

    processed = parallel_map(process_item, dataset[:50], max_workers=4)
    print(f"   Processed {len(processed)} items")

    # Step 6: Split data
    splitter = DataSplitter()
    train, val, test = splitter.train_val_test_split(
        processed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"6️⃣ Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Step 7: Cache results
    ResponseCache()
    print("7️⃣ Response cache ready for inference")

    print("\n✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
    print("\nPhase 3 Components Demonstrated:")
    print("  ✓ Configuration Management")
    print("  ✓ Data Loading with Multiple Formats")
    print("  ✓ Text Processing and Cleaning")
    print("  ✓ Dataset Splitting with Stratification")
    print("  ✓ Caching with TTL and Namespaces")
    print("  ✓ Comprehensive Validation")
    print("  ✓ Parallel and Async Execution")

    logger.info("Complete pipeline demonstration finished")


def create_mock_dataset(size: int) -> List[Dict[str, Any]]:
    """Create mock dataset for demonstration."""
    import random

    texts = [
        "This is a great product!",
        "Terrible experience, would not recommend.",
        "Average quality, nothing special.",
        "Absolutely love it!",
        "Waste of money.",
        "Decent value for the price.",
        "Outstanding service!",
        "Very disappointed.",
        "It's okay, could be better.",
        "Highly recommended!",
    ]

    labels = [1, 0, 2, 1, 0, 2, 1, 0, 2, 1]  # 1=positive, 0=negative, 2=neutral

    dataset = []
    for i in range(size):
        idx = i % len(texts)
        dataset.append(
            {
                "text": texts[idx],
                "label": labels[idx],
                "id": f"sample_{i:04d}",
            }
        )

    random.shuffle(dataset)
    return dataset


def main():
    """Run complete Phase 3 demonstration."""
    try:
        # Setup
        config, logger = setup_environment()

        # Demonstrations
        dataset = demonstrate_data_loading(logger)
        processed_dataset = demonstrate_data_processing(dataset, logger)
        demonstrate_data_splitting(processed_dataset, logger)
        demonstrate_caching(logger)
        demonstrate_validation(dataset, logger)
        demonstrate_parallel_execution(dataset, logger)

        # Complete pipeline
        demonstrate_complete_pipeline(logger)

        print("\n" + "=" * 80)
        print("PHASE 3 DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\n✨ All Phase 3 features demonstrated successfully!")
        print("\nKey Capabilities Shown:")
        print("  • Professional configuration management")
        print("  • Multi-format data loading with factory pattern")
        print("  • Advanced text processing and cleaning")
        print("  • Stratified data splitting and k-fold CV")
        print("  • Response caching with TTL and namespaces")
        print("  • Comprehensive input/output validation")
        print("  • Parallel and async execution with progress tracking")

    except Exception as e:
        logging.error(f"Demonstration failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
