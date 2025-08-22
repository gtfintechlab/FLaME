"""Complete example of FLAME QA and NER task migration to BenchForge.

This example demonstrates how to use the migrated FLAME tasks:
1. ConvFinQA - Multi-turn financial conversations
2. FinQA - Financial question answering with tables
3. FiNER - Financial named entity recognition
4. FinEntity - Financial entity classification
5. EDTSum - Earnings call question answering

Each task showcases the improved extraction, evaluation, and integration patterns.
"""

import logging

from bench_forge.llm.config import LLMConfig
from bench_forge.flame.adapter import FLAMEAdapter
from bench_forge.flame.evaluation import FLAMEEvaluator
from bench_forge.flame.tasks import (
    ConvFinQATask,
    ConvFinQAConfig,
    FinQATask,
    FinQAConfig,
    FiNERTask,
    FiNERConfig,
    FinEntityTask,
    FinEntityConfig,
    EDTSumTask,
    EDTSumConfig,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_convfinqa_example():
    """Example: ConvFinQA multi-turn conversation QA."""
    logger.info("=== ConvFinQA Example: Multi-turn Financial Conversations ===")

    # Configuration
    config = ConvFinQAConfig(
        name="convfinqa",
        huggingface_dataset="gtfintechlab/convfinqa",
        dataset_split="dev",
        num_samples=5,  # Small example
        prompt_format="zero_shot",
        extraction_strategy="numeric",
    )

    # Create task
    task = ConvFinQATask(config)

    # Sample data structure for ConvFinQA
    sample_data = {
        "pre_text": ["Apple Inc. reported strong quarterly results"],
        "post_text": ["The company continues to see growth in all segments"],
        "table_ori": [
            ["Quarter", "Revenue", "Profit"],
            ["Q1 2023", "$97.3B", "$24.2B"],
            ["Q2 2023", "$81.8B", "$19.9B"],
        ],
        "question_0": "What was the revenue in Q1 2023?",
        "answer_0": "$97.3B",
        "question_1": "How much did revenue decrease from Q1 to Q2?",
        "answer_1": "$15.5B",
    }

    # Generate prompt
    prompt = task.create_prompt(sample_data)
    logger.info(f"ConvFinQA Prompt Preview:\n{prompt[:300]}...")

    # Example response processing
    mock_response = "Based on the financial data, revenue decreased from $97.3B in Q1 to $81.8B in Q2, which is a decrease of $15.5B."
    extracted = task.extract_response(mock_response, sample_data)
    logger.info(f"Extracted Answer: {extracted}")

    return {
        "task": "convfinqa",
        "extraction_success": extracted is not None,
        "extracted_value": extracted,
    }


def run_finqa_example():
    """Example: FinQA financial table reasoning."""
    logger.info("=== FinQA Example: Financial Table Reasoning ===")

    # Configuration
    config = FinQAConfig(
        name="finqa",
        huggingface_dataset="gtfintechlab/finqa",
        dataset_split="test",
        num_samples=5,
        prompt_format="few_shot",
        extraction_strategy="numeric",
    )

    # Create task
    task = FinQATask(config)

    # Sample data structure for FinQA
    sample_data = {
        "pre_text": ["Tesla Inc. Financial Performance Summary"],
        "post_text": ["All figures in millions USD"],
        "table_ori": [
            ["Year", "Revenue", "Net Income", "Gross Margin"],
            ["2021", "53,823", "5,519", "21.0%"],
            ["2022", "81,462", "12,556", "19.3%"],
        ],
        "question": "What is the revenue growth rate from 2021 to 2022?",
        "answer": "51.3%",
    }

    # Generate prompt with table formatting
    prompt = task.create_prompt(sample_data)
    logger.info(f"FinQA Prompt Preview:\n{prompt[:400]}...")

    # Example response processing
    mock_response = (
        "To calculate revenue growth: (81,462 - 53,823) / 53,823 * 100 = 51.3%"
    )
    extracted = task.extract_response(mock_response, sample_data)
    logger.info(f"Extracted Answer: {extracted}")

    return {
        "task": "finqa",
        "extraction_success": extracted is not None,
        "extracted_value": extracted,
    }


def run_finer_example():
    """Example: FiNER financial named entity recognition."""
    logger.info("=== FiNER Example: Financial Named Entity Recognition ===")

    # Configuration
    config = FiNERConfig(
        name="finer",
        huggingface_dataset="gtfintechlab/finer-ord-bio",
        dataset_split="test",
        num_samples=5,
        prompt_format="zero_shot",
        extraction_strategy="structured",
    )

    # Create task
    task = FiNERTask(config)

    # Sample data structure for FiNER (BIO tagging)
    sample_data = {
        "tokens": [
            "Apple",
            "Inc.",
            "reported",
            "$",
            "97.3",
            "billion",
            "revenue",
            "in",
            "Q1",
            "2023",
        ],
        "tags": [
            "B-ORG",
            "I-ORG",
            "O",
            "B-MONEY",
            "I-MONEY",
            "I-MONEY",
            "O",
            "O",
            "B-DATE",
            "I-DATE",
        ],
    }

    # Generate prompt
    prompt = task.create_prompt(sample_data)
    logger.info(f"FiNER Prompt Preview:\n{prompt[:300]}...")

    # Example response processing
    mock_response = "Tags: B-ORG I-ORG O B-MONEY I-MONEY I-MONEY O O B-DATE I-DATE"
    extracted = task.extract_response(mock_response, sample_data)
    logger.info(f"Extracted BIO Tags: {extracted}")

    return {
        "task": "finer",
        "extraction_success": extracted is not None,
        "extracted_value": extracted,
    }


def run_finentity_example():
    """Example: FinEntity financial entity classification."""
    logger.info("=== FinEntity Example: Financial Entity Classification ===")

    # Configuration
    config = FinEntityConfig(
        name="finentity",
        huggingface_dataset="yixuantt/FinEntity",
        dataset_split="test",
        num_samples=5,
        prompt_format="few_shot",
        extraction_strategy="keyword",
    )

    # Create task
    task = FinEntityTask(config)

    # Sample data structure for FinEntity
    sample_data = {
        "sentence": "The Federal Reserve announced a 0.25% interest rate cut.",
        "entity": "Federal Reserve",
        "label": "ORGANIZATION",
    }

    # Generate prompt
    prompt = task.create_prompt(sample_data)
    logger.info(f"FinEntity Prompt Preview:\n{prompt[:300]}...")

    # Example response processing
    mock_response = "Classification: ORGANIZATION"
    extracted = task.extract_response(mock_response, sample_data)
    logger.info(f"Extracted Entity Type: {extracted}")

    return {
        "task": "finentity",
        "extraction_success": extracted is not None,
        "extracted_value": extracted,
    }


def run_edtsum_example():
    """Example: EDTSum earnings call QA."""
    logger.info("=== EDTSum Example: Earnings Call Question Answering ===")

    # Configuration
    config = EDTSumConfig(
        name="edtsum",
        huggingface_dataset="gtfintechlab/edtsum",
        dataset_split="test",
        num_samples=5,
        prompt_format="chain_of_thought",
        extraction_strategy="extractive",
    )

    # Create task
    task = EDTSumTask(config)

    # Sample data structure for EDTSum
    sample_data = {
        "summary": "Management discussed strong Q3 performance with revenue growth of 15% year-over-year, driven by increased demand in cloud services and hardware sales.",
        "company": "TechCorp Inc.",
        "quarter": "Q3 2023",
        "question": "What factors drove the revenue growth this quarter?",
    }

    # Generate prompt
    prompt = task.create_prompt(sample_data)
    logger.info(f"EDTSum Prompt Preview:\n{prompt[:400]}...")

    # Example response processing
    mock_response = "Step 3 - Answer: The revenue growth was driven by increased demand in cloud services and hardware sales, as mentioned by management."
    extracted = task.extract_response(mock_response, sample_data)
    logger.info(f"Extracted Answer: {extracted}")

    return {
        "task": "edtsum",
        "extraction_success": extracted is not None,
        "extracted_value": extracted,
    }


def run_evaluation_example():
    """Example: Comprehensive evaluation across task types."""
    logger.info("=== Evaluation Example: FLAME Task Metrics ===")

    evaluator = FLAMEEvaluator()

    # QA evaluation example (ConvFinQA/FinQA)
    qa_predictions = ["$15.5B", "51.3%", None, "20%", "Cannot be determined"]
    qa_ground_truth = ["$15.5B", "51.3%", "10.2%", "20.0%", "12.5%"]
    qa_metrics = evaluator.evaluate_task("convfinqa", qa_predictions, qa_ground_truth)
    logger.info(f"QA Metrics: {qa_metrics}")

    # NER evaluation example (FiNER)
    ner_predictions = [
        ["B-ORG", "I-ORG", "O", "B-MONEY"],
        ["B-PER", "O", "B-DATE", "I-DATE"],
        None,  # Failed extraction
        ["O", "O", "B-ORG", "I-ORG"],
    ]
    ner_ground_truth = [
        ["B-ORG", "I-ORG", "O", "B-MONEY"],
        ["B-PER", "O", "B-DATE", "I-DATE"],
        ["B-LOC", "I-LOC", "O", "O"],
        ["O", "O", "B-ORG", "I-ORG"],
    ]
    ner_metrics = evaluator.evaluate_task("finer", ner_predictions, ner_ground_truth)
    logger.info(f"NER Metrics: {ner_metrics}")

    # Classification evaluation example (FinEntity)
    class_predictions = ["ORGANIZATION", "MONEY", None, "PERCENT", "PERSON"]
    class_ground_truth = ["ORGANIZATION", "MONEY", "DATE", "PERCENT", "PERSON"]
    class_metrics = evaluator.evaluate_task(
        "finentity", class_predictions, class_ground_truth
    )
    logger.info(f"Classification Metrics: {class_metrics}")

    return {
        "qa_metrics": qa_metrics,
        "ner_metrics": ner_metrics,
        "classification_metrics": class_metrics,
    }


def run_full_pipeline_example():
    """Example: Full BenchForge pipeline with FLAME task."""
    logger.info("=== Full Pipeline Example ===")

    # This would be a complete pipeline example
    # Note: Requires actual LLM and dataset access

    try:
        # Setup LLM configuration
        llm_config = LLMConfig(
            provider="openai",  # or your preferred provider
            model="gpt-3.5-turbo",
            max_tokens=256,
            temperature=0.0,
        )

        # Create task configuration
        task_config = ConvFinQAConfig(
            name="convfinqa_pipeline",
            num_samples=10,
            batch_size=5,
        )

        # Initialize FLAME adapter
        adapter = FLAMEAdapter()

        # Create task instance
        task = adapter.create_task("convfinqa", task_config)

        logger.info("Pipeline setup complete. In a real scenario, you would:")
        logger.info("1. Load dataset using task.load_dataset()")
        logger.info("2. Create inference engine with LLM config")
        logger.info("3. Run inference to get model responses")
        logger.info("4. Evaluate results using FLAME evaluator")
        logger.info("5. Generate comprehensive reports")

        return {"status": "setup_complete", "task_ready": True}

    except Exception as e:
        logger.warning(f"Full pipeline requires actual LLM access: {e}")
        return {"status": "demo_only", "task_ready": False}


def main():
    """Run all FLAME QA and NER migration examples."""
    logger.info("Starting FLAME QA and NER Migration Examples")

    results = {}

    # Run individual task examples
    results["convfinqa"] = run_convfinqa_example()
    results["finqa"] = run_finqa_example()
    results["finer"] = run_finer_example()
    results["finentity"] = run_finentity_example()
    results["edtsum"] = run_edtsum_example()

    # Run evaluation example
    results["evaluation"] = run_evaluation_example()

    # Run pipeline example
    results["pipeline"] = run_full_pipeline_example()

    # Summary
    logger.info("=== Migration Summary ===")
    successful_extractions = sum(
        1
        for r in results.values()
        if isinstance(r, dict) and r.get("extraction_success")
    )
    logger.info(f"Successful extractions: {successful_extractions}/5 tasks")

    logger.info("\nKey Features Demonstrated:")
    logger.info("✓ Multi-turn conversation handling (ConvFinQA)")
    logger.info("✓ Financial table processing (FinQA)")
    logger.info("✓ BIO sequence labeling (FiNER)")
    logger.info("✓ Entity classification (FinEntity)")
    logger.info("✓ Earnings call QA (EDTSum)")
    logger.info("✓ Comprehensive evaluation metrics")
    logger.info("✓ FLAME-compatible output formats")
    logger.info("✓ Robust extraction strategies")

    return results


if __name__ == "__main__":
    results = main()
