"""Example task implementation for the bench-forge.

This module demonstrates how to implement a simple task using the Bench Forge framework.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple

from bench_forge import BaseTask, TaskConfig, register_task, task_decorator


# Method 1: Using decorator functions
@task_decorator("simple_qa", mode="inference")
def simple_qa_inference(args) -> pd.DataFrame:
    """Run inference for simple QA task.

    Args:
        args: Arguments containing model configuration

    Returns:
        DataFrame with inference results
    """
    # Example data - in real task, load from dataset
    questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
    ]

    # Prepare prompts
    prompts = []
    for q in questions:
        prompt = f"Question: {q}\nAnswer:"
        prompts.append([{"role": "user", "content": prompt}])

    # Run inference using batch processing
    from bench_forge.utils.batch import process_batch_with_retry

    model_config = {
        "model": args.model,
        "max_tokens": getattr(args, "max_tokens", 50),
        "temperature": getattr(args, "temperature", 0.0),
        "top_p": getattr(args, "top_p", 0.9),
        "batch_size": getattr(args, "batch_size", 10),
    }

    # Process batches
    try:
        responses = process_batch_with_retry(
            model_config, prompts, batch_idx=0, total_batches=1
        )

        # Extract text from responses
        answers = [r.choices[0].message.content for r in responses]
    except Exception as e:
        print(f"Inference failed: {e}")
        answers = ["ERROR"] * len(questions)

    # Return as DataFrame
    return pd.DataFrame(
        {
            "question": questions,
            "model_answer": answers,
            "expected_answer": ["Paris", "4", "William Shakespeare"],
        }
    )


@task_decorator("simple_qa", mode="evaluation")
def simple_qa_evaluate(file_name: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate results for simple QA task.

    Args:
        file_name: Path to inference results
        args: Additional arguments

    Returns:
        Tuple of (results DataFrame, metrics DataFrame)
    """
    # Load predictions
    df = pd.read_csv(file_name)

    # Simple exact match evaluation
    correct = 0
    total = len(df)

    for _, row in df.iterrows():
        if row["expected_answer"].lower() in row["model_answer"].lower():
            correct += 1

    accuracy = correct / total if total > 0 else 0

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        [
            {
                "task": "simple_qa",
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
        ]
    )

    return df, metrics_df


# Method 2: Using BaseTask class
class SimpleQATask(BaseTask):
    """Example task using the BaseTask interface."""

    def __init__(self):
        """Initialize the task with configuration."""
        config = TaskConfig(
            name="simple_qa_v2",
            dataset="simple_qa_dataset",
            metrics=["accuracy", "exact_match"],
            prompt_format="zero_shot",
            batch_size=5,
            max_tokens=50,
        )
        super().__init__(config)

    def load_dataset(self, split: str = "test") -> List[Dict[str, str]]:
        """Load the dataset.

        Args:
            split: Dataset split to load

        Returns:
            List of dataset items
        """
        # In real implementation, load from HuggingFace or file
        return [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2 + 2?", "answer": "4"},
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare",
            },
            {"question": "What is the largest planet?", "answer": "Jupiter"},
            {"question": "What year did WW2 end?", "answer": "1945"},
        ]

    def prepare_prompts(
        self, dataset: List[Dict[str, str]], format: str = "zero_shot"
    ) -> List[List[Dict[str, str]]]:
        """Prepare prompts from dataset.

        Args:
            dataset: Dataset items
            format: Prompt format

        Returns:
            List of message lists for LLM
        """
        prompts = []

        if format == "zero_shot":
            for item in dataset:
                prompt = f"Question: {item['question']}\nAnswer:"
                prompts.append([{"role": "user", "content": prompt}])
        elif format == "few_shot":
            # Add few-shot examples
            examples = [
                "Question: What is the capital of UK?\nAnswer: London",
                "Question: What is 5 + 3?\nAnswer: 8",
            ]

            for item in dataset:
                prompt = (
                    "\n".join(examples) + f"\n\nQuestion: {item['question']}\nAnswer:"
                )
                prompts.append([{"role": "user", "content": prompt}])

        return prompts

    def run_inference(
        self, prompts: List[List[Dict[str, str]]], model_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Run inference on prompts.

        Args:
            prompts: Prepared prompts
            model_config: Model configuration

        Returns:
            DataFrame with results
        """
        from bench_forge.utils.batch import BatchProcessor

        # Use batch processor
        processor = BatchProcessor(
            batch_size=model_config.get("batch_size", 5), max_retries=3
        )

        # Process batches
        try:
            responses = processor.process_llm_batches(prompts, model_config)

            # Extract answers
            answers = []
            for r in responses:
                if r:
                    answers.append(r.choices[0].message.content)
                else:
                    answers.append("ERROR")
        except Exception as e:
            print(f"Inference error: {e}")
            answers = ["ERROR"] * len(prompts)

        # Load dataset to get expected answers
        dataset = self.load_dataset()

        return pd.DataFrame(
            {
                "question": [d["question"] for d in dataset],
                "model_answer": answers,
                "expected_answer": [d["answer"] for d in dataset],
            }
        )

    def evaluate_results(
        self, predictions: pd.DataFrame, ground_truth: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate predictions.

        Args:
            predictions: Model predictions
            ground_truth: Optional ground truth

        Returns:
            Tuple of (results, metrics)
        """
        # Calculate metrics
        correct = 0
        exact_match = 0
        total = len(predictions)

        for _, row in predictions.iterrows():
            expected = str(row["expected_answer"]).lower()
            predicted = str(row["model_answer"]).lower()

            # Check if answer is contained
            if expected in predicted:
                correct += 1

            # Check exact match
            if expected == predicted.strip():
                exact_match += 1

        # Create metrics
        metrics = {
            "task": self.config.name,
            "accuracy": correct / total if total > 0 else 0,
            "exact_match": exact_match / total if total > 0 else 0,
            "correct": correct,
            "exact_matches": exact_match,
            "total": total,
        }

        metrics_df = pd.DataFrame([metrics])

        return predictions, metrics_df


# Register the class-based task
if __name__ == "__main__":
    # Create and register task instance
    task = SimpleQATask()
    register_task(
        "simple_qa_v2",
        inference_fn=task.execute_inference,
        evaluation_fn=task.execute_evaluation,
        config=task.config.__dict__,
    )

    print("Tasks registered successfully!")
    print("You can now run:")
    print("  bench-forge --mode inference --tasks simple_qa --model 'your-model'")
    print("  bench-forge --mode inference --tasks simple_qa_v2 --model 'your-model'")
