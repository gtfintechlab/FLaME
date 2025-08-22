"""FiQA-SA (Financial Question Answering - Sentiment Analysis) task implementation."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import re
import json

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class FiQASAConfig(FLAMEConfig):
    """Configuration for FiQA-SA task."""

    # Model configuration
    model: Optional[str] = None

    # FiQA-SA specific fields
    valid_range: tuple = (-1.0, 1.0)  # Sentiment score range
    
    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/FiQA_Task1"
    text_field: str = "sentence"
    target_field: str = "target"
    sentiment_field: str = "sentiment_score"
    snippets_field: str = "snippets"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "fiqa_sa"
        super().__post_init__()


class FiQASATask(FLAMETask):
    """FiQA-SA task for target-specific sentiment scoring.

    Features:
    - Target-specific sentiment analysis
    - Continuous sentiment scoring from -1.0 to 1.0
    - Financial text with target aspects
    - Regression metrics (MSE, MAE, Pearson correlation)
    """

    def __init__(self, config: Optional[FiQASAConfig] = None, llm_client=None):
        """Initialize FiQA-SA task.

        Args:
            config: FiQA-SA task configuration
            llm_client: Optional LLM client for advanced extraction
        """
        if config is None:
            config = FiQASAConfig(name="fiqa_sa")
        elif not isinstance(config, FiQASAConfig):
            fiqa_config = FiQASAConfig(**config.__dict__)
            config = fiqa_config

        super().__init__(config)
        self.config: FiQASAConfig = config
        self.llm_client = llm_client

        logger.info("Initialized FiQA-SA task")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for FiQA-SA sentiment scoring."""
        format = format or self.config.prompt_format

        # Extract fields from sample
        sentence = sample.get(self.config.text_field, "")
        target = sample.get(self.config.target_field, "")
        snippets = sample.get(self.config.snippets_field, "")

        if format == PromptFormat.ZERO_SHOT:
            # Use EXACT FLAME prompt for research replication
            prompt = f"""You are a financial sentiment analysis expert. Analyze the provided sentence, identify relevant target aspects (such as companies, products, or strategies), and assign a sentiment score for each target.
                The sentiment score should be between -1 (highly negative) and 1 (highly positive), using up to three decimal places to capture nuances in sentiment.

                Financial sentence:
                Sentence: {sentence}. Snippets: {snippets}. Target aspect: {target}"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""You are a financial sentiment analysis expert. Analyze sentences and assign sentiment scores.

Examples:
Sentence: "Apple's revenue exceeded expectations, driving stock price up 5%"
Target: Apple
Sentiment Score: 0.750

Sentence: "Tesla faces production challenges that may impact Q4 deliveries"
Target: Tesla
Sentiment Score: -0.425

Sentence: "Amazon reported steady growth in cloud services"
Target: Amazon
Sentiment Score: 0.325

Now analyze:
Sentence: {sentence}
Snippets: {snippets}
Target aspect: {target}

Provide a sentiment score between -1 and 1 (up to 3 decimal places):"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_score_from_response(self, response: str) -> Optional[float]:
        """Extract sentiment score from response using robust strategies.

        Args:
            response: Raw LLM response

        Returns:
            Float score between -1.0 and 1.0, or None if extraction fails
        """
        if not response:
            return None

        # Clean the response
        response = response.strip()

        # Strategy 1: Look for explicit sentiment score patterns
        patterns = [
            r"sentiment\s*score[:\s]*(-?\d*\.?\d+)",
            r"score[:\s]*(-?\d*\.?\d+)",
            r"(-?\d*\.?\d+)\s*\(sentiment",
            r"sentiment[:\s]*(-?\d*\.?\d+)",
            r"rating[:\s]*(-?\d*\.?\d+)",
            r"^(-?\d*\.?\d+)$",  # Just a number
            r":\s*(-?\d*\.?\d+)",  # After colon
            r"(-?\d*\.?\d+)\s*out of",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    # Validate range
                    if -1.0 <= score <= 1.0:
                        logger.debug(f"Extracted score {score} using pattern: {pattern}")
                        return score
                    # Handle percentage (0-100) conversion
                    elif 0 <= score <= 100:
                        normalized = (score / 100) * 2 - 1  # Convert to -1 to 1
                        logger.debug(f"Converted percentage {score} to {normalized}")
                        return normalized
                except ValueError:
                    continue

        # Strategy 2: Extract all numbers and find the most likely score
        numbers = re.findall(r"-?\d*\.?\d+", response)
        for num_str in numbers:
            try:
                num = float(num_str)
                if -1.0 <= num <= 1.0:
                    logger.debug(f"Extracted score {num} from numbers in response")
                    return num
            except ValueError:
                continue

        # Strategy 3: Look for sentiment words and map to scores
        response_lower = response.lower()
        sentiment_mappings = [
            (["highly positive", "very positive", "strongly positive"], 0.8),
            (["positive", "bullish", "favorable"], 0.5),
            (["slightly positive", "somewhat positive", "mildly positive"], 0.25),
            (["neutral", "mixed", "balanced"], 0.0),
            (["slightly negative", "somewhat negative", "mildly negative"], -0.25),
            (["negative", "bearish", "unfavorable"], -0.5),
            (["highly negative", "very negative", "strongly negative"], -0.8),
        ]

        for phrases, score in sentiment_mappings:
            for phrase in phrases:
                if phrase in response_lower:
                    logger.debug(f"Mapped sentiment phrase '{phrase}' to score {score}")
                    return score

        # If all strategies fail, return None
        logger.warning(f"All extraction strategies failed for response: {response[:100]}...")
        return None

    def format_results_with_evaluation(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> Dict[str, pd.DataFrame]:
        """Format results with regression evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr
        import numpy as np
        
        # First, format the results using the standard method
        results_df = self.format_results(samples, prompts, raw_responses, extracted_responses)
        
        # Extract numeric scores
        extracted_scores = []
        ground_truth_scores = []
        
        for idx, row in results_df.iterrows():
            extracted = row["extracted_scores"]
            actual = row["actual_scores"]
            
            # Convert to float if possible
            try:
                if extracted is not None and not pd.isna(extracted):
                    extracted_scores.append(float(extracted))
                else:
                    extracted_scores.append(None)
            except:
                extracted_scores.append(None)
            
            try:
                if actual is not None and not pd.isna(actual):
                    ground_truth_scores.append(float(actual))
                else:
                    ground_truth_scores.append(None)
            except:
                ground_truth_scores.append(None)
        
        # Calculate metrics only for valid predictions
        valid_indices = [
            i for i, (pred, gt) in enumerate(zip(extracted_scores, ground_truth_scores))
            if pred is not None and gt is not None
        ]
        
        if len(valid_indices) > 0:
            valid_extracted = [extracted_scores[i] for i in valid_indices]
            valid_ground_truth = [ground_truth_scores[i] for i in valid_indices]
            
            # Calculate regression metrics
            mse = mean_squared_error(valid_ground_truth, valid_extracted)
            mae = mean_absolute_error(valid_ground_truth, valid_extracted)
            
            # Pearson correlation
            if len(valid_extracted) > 1:
                correlation, p_value = pearsonr(valid_ground_truth, valid_extracted)
            else:
                correlation, p_value = 0.0, 1.0
            
            metrics_data = [
                {"Metric": "Mean Squared Error (MSE)", "Value": mse},
                {"Metric": "Mean Absolute Error (MAE)", "Value": mae},
                {"Metric": "Pearson Correlation", "Value": correlation},
                {"Metric": "P-value", "Value": p_value},
                {"Metric": "Valid Predictions", "Value": len(valid_extracted)},
                {"Metric": "Invalid Predictions", "Value": len(extracted_scores) - len(valid_extracted)},
                {"Metric": "Total Samples", "Value": len(extracted_scores)},
                {"Metric": "Extraction Success Rate", "Value": len(valid_extracted) / len(extracted_scores) if len(extracted_scores) > 0 else 0},
            ]
            
            # Add distribution statistics
            if valid_extracted:
                metrics_data.extend([
                    {"Metric": "Mean Predicted Score", "Value": np.mean(valid_extracted)},
                    {"Metric": "Std Predicted Score", "Value": np.std(valid_extracted)},
                    {"Metric": "Mean Actual Score", "Value": np.mean(valid_ground_truth)},
                    {"Metric": "Std Actual Score", "Value": np.std(valid_ground_truth)},
                ])
                
        else:
            # No valid predictions
            metrics_data = [
                {"Metric": "Mean Squared Error (MSE)", "Value": float('inf')},
                {"Metric": "Mean Absolute Error (MAE)", "Value": float('inf')},
                {"Metric": "Pearson Correlation", "Value": 0.0},
                {"Metric": "P-value", "Value": 1.0},
                {"Metric": "Valid Predictions", "Value": 0},
                {"Metric": "Invalid Predictions", "Value": len(extracted_scores)},
                {"Metric": "Total Samples", "Value": len(extracted_scores)},
                {"Metric": "Extraction Success Rate", "Value": 0.0},
            ]
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Log key metrics
        logger.info("FiQA-SA Regression Evaluation Results:")
        for _, row in metrics_df.head(8).iterrows():
            if isinstance(row["Value"], float):
                logger.info(f"  {row['Metric']}: {row['Value']:.4f}")
            else:
                logger.info(f"  {row['Metric']}: {row['Value']}")
        
        return {
            "results": results_df,
            "metrics": metrics_df
        }

    def format_results(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results in FLAME-compatible format."""
        results = []

        for i, (sample, prompt, raw_response, extracted) in enumerate(
            zip(samples, prompts, raw_responses, extracted_responses)
        ):
            # Store the complete raw response
            if hasattr(raw_response, "choices"):
                complete_response = raw_response
                response_text = (
                    raw_response.choices[0].message.content
                    if raw_response.choices
                    else ""
                )
            else:
                complete_response = raw_response
                response_text = str(raw_response) if raw_response else ""

            # Extract score if not already extracted
            if extracted is None and response_text:
                extracted = self.extract_score_from_response(response_text)

            # FLAME-compatible result format
            result = {
                "index": i,
                "sentences": sample.get(self.config.text_field, ""),  # FLAME field
                "targets": sample.get(self.config.target_field, ""),  # FLAME field
                "actual_scores": sample.get(self.config.sentiment_field),  # FLAME field
                "llm_responses": response_text,  # FLAME field
                "complete_responses": complete_response,  # FLAME field
                "extracted_scores": extracted,  # FLAME field
                # BenchForge aliases
                "input": f"Sentence: {sample.get(self.config.text_field, '')}. Target: {sample.get(self.config.target_field, '')}",
                "ground_truth": sample.get(self.config.sentiment_field),
                "raw_response": response_text,
                "extracted_response": extracted,
                # Metadata
                "prompt": prompt,
                "sample": sample,
                "snippets": sample.get(self.config.snippets_field, ""),
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Log extraction statistics
        total = len(df)
        successful = df["extracted_scores"].notna().sum()
        success_rate = (successful / total * 100) if total > 0 else 0
        logger.info(
            f"Extraction success rate: {successful}/{total} ({success_rate:.1f}%)"
        )

        return df

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract sentiment score from response."""
        self._stats["responses_extracted"] += 1

        extracted = self.extract_score_from_response(raw_response)

        if extracted is None:
            self._stats["extraction_failures"] += 1
            logger.debug(f"Failed to extract score from: {raw_response[:100]}...")

        return extracted


# Register the task
def register_fiqa_sa_task():
    """Register FiQA-SA task."""
    from bench_forge.tasks.registry import get_registry

    registry = get_registry()
    registry.register("fiqa_sa", FiQASATask)
    logger.info("Registered FiQA-SA task")


# Auto-register when imported
try:
    register_fiqa_sa_task()
except Exception as e:
    logger.warning(f"Could not auto-register FiQA-SA task: {e}")