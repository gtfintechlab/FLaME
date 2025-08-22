"""Causal Classification task for document-level causality classification."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import re

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class CausalClassificationConfig(FLAMEConfig):
    """Configuration for Causal Classification task."""

    # Model configuration
    model: Optional[str] = None

    # Causal Classification specific fields
    # Note: Label meanings need to be clarified from FLAME dataset
    valid_labels: List[str] = field(
        default_factory=lambda: ["0", "1", "2"]  # Numeric labels from dataset
    )
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "0": 0,  # Non-causal
            "1": 1,  # Direct causality
            "2": 2,  # Indirect/conditional causality
        }
    )

    # Alternative text labels (if clarified)
    text_label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "NON_CAUSAL": 0,
            "DIRECT_CAUSAL": 1,
            "INDIRECT_CAUSAL": 2,
        }
    )

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/CausalClassification"
    text_field: str = "text"
    label_field: str = "label"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "causal_classification"
        super().__post_init__()


class CausalClassificationTask(FLAMETask):
    """Causal Classification task for document-level causality classification.

    Features:
    - Multi-class classification for causality types
    - Robust extraction with multiple strategies
    - FLAME-compatible column names for seamless integration
    - Complete response storage for evaluation fallback
    - Handles financial document causality analysis
    """

    def __init__(
        self, config: Optional[CausalClassificationConfig] = None, llm_client=None
    ):
        """Initialize Causal Classification task.

        Args:
            config: Causal Classification task configuration
            llm_client: Optional LLM client for advanced extraction
        """
        if config is None:
            config = CausalClassificationConfig(name="causal_classification")
        elif not isinstance(config, CausalClassificationConfig):
            causal_config = CausalClassificationConfig(**config.__dict__)
            config = causal_config

        super().__init__(config)
        self.config: CausalClassificationConfig = config
        self.llm_client = llm_client

        logger.info("Initialized Causal Classification task")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for causal classification."""
        format = format or self.config.prompt_format

        # Extract text from sample
        text = sample.get(self.config.text_field, "")

        if format == PromptFormat.ZERO_SHOT:
            # Use EXACT FLAME prompt for research replication
            prompt = f"""Discard all the previous instructions. Behave like you are an expert causal classification model.
    Below is a sentence. Classify it into one of the following categories:
                    0 - Non-causal
                    1 - Direct causal
                    2 - Indirect causal
                    Only return the label number without any additional text. \n\n {text}"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""Classify the causality type in financial text using these categories:

Examples:
Text: "The company reported strong earnings which boosted share prices by 15%"
Classification: 1

Text: "Market volatility may impact our quarterly results if conditions persist"
Classification: 2

Text: "The quarterly board meeting is scheduled for next Tuesday"
Classification: 0

Text: "Rising interest rates led to decreased loan demand"
Classification: 1

Text: "Our revenue grew significantly this quarter"
Classification: 0

Now classify:
Text: {text}
Classification (respond with only the number 0, 1, or 2):"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_label_from_response(self, response: str) -> Optional[str]:
        """Extract causal classification label using robust strategies.

        Args:
            response: Raw LLM response

        Returns:
            "0", "1", "2", or None if extraction fails
        """
        if not response:
            return None

        # Clean the response
        response = response.strip()

        # Strategy 1: Look for single digit at start
        if response and response[0] in self.config.valid_labels:
            logger.debug(f"Extracted '{response[0]}' from start of response")
            return response[0]

        # Strategy 2: Remove common prefixes and check
        prefixes_to_remove = [
            "CLASSIFICATION:",
            "ANSWER:",
            "LABEL:",
            "RESULT:",
            "OUTPUT:",
            "THE CLASSIFICATION IS:",
            "THE ANSWER IS:",
            "MY CLASSIFICATION:",
            "FINAL ANSWER:",
            "RESPONSE:",
        ]

        cleaned = response.upper()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        # Check cleaned response for digit
        if cleaned and cleaned[0] in self.config.valid_labels:
            logger.debug(f"Extracted '{cleaned[0]}' after removing prefix")
            return cleaned[0]

        # Strategy 3: Search for isolated digits
        for label in self.config.valid_labels:
            # Look for the digit surrounded by word boundaries or punctuation
            patterns = [
                rf"\b{label}\b",  # Word boundary
                rf"^{label}$",  # Entire response
                rf"^{label}\.",  # Digit followed by period
                rf"^{label}:",  # Digit followed by colon
                rf"^{label}\s",  # Digit followed by space
            ]

            for pattern in patterns:
                if re.search(pattern, response):
                    logger.debug(f"Extracted '{label}' using pattern: {pattern}")
                    return label

        # Strategy 4: Look for text labels and convert
        response_upper = response.upper()
        text_mappings = [
            (["NON_CAUSAL", "NON-CAUSAL", "NO CAUSAL", "NOT CAUSAL"], "0"),
            (["DIRECT_CAUSAL", "DIRECT-CAUSAL", "DIRECT CAUSAL"], "1"),
            (["INDIRECT_CAUSAL", "INDIRECT-CAUSAL", "INDIRECT CAUSAL"], "2"),
        ]

        for text_variants, numeric_label in text_mappings:
            for variant in text_variants:
                if variant in response_upper:
                    logger.debug(
                        f"Extracted '{numeric_label}' from text variant: {variant}"
                    )
                    return numeric_label

        # Strategy 5: Line-by-line search
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0] in self.config.valid_labels:
                logger.debug(f"Extracted '{line[0]}' from line: {line}")
                return line[0]

        # Strategy 6: Extract from parentheses or quotes
        patterns = [
            r"\(([0-2])\)",  # (0), (1), (2)
            r'"([0-2])"',  # "0", "1", "2"
            r"'([0-2])'",  # '0', '1', '2'
            r"\[([0-2])\]",  # [0], [1], [2]
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                potential_label = match.group(1)
                if potential_label in self.config.valid_labels:
                    logger.debug(
                        f"Extracted '{potential_label}' from pattern: {pattern}"
                    )
                    return potential_label

        # Strategy 7: Last resort - find any digit in valid range
        for char in response:
            if char in self.config.valid_labels:
                logger.debug(
                    f"Extracted '{char}' as last resort from: {response[:50]}..."
                )
                return char

        # If all strategies fail, return None
        logger.warning(
            f"All extraction strategies failed for response: {response[:100]}..."
        )
        return None

    def format_results_with_evaluation(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> Dict[str, pd.DataFrame]:
        """Format results with multi-class classification evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
        )

        # First, format the results using the standard method
        results_df = self.format_results(
            samples, prompts, raw_responses, extracted_responses
        )

        # Convert labels to numeric for evaluation
        extracted_nums = []
        for label in results_df["extracted_labels"]:
            if label is None or pd.isna(label):
                extracted_nums.append(-1)
            elif str(label) in self.config.valid_labels:
                extracted_nums.append(int(label))
            else:
                # Try to extract digit from string
                found = False
                for valid_label in self.config.valid_labels:
                    if valid_label in str(label):
                        extracted_nums.append(int(valid_label))
                        found = True
                        break
                if not found:
                    extracted_nums.append(-1)

        # Map ground truth labels
        ground_truth_nums = []
        for label in results_df["actual_labels"]:
            if isinstance(label, (int, float)) and not pd.isna(label):
                ground_truth_nums.append(int(label))
            elif str(label) in self.config.valid_labels:
                ground_truth_nums.append(int(label))
            else:
                try:
                    num_label = int(label)
                    if 0 <= num_label <= 2:
                        ground_truth_nums.append(num_label)
                    else:
                        ground_truth_nums.append(-1)
                except:
                    ground_truth_nums.append(-1)

        # Calculate metrics only for valid predictions
        valid_indices = [
            i
            for i, (pred, gt) in enumerate(zip(extracted_nums, ground_truth_nums))
            if pred != -1 and gt != -1
        ]

        if len(valid_indices) > 0:
            valid_extracted = [extracted_nums[i] for i in valid_indices]
            valid_ground_truth = [ground_truth_nums[i] for i in valid_indices]

            # Calculate metrics
            accuracy = accuracy_score(valid_ground_truth, valid_extracted)
            precision, recall, f1, support = precision_recall_fscore_support(
                valid_ground_truth, valid_extracted, average="weighted", zero_division=0
            )

            # Per-class metrics
            try:
                (
                    precision_per_class,
                    recall_per_class,
                    f1_per_class,
                    support_per_class,
                ) = precision_recall_fscore_support(
                    valid_ground_truth, valid_extracted, average=None, zero_division=0
                )

                # Confusion matrix
                cm = confusion_matrix(
                    valid_ground_truth, valid_extracted, labels=[0, 1, 2]
                )

                metrics_data = [
                    {"Metric": "Accuracy", "Value": accuracy},
                    {"Metric": "Precision", "Value": precision},
                    {"Metric": "Recall", "Value": recall},
                    {"Metric": "F1 Score", "Value": f1},
                    {"Metric": "Valid Predictions", "Value": len(valid_extracted)},
                    {
                        "Metric": "Invalid Predictions",
                        "Value": len(extracted_nums) - len(valid_extracted),
                    },
                    {"Metric": "Total Samples", "Value": len(extracted_nums)},
                    {
                        "Metric": "Extraction Success Rate",
                        "Value": len(valid_extracted) / len(extracted_nums)
                        if len(extracted_nums) > 0
                        else 0,
                    },
                ]

                # Add per-class metrics
                class_names = ["NON_CAUSAL", "DIRECT_CAUSAL", "INDIRECT_CAUSAL"]
                for i, class_name in enumerate(class_names):
                    if i < len(f1_per_class):
                        metrics_data.extend(
                            [
                                {
                                    "Metric": f"Precision {class_name}",
                                    "Value": precision_per_class[i]
                                    if i < len(precision_per_class)
                                    else 0,
                                },
                                {
                                    "Metric": f"Recall {class_name}",
                                    "Value": recall_per_class[i]
                                    if i < len(recall_per_class)
                                    else 0,
                                },
                                {
                                    "Metric": f"F1 {class_name}",
                                    "Value": f1_per_class[i]
                                    if i < len(f1_per_class)
                                    else 0,
                                },
                                {
                                    "Metric": f"Support {class_name}",
                                    "Value": int(support_per_class[i])
                                    if i < len(support_per_class)
                                    else 0,
                                },
                            ]
                        )

                # Add confusion matrix as flattened values
                for i in range(3):
                    for j in range(3):
                        if i < cm.shape[0] and j < cm.shape[1]:
                            metrics_data.append(
                                {
                                    "Metric": f"CM[{class_names[i]} -> {class_names[j]}]",
                                    "Value": int(cm[i, j]),
                                }
                            )

            except Exception as e:
                logger.warning(f"Error calculating detailed metrics: {e}")
                metrics_data = [
                    {"Metric": "Accuracy", "Value": accuracy},
                    {"Metric": "Precision", "Value": precision},
                    {"Metric": "Recall", "Value": recall},
                    {"Metric": "F1 Score", "Value": f1},
                ]
        else:
            # No valid predictions
            metrics_data = [
                {"Metric": "Accuracy", "Value": 0.0},
                {"Metric": "Precision", "Value": 0.0},
                {"Metric": "Recall", "Value": 0.0},
                {"Metric": "F1 Score", "Value": 0.0},
                {"Metric": "Valid Predictions", "Value": 0},
                {"Metric": "Invalid Predictions", "Value": len(extracted_nums)},
                {"Metric": "Total Samples", "Value": len(extracted_nums)},
                {"Metric": "Extraction Success Rate", "Value": 0.0},
            ]

        metrics_df = pd.DataFrame(metrics_data)

        # Log key metrics
        logger.info("Multi-class Classification Evaluation Results:")
        for _, row in metrics_df.head(8).iterrows():
            if isinstance(row["Value"], float):
                logger.info(f"  {row['Metric']}: {row['Value']:.4f}")
            else:
                logger.info(f"  {row['Metric']}: {row['Value']}")

        return {"results": results_df, "metrics": metrics_df}

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

            # Extract labels if not already extracted
            if extracted is None and response_text:
                extracted = self.extract_label_from_response(response_text)

            # FLAME-compatible result format
            result = {
                "index": i,
                "texts": sample.get(self.config.text_field, ""),  # FLAME field
                "actual_labels": sample.get(self.config.label_field),  # FLAME field
                "llm_responses": response_text,  # FLAME field
                "complete_responses": complete_response,  # FLAME field
                "extracted_labels": extracted,  # FLAME field
                # BenchForge aliases
                "input": sample.get(self.config.text_field, ""),
                "ground_truth": sample.get(self.config.label_field),
                "raw_response": response_text,
                "extracted_response": extracted,
                # Metadata
                "prompt": prompt,
                "sample": sample,
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Log extraction statistics
        total = len(df)
        successful = df["extracted_labels"].notna().sum()
        success_rate = (successful / total * 100) if total > 0 else 0
        logger.info(
            f"Extraction success rate: {successful}/{total} ({success_rate:.1f}%)"
        )

        return df

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract causal classification label from response."""
        self._stats["responses_extracted"] += 1

        extracted = self.extract_label_from_response(raw_response)

        if extracted is None:
            self._stats["extraction_failures"] += 1
            logger.debug(f"Failed to extract label from: {raw_response[:100]}...")

        return extracted


# Register the task
def register_causal_classification_task():
    """Register Causal Classification task."""
    from bench_forge.tasks.registry import get_registry

    registry = get_registry()
    registry.register("causal_classification", CausalClassificationTask)
    logger.info("Registered Causal Classification task")


# Auto-register when imported
try:
    register_causal_classification_task()
except Exception as e:
    logger.warning(f"Could not auto-register Causal Classification task: {e}")
