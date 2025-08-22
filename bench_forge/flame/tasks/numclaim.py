"""Numerical Claim Classification task for binary classification of numerical claims."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import re

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class NumClaimConfig(FLAMEConfig):
    """Configuration for Numerical Claim Classification task."""

    # Model configuration
    model: Optional[str] = None

    # NumClaim specific fields
    valid_labels: List[str] = field(
        default_factory=lambda: ["OUTOFCLAIM", "INCLAIM"]
    )
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "OUTOFCLAIM": 0,
            "INCLAIM": 1,
        }
    )

    # Dataset configuration  
    huggingface_dataset: str = "gtfintechlab/Numclaim"
    text_field: str = "context"
    label_field: str = "response"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "numclaim"
        super().__post_init__()


class NumClaimTask(FLAMETask):
    """Numerical Claim Classification task for identifying numerical claims in financial text.

    Features:
    - Binary classification for numerical claims
    - Robust extraction with multiple strategies
    - FLAME-compatible column names for seamless integration
    - Complete response storage for evaluation fallback
    - Handles financial numerical claim detection
    """

    def __init__(self, config: Optional[NumClaimConfig] = None, llm_client=None):
        """Initialize Numerical Claim Classification task.

        Args:
            config: NumClaim task configuration
            llm_client: Optional LLM client for advanced extraction
        """
        if config is None:
            config = NumClaimConfig(name="numclaim")
        elif not isinstance(config, NumClaimConfig):
            numclaim_config = NumClaimConfig(**config.__dict__)
            config = numclaim_config

        super().__init__(config)
        self.config: NumClaimConfig = config
        self.llm_client = llm_client

        logger.info("Initialized Numerical Claim Classification task")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for numerical claim classification."""
        format = format or self.config.prompt_format

        # Extract text from sample
        text = sample.get(self.config.text_field, "")

        if format == PromptFormat.ZERO_SHOT:
            # Use EXACT FLAME prompt for research replication
            prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence senti-
            ment classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class.
            Label 'INCLAIM' if consist of a claim and not just factual past or present information, or
            'OUTOFCLAIM' if it has just factual past or present information. Provide the label in the
            first line and provide a short explanation in the second line. The sentence:{text}"""

        elif format == PromptFormat.FEW_SHOT:
            # FLAME doesn't have few-shot prompt for NumClaim
            return None

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_label_from_response(self, response: str) -> Optional[str]:
        """Extract numerical claim label using robust strategies.

        Args:
            response: Raw LLM response

        Returns:
            "INCLAIM", "OUTOFCLAIM", or None if extraction fails
        """
        if not response:
            return None

        # Clean the response
        response_upper = response.strip().upper()

        # Strategy 1: Direct label match
        for label in self.config.valid_labels:
            if response_upper.startswith(label):
                logger.debug(f"Extracted '{label}' using direct match")
                return label

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

        cleaned = response_upper
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        # Check cleaned response
        for label in self.config.valid_labels:
            if cleaned.startswith(label):
                logger.debug(f"Extracted '{label}' after removing prefix")
                return label

        # Strategy 3: Word boundary search
        for label in self.config.valid_labels:
            pattern = r"\b" + label + r"\b"
            if re.search(pattern, response_upper):
                logger.debug(f"Extracted '{label}' using word boundary search")
                return label

        # Strategy 4: Check for alternative phrasings
        if any(word in response_upper for word in ["IN CLAIM", "IN-CLAIM", "CONTAINS CLAIM", "HAS CLAIM", 
                                                    "CONTAINS A CLAIM", "HAS A CLAIM", "IS A CLAIM",
                                                    "CONSIST OF A CLAIM", "THIS IS INCLAIM"]):
            logger.debug("Extracted 'INCLAIM' from alternative phrasing")
            return "INCLAIM"
        
        if any(word in response_upper for word in ["OUT OF CLAIM", "OUT-OF-CLAIM", "NO CLAIM", "WITHOUT CLAIM",
                                                    "NOT A CLAIM", "JUST FACTUAL", "ONLY FACTUAL",
                                                    "THIS IS OUTOFCLAIM"]):
            logger.debug("Extracted 'OUTOFCLAIM' from alternative phrasing")
            return "OUTOFCLAIM"

        # Strategy 5: Binary keywords
        if any(word in response_upper for word in ["YES", "TRUE", "POSITIVE", "1"]):
            # Assume this means it contains a claim
            logger.debug("Extracted 'INCLAIM' from positive binary keyword")
            return "INCLAIM"
        
        if any(word in response_upper for word in ["NO", "FALSE", "NEGATIVE", "0"]):
            # Assume this means it doesn't contain a claim
            logger.debug("Extracted 'OUTOFCLAIM' from negative binary keyword")
            return "OUTOFCLAIM"

        # Strategy 6: Line-by-line search
        lines = response_upper.split("\n")
        for line in lines:
            line = line.strip()
            for label in self.config.valid_labels:
                if line == label or line.startswith(label + ":") or line.startswith(label + "."):
                    logger.debug(f"Extracted '{label}' from line: {line}")
                    return label

        # Strategy 7: Pattern extraction from quotes/parentheses
        patterns = [
            r"\(([A-Z]+)\)",  # (LABEL)
            r'"([A-Z]+)"',    # "LABEL"
            r"'([A-Z]+)'",    # 'LABEL'
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                potential_label = match.group(1)
                if potential_label in self.config.valid_labels:
                    logger.debug(f"Extracted '{potential_label}' from pattern: {pattern}")
                    return potential_label

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
        """Format results with binary classification evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
        import numpy as np
        
        # First, format the results using the standard method
        results_df = self.format_results(samples, prompts, raw_responses, extracted_responses)
        
        # Convert labels to numeric for evaluation
        label_to_num = self.config.label_mapping
        
        # Map extracted labels to numbers (-1 for failures/None)
        extracted_nums = []
        for label in results_df["extracted_labels"]:
            if label is None or pd.isna(label):
                extracted_nums.append(-1)
            elif label in label_to_num:
                extracted_nums.append(label_to_num[label])
            else:
                # Try partial matching
                found = False
                for valid_label in self.config.valid_labels:
                    if valid_label in str(label).upper():
                        extracted_nums.append(label_to_num[valid_label])
                        found = True
                        break
                if not found:
                    extracted_nums.append(-1)
        
        # Map ground truth labels
        ground_truth_nums = []
        for label in results_df["actual_labels"]:
            if isinstance(label, str) and label.upper() in label_to_num:
                ground_truth_nums.append(label_to_num[label.upper()])
            elif isinstance(label, (int, float)) and not pd.isna(label):
                ground_truth_nums.append(int(label))
            else:
                # Try to map from text
                try:
                    if str(label).upper() in label_to_num:
                        ground_truth_nums.append(label_to_num[str(label).upper()])
                    else:
                        ground_truth_nums.append(-1)
                except:
                    ground_truth_nums.append(-1)
        
        # Calculate metrics only for valid predictions
        valid_indices = [i for i, label in enumerate(extracted_nums) if label != -1]
        
        if len(valid_indices) > 0:
            valid_extracted = [extracted_nums[i] for i in valid_indices]
            valid_ground_truth = [ground_truth_nums[i] for i in valid_indices]
            
            # Remove any remaining invalid ground truth
            final_valid_indices = [i for i, gt in enumerate(valid_ground_truth) if gt != -1]
            if final_valid_indices:
                valid_extracted = [valid_extracted[i] for i in final_valid_indices]
                valid_ground_truth = [valid_ground_truth[i] for i in final_valid_indices]
            
            if valid_extracted and valid_ground_truth:
                # Calculate metrics
                accuracy = accuracy_score(valid_ground_truth, valid_extracted)
                precision, recall, f1, support = precision_recall_fscore_support(
                    valid_ground_truth, valid_extracted, average='weighted', zero_division=0
                )
                
                # Binary classification specific metrics
                try:
                    # Per-class metrics
                    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
                        precision_recall_fscore_support(
                            valid_ground_truth, valid_extracted, average=None, zero_division=0
                        )
                    )
                    
                    # Confusion matrix
                    cm = confusion_matrix(valid_ground_truth, valid_extracted)
                    
                    metrics_data = [
                        {"Metric": "Accuracy", "Value": accuracy},
                        {"Metric": "Precision", "Value": precision},
                        {"Metric": "Recall", "Value": recall},
                        {"Metric": "F1 Score", "Value": f1},
                        {"Metric": "Valid Predictions", "Value": len(valid_extracted)},
                        {"Metric": "Invalid Predictions", "Value": len(extracted_nums) - len(valid_extracted)},
                        {"Metric": "Total Samples", "Value": len(extracted_nums)},
                        {"Metric": "Extraction Success Rate", "Value": len(valid_extracted) / len(extracted_nums) if len(extracted_nums) > 0 else 0},
                    ]
                    
                    # Add per-class metrics
                    label_names = ["OUTOFCLAIM", "INCLAIM"]
                    for i, label_name in enumerate(label_names):
                        if i < len(f1_per_class):
                            metrics_data.extend([
                                {"Metric": f"Precision {label_name}", "Value": precision_per_class[i] if i < len(precision_per_class) else 0},
                                {"Metric": f"Recall {label_name}", "Value": recall_per_class[i] if i < len(recall_per_class) else 0},
                                {"Metric": f"F1 {label_name}", "Value": f1_per_class[i] if i < len(f1_per_class) else 0},
                                {"Metric": f"Support {label_name}", "Value": int(support_per_class[i]) if i < len(support_per_class) else 0},
                            ])
                    
                    # Add confusion matrix values
                    if cm.shape == (2, 2):
                        metrics_data.extend([
                            {"Metric": "True Negatives", "Value": int(cm[0, 0])},
                            {"Metric": "False Positives", "Value": int(cm[0, 1])},
                            {"Metric": "False Negatives", "Value": int(cm[1, 0])},
                            {"Metric": "True Positives", "Value": int(cm[1, 1])},
                        ])
                        
                except Exception as e:
                    logger.warning(f"Error calculating detailed metrics: {e}")
                    metrics_data = [
                        {"Metric": "Accuracy", "Value": accuracy},
                        {"Metric": "Precision", "Value": precision},
                        {"Metric": "Recall", "Value": recall},
                        {"Metric": "F1 Score", "Value": f1},
                    ]
            else:
                # No valid comparisons possible
                metrics_data = [
                    {"Metric": "Accuracy", "Value": 0.0},
                    {"Metric": "Error", "Value": "No valid ground truth labels found"},
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
        logger.info("Binary Classification Evaluation Results:")
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

            # Extract labels if not already extracted
            if extracted is None and response_text:
                extracted = self.extract_label_from_response(response_text)

            # FLAME-compatible result format
            result = {
                "index": i,
                "sentences": sample.get(self.config.text_field, ""),  # FLAME primary field
                "actual_labels": sample.get(self.config.label_field),  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # FLAME primary field
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
        """Extract numerical claim label from response."""
        self._stats["responses_extracted"] += 1

        extracted = self.extract_label_from_response(raw_response)

        if extracted is None:
            self._stats["extraction_failures"] += 1
            logger.debug(f"Failed to extract label from: {raw_response[:100]}...")

        return extracted


# Register the task
def register_numclaim_task():
    """Register Numerical Claim Classification task."""
    from bench_forge.tasks.registry import get_registry

    registry = get_registry()
    registry.register("numclaim", NumClaimTask)
    logger.info("Registered Numerical Claim Classification task")


# Auto-register when imported
try:
    register_numclaim_task()
except Exception as e:
    logger.warning(f"Could not auto-register NumClaim task: {e}")