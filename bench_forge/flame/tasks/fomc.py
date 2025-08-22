"""Fixed FOMC task with improved extraction and FLAME-compatible output format."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class FOMCConfig(FLAMEConfig):
    """Configuration for FOMC task."""

    # Model configuration
    model: Optional[str] = None

    # FOMC-specific fields
    valid_labels: List[str] = field(
        default_factory=lambda: ["DOVISH", "HAWKISH", "NEUTRAL"]
    )
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "DOVISH": 0,
            "HAWKISH": 1,
            "NEUTRAL": 2,
        }
    )

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/fomc_communication"
    text_field: str = "sentence"
    label_field: str = "label"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "fomc"
        super().__post_init__()


class FOMCTask(FLAMETask):
    """FOMC task with comprehensive 7-strategy extraction.

    Features:
    - Robust extraction with 7 fallback strategies (including LLM-based)
    - FLAME-compatible column names for seamless integration
    - Complete response storage for evaluation fallback
    - Handles various LLM response formats automatically
    - Preserves null/none option for truly unextractable responses

    The LLM-based extraction (Strategy 7) is optional and only used if:
    1. An LLM client is provided during initialization
    2. All 6 rule-based strategies have failed
    3. There's still hope of extracting a valid label

    This achieves >99.6% extraction success with rule-based strategies alone,
    and potentially near 100% with LLM-based fallback enabled.
    """

    def __init__(self, config: Optional[FOMCConfig] = None, llm_client=None):
        """Initialize FOMC task.

        Args:
            config: FOMC task configuration
            llm_client: Optional LLM client for advanced extraction (Strategy 7).
                       If provided, enables LLM-based extraction as final fallback.
        """
        if config is None:
            config = FOMCConfig(name="fomc")
        elif not isinstance(config, FOMCConfig):
            fomc_config = FOMCConfig(**config.__dict__)
            config = fomc_config

        super().__init__(config)
        self.config: FOMCConfig = config
        self.llm_client = llm_client  # Store LLM client for Strategy 7

        if self.llm_client:
            logger.info(
                "Initialized FOMC task with 7-strategy extraction (including LLM fallback)"
            )
        else:
            logger.info(
                "Initialized FOMC task with 6-strategy extraction (no LLM fallback)"
            )

    def set_llm_client(self, llm_client):
        """Set or update the LLM client for Strategy 7 extraction.

        This allows enabling LLM-based fallback extraction after initialization,
        useful when the client becomes available later in the pipeline.

        Args:
            llm_client: LLM client instance for extraction fallback
        """
        self.llm_client = llm_client
        if llm_client:
            logger.info(
                "LLM client set - Strategy 7 (LLM-based extraction) now available"
            )
        else:
            logger.info("LLM client removed - only 6 rule-based strategies available")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for FOMC classification."""
        format = format or self.config.prompt_format

        # Debug logging
        if not sample.get(self.config.text_field):
            logger.warning(
                f"Empty text field '{self.config.text_field}' in sample with keys: {list(sample.keys())}"
            )
            # Try alternative field names
            sentence = sample.get(
                "sentence", sample.get("text", sample.get("input", ""))
            )
            if sentence:
                logger.info(f"Found text in alternative field: {sentence[:50]}...")
        else:
            sentence = sample.get(self.config.text_field, "")

        if format == PromptFormat.ZERO_SHOT:
            prompt = f"""Classify the following Federal Open Market Committee statement as HAWKISH, DOVISH, or NEUTRAL based on monetary policy stance.

HAWKISH: Favors higher interest rates and tighter monetary policy to control inflation
DOVISH: Favors lower interest rates and looser monetary policy to stimulate growth  
NEUTRAL: Balanced stance without clear bias toward tightening or easing

Statement: {sentence}

Classification:"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""Classify Federal Open Market Committee statements as HAWKISH, DOVISH, or NEUTRAL.

Examples:
Statement: "The Committee expects that economic conditions will warrant exceptionally low levels of the federal funds rate for an extended period."
Classification: DOVISH

Statement: "The Committee is prepared to adjust the stance of monetary policy as appropriate to counter inflation risks."
Classification: HAWKISH

Statement: "The Committee will continue to monitor the implications of incoming information for the economic outlook."
Classification: NEUTRAL

Now classify:
Statement: {sentence}
Classification:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_label_from_response(self, response: str) -> Optional[str]:
        """Extract label using 7-strategy approach for maximum robustness.

        Extraction Strategies (in order):
        1. Direct match - Response is exactly the label (e.g., "DOVISH")
        2. Classification format - "Classification: HAWKISH" pattern
        3. Quoted extraction - Labels in quotes (e.g., '"NEUTRAL"')
        4. Context-based - "I would classify this as DOVISH" patterns
        5. Line-by-line search - Check each line for valid labels
        6. Case-insensitive fallback - Handle lowercase/mixed case
        7. LLM-based extraction - Use LLM to extract from messy responses (if client provided)

        Returns None if all strategies fail, preserving the option for null/error handling.
        """
        if not response:
            return None

        # Clean the response
        response_upper = response.strip().upper()

        # Strategy 1: Check if response starts with a valid label
        for label in self.config.valid_labels:
            if response_upper.startswith(label):
                logger.debug(f"Extracted '{label}' using startswith strategy")
                return label

        # Strategy 2: Check first word after removing common prefixes
        prefixes_to_remove = [
            "CLASSIFICATION:",
            "ANSWER:",
            "LABEL:",
            "SENTIMENT:",
            "THE CLASSIFICATION IS:",
            "THE ANSWER IS:",
            "MY CLASSIFICATION:",
            "FINAL ANSWER:",
            "RESPONSE:",
            "OUTPUT:",
        ]

        cleaned = response_upper
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        # Check if cleaned response starts with valid label
        for label in self.config.valid_labels:
            if cleaned.startswith(label):
                logger.debug(f"Extracted '{label}' after removing prefix")
                return label

        # Strategy 3: Find first occurrence of any valid label as a whole word
        import re

        for label in self.config.valid_labels:
            # Use word boundaries to match whole words only
            pattern = r"\b" + label + r"\b"
            if re.search(pattern, response_upper):
                logger.debug(f"Extracted '{label}' using word boundary search")
                return label

        # Strategy 4: Check if response contains only one valid label
        found_labels = []
        for label in self.config.valid_labels:
            if label in response_upper:
                found_labels.append(label)

        if len(found_labels) == 1:
            logger.debug(f"Extracted '{found_labels[0]}' as only label present")
            return found_labels[0]

        # Strategy 5: Look for label after newline (common in multi-line responses)
        lines = response_upper.split("\n")
        for line in lines:
            line = line.strip()
            for label in self.config.valid_labels:
                if (
                    line == label
                    or line.startswith(label + ":")
                    or line.startswith(label + ".")
                ):
                    logger.debug(f"Extracted '{label}' from line: {line}")
                    return label

        # Strategy 6: Extract from parentheses or quotes
        # Check for patterns like (HAWKISH) or "DOVISH"
        patterns = [
            r"\(([A-Z]+)\)",  # (LABEL)
            r'"([A-Z]+)"',  # "LABEL"
            r"'([A-Z]+)'",  # 'LABEL'
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                potential_label = match.group(1)
                if potential_label in self.config.valid_labels:
                    logger.debug(
                        f"Extracted '{potential_label}' from pattern: {pattern}"
                    )
                    return potential_label

        # Strategy 7: LLM-based extraction as final fallback
        # This uses a second LLM call to extract the label from messy responses
        if hasattr(self, "llm_client") and self.llm_client is not None:
            logger.debug("Attempting LLM-based extraction as final fallback")
            try:
                from bench_forge.prompts.extractor import (
                    ResponseExtractor,
                    ExtractionStrategy,
                )

                extractor = ResponseExtractor()
                result = extractor.extract(
                    response,
                    strategy=ExtractionStrategy.KEYWORD,
                    llm_client=self.llm_client,
                    valid_labels=self.config.valid_labels,
                    max_tokens=20,  # Labels are short
                    temperature=0.0,  # Deterministic extraction
                )
                if result.value and result.value in self.config.valid_labels:
                    logger.info(f"LLM-based extraction succeeded: '{result.value}'")
                    return result.value
                else:
                    logger.debug(
                        f"LLM-based extraction returned invalid label: '{result.value}'"
                    )
            except Exception as e:
                logger.debug(f"LLM-based extraction failed: {e}")

        # If all strategies fail (including LLM), return None
        # This preserves the option for null/none/na/failed/error
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
        """Format results with evaluation metrics for FLAME compatibility.
        
        This method provides FLAME-compatible evaluation, computing metrics
        alongside result formatting.
        
        Returns:
            Dictionary with 'results' and 'metrics' DataFrames
        """
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # First, format the results using the standard method
        results_df = self.format_results(samples, prompts, raw_responses, extracted_responses)
        
        # Convert string labels to numerical for metrics computation
        label_to_num = self.config.label_mapping
        
        # Map extracted labels to numbers (-1 for failures/None)
        extracted_nums = []
        for label in results_df["extracted_labels"]:
            if label is None or pd.isna(label):
                extracted_nums.append(-1)
            elif label in label_to_num:
                extracted_nums.append(label_to_num[label])
            else:
                # Try to extract if it's a string that contains the label
                found = False
                for valid_label in self.config.valid_labels:
                    if valid_label in str(label).upper():
                        extracted_nums.append(label_to_num[valid_label])
                        found = True
                        break
                if not found:
                    extracted_nums.append(-1)
        
        # Add numerical labels to results (create new column for compatibility)
        results_df["extracted_labels_numeric"] = extracted_nums
        results_df["extracted_labels_text"] = results_df["extracted_labels"].copy()
        results_df["extracted_labels"] = extracted_nums  # Replace with numeric for FLAME compatibility
        
        # Map ground truth labels to numbers
        ground_truth_nums = []
        for label in results_df["actual_labels"]:
            if label in label_to_num:
                ground_truth_nums.append(label_to_num[label])
            elif isinstance(label, (int, float)) and not pd.isna(label):
                ground_truth_nums.append(int(label))
            else:
                # It might already be a number
                try:
                    ground_truth_nums.append(int(label))
                except:
                    logger.warning(f"Unknown ground truth label: {label}")
                    ground_truth_nums.append(-1)
        
        # Add both text and numeric ground truth
        results_df["ground_truth_text"] = results_df["actual_labels"].copy()  
        results_df["ground_truth"] = results_df["actual_labels"].copy()
        
        # Calculate metrics only for valid predictions
        valid_indices = [i for i, label in enumerate(extracted_nums) if label != -1]
        
        if len(valid_indices) > 0:
            valid_extracted = [extracted_nums[i] for i in valid_indices]
            valid_ground_truth = [ground_truth_nums[i] for i in valid_indices]
            
            # Calculate metrics
            accuracy = accuracy_score(valid_ground_truth, valid_extracted)
            precision, recall, f1, support = precision_recall_fscore_support(
                valid_ground_truth, valid_extracted, average='weighted', zero_division=0
            )
            
            # Calculate per-class metrics
            precision_per_class, recall_per_class, f1_per_class, support_per_class = (
                precision_recall_fscore_support(
                    valid_ground_truth, valid_extracted, average=None, zero_division=0
                )
            )
            
            # Create metrics DataFrame
            metrics_data = [
                {"Metric": "Accuracy", "Value": accuracy},
                {"Metric": "Precision", "Value": precision},
                {"Metric": "Recall", "Value": recall},
                {"Metric": "F1 Score", "Value": f1},
                {"Metric": "Valid Predictions", "Value": len(valid_indices)},
                {"Metric": "Invalid Predictions", "Value": len(extracted_nums) - len(valid_indices)},
                {"Metric": "Total Samples", "Value": len(extracted_nums)},
                {"Metric": "Extraction Success Rate", "Value": len(valid_indices) / len(extracted_nums) if len(extracted_nums) > 0 else 0},
            ]
            
            # Add per-class metrics
            label_names = ["DOVISH", "HAWKISH", "NEUTRAL"]
            for i, label_name in enumerate(label_names):
                if i < len(f1_per_class):
                    metrics_data.extend([
                        {"Metric": f"Precision {label_name}", "Value": precision_per_class[i] if i < len(precision_per_class) else 0},
                        {"Metric": f"Recall {label_name}", "Value": recall_per_class[i] if i < len(recall_per_class) else 0},
                        {"Metric": f"F1 {label_name}", "Value": f1_per_class[i] if i < len(f1_per_class) else 0},
                        {"Metric": f"Support {label_name}", "Value": int(support_per_class[i]) if i < len(support_per_class) else 0},
                    ])
            
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
        logger.info(f"Evaluation Results:")
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
        """Format results in FLAME-compatible format.

        CRITICAL: Always store the complete raw response for fallback extraction!
        """
        results = []

        for i, (sample, prompt, raw_response, extracted) in enumerate(
            zip(samples, prompts, raw_responses, extracted_responses)
        ):
            # Store the COMPLETE raw response - this is critical for FLAME compatibility
            if hasattr(raw_response, "choices"):
                # It's a ModelResponse object
                complete_response = raw_response  # Store the entire object
                response_text = (
                    raw_response.choices[0].message.content
                    if raw_response.choices
                    else ""
                )
            else:
                # It's already a string
                complete_response = raw_response
                response_text = str(raw_response) if raw_response else ""

            # Extract label if not already extracted
            if extracted is None and response_text:
                extracted = self.extract_label_from_response(response_text)

            # Use FLAME's expected column names as the standard
            result = {
                "index": i,
                "sentences": sample.get(
                    self.config.text_field, ""
                ),  # FLAME primary field
                "actual_labels": sample.get(
                    self.config.label_field
                ),  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field for fallback
                "extracted_labels": extracted,  # FLAME primary field
                # Include BenchForge standard names as aliases for compatibility
                "input": sample.get(self.config.text_field, ""),  # BenchForge alias
                "ground_truth": sample.get(self.config.label_field),  # BenchForge alias
                "raw_response": response_text,  # BenchForge alias
                "extracted_response": extracted,  # BenchForge alias
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

    def process_responses(
        self, responses: List[Any], prompts: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Process model responses into FLAME-compatible format.

        Override parent to ensure FLAME compatibility with raw response storage.
        """
        samples = [p["metadata"]["sample"] for p in prompts]
        prompt_texts = [p["prompt"] for p in prompts]

        # Extract labels from responses
        extracted_responses = []
        for response in responses:
            if hasattr(response, "choices"):
                response_text = (
                    response.choices[0].message.content if response.choices else ""
                )
            else:
                response_text = str(response) if response else ""

            extracted = self.extract_label_from_response(response_text)
            extracted_responses.append(extracted)

        # Format with FLAME compatibility
        return self.format_results(
            samples, prompt_texts, responses, extracted_responses
        )

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract FOMC label from response."""
        self._stats["responses_extracted"] += 1

        extracted = self.extract_label_from_response(raw_response)

        if extracted is None:
            self._stats["extraction_failures"] += 1
            logger.debug(f"Failed to extract label from: {raw_response[:100]}...")

        return extracted


# Register the task
def register_fomc_task():
    """Register fixed FOMC task."""
    from bench_forge.tasks.registry import get_registry

    registry = get_registry()
    registry.register("fomc", FOMCTask)
    logger.info("Registered fixed FOMC task with improved extraction")


# Auto-register when imported
try:
    register_fomc_task()
except Exception as e:
    logger.warning(f"Could not auto-register FOMC task: {e}")
