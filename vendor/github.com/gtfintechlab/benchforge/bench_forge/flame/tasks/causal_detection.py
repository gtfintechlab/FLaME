"""Causal Detection task with BIO sequence labeling for cause-effect relationships."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import re

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class CausalDetectionConfig(FLAMEConfig):
    """Configuration for Causal Detection task."""

    # Model configuration
    model: Optional[str] = None

    # Causal Detection specific fields
    valid_labels: List[str] = field(
        default_factory=lambda: ["B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT", "O"]
    )
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "O": 0,
            "B-CAUSE": 1,
            "I-CAUSE": 2,
            "B-EFFECT": 3,
            "I-EFFECT": 4,
        }
    )

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/CausalDetection"
    text_field: str = "tokens"
    label_field: str = "tags"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "causal_detection"
        super().__post_init__()


class CausalDetectionTask(FLAMETask):
    """Causal Detection task for identifying cause-effect relationships at token level.

    Features:
    - BIO tagging for cause and effect entities
    - Token-level sequence labeling
    - Robust extraction with multiple strategies
    - FLAME-compatible column names for seamless integration
    - Complete response storage for evaluation fallback
    """

    def __init__(self, config: Optional[CausalDetectionConfig] = None, llm_client=None):
        """Initialize Causal Detection task.

        Args:
            config: Causal Detection task configuration
            llm_client: Optional LLM client for advanced extraction
        """
        if config is None:
            config = CausalDetectionConfig(name="causal_detection")
        elif not isinstance(config, CausalDetectionConfig):
            causal_config = CausalDetectionConfig(**config.__dict__)
            config = causal_config

        super().__init__(config)
        self.config: CausalDetectionConfig = config
        self.llm_client = llm_client

        logger.info("Initialized Causal Detection task with BIO sequence labeling")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for causal detection sequence labeling."""
        format = format or self.config.prompt_format

        # Extract tokens from sample
        tokens = sample.get(self.config.text_field, [])
        if isinstance(tokens, str):
            # If tokens is a string, split it
            tokens = tokens.split()
        
        # Join tokens into a sentence
        sentence = " ".join(tokens)

        if format == PromptFormat.ZERO_SHOT:
            prompt = f"""Label each token in the following financial text with BIO tags for cause-effect relationships.

Labels:
- B-CAUSE: Beginning of a cause phrase
- I-CAUSE: Inside/continuation of a cause phrase  
- B-EFFECT: Beginning of an effect phrase
- I-EFFECT: Inside/continuation of an effect phrase
- O: Outside any cause-effect phrase

Text: {sentence}

Provide one label per token, separated by spaces. The number of labels must exactly match the number of tokens.

Labels:"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""Label financial text tokens with BIO tags for cause-effect relationships.

Example 1:
Text: The company reported strong earnings which boosted share prices
Labels: B-CAUSE I-CAUSE I-CAUSE I-CAUSE O B-EFFECT I-EFFECT I-EFFECT

Example 2:
Text: Due to increased demand the factory expanded production capacity
Labels: B-CAUSE I-CAUSE I-CAUSE O B-EFFECT I-EFFECT I-EFFECT I-EFFECT

Now label this text:
Text: {sentence}
Labels:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_label_from_response(self, response: str, tokens: List[str] = None) -> Optional[List[str]]:
        """Extract BIO labels from response with robust parsing.

        Args:
            response: Raw LLM response
            tokens: Original tokens for length validation

        Returns:
            List of BIO labels or None if extraction fails
        """
        if not response:
            return None

        # Clean the response
        response = response.strip()

        # Strategy 1: Direct label extraction
        # Look for labels after "Labels:" or similar patterns
        patterns = [
            r"Labels?:\s*(.+)",
            r"Output:\s*(.+)",
            r"Answer:\s*(.+)",
            r"Result:\s*(.+)",
        ]

        extracted_text = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted_text = match.group(1).strip()
                break

        # Strategy 2: If no pattern found, assume the entire response is labels
        if extracted_text is None:
            # Look for the last line that contains labels
            lines = response.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and any(label in line for label in self.config.valid_labels):
                    extracted_text = line
                    break
            
            # If still nothing, use the entire response
            if extracted_text is None:
                extracted_text = response

        # Parse labels from extracted text
        labels = extracted_text.split()
        
        # Strategy 3: Clean and validate labels
        valid_labels = []
        for label in labels:
            # Remove punctuation and clean
            clean_label = re.sub(r'[^\w-]', '', label.upper())
            
            # Check if it's a valid label
            if clean_label in self.config.valid_labels:
                valid_labels.append(clean_label)
            elif clean_label.startswith(('B-', 'I-')) and any(clean_label.endswith(suffix) for suffix in ['CAUSE', 'EFFECT']):
                # Handle slight variations
                if 'CAUSE' in clean_label:
                    if clean_label.startswith('B-'):
                        valid_labels.append('B-CAUSE')
                    else:
                        valid_labels.append('I-CAUSE')
                elif 'EFFECT' in clean_label:
                    if clean_label.startswith('B-'):
                        valid_labels.append('B-EFFECT')
                    else:
                        valid_labels.append('I-EFFECT')
            elif clean_label == 'O':
                valid_labels.append('O')

        # Strategy 4: Length validation and padding/truncation
        if tokens and len(tokens) > 0:
            target_length = len(tokens)
            
            if len(valid_labels) < target_length:
                # Pad with 'O' labels
                valid_labels.extend(['O'] * (target_length - len(valid_labels)))
                logger.debug(f"Padded labels from {len(valid_labels) - (target_length - len(valid_labels))} to {target_length}")
            elif len(valid_labels) > target_length:
                # Truncate to match token length
                valid_labels = valid_labels[:target_length]
                logger.debug(f"Truncated labels from {len(valid_labels) + (len(valid_labels) - target_length)} to {target_length}")

        # Return None if we have no valid labels
        if not valid_labels:
            logger.warning(f"No valid labels extracted from response: {response[:100]}...")
            return None

        logger.debug(f"Extracted {len(valid_labels)} labels: {valid_labels[:10]}...")
        return valid_labels

    def format_results_with_evaluation(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> Dict[str, pd.DataFrame]:
        """Format results with sequence labeling evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        import numpy as np
        
        # First, format the results using the standard method
        results_df = self.format_results(samples, prompts, raw_responses, extracted_responses)
        
        # Prepare data for evaluation
        all_true_labels = []
        all_pred_labels = []
        valid_sequences = 0
        
        for i, row in results_df.iterrows():
            tokens = row['tokens']
            true_tags = row['actual_tags']
            pred_tags = row['extracted_labels']
            
            if pred_tags is not None and true_tags is not None:
                # Ensure same length
                min_len = min(len(tokens), len(true_tags), len(pred_tags))
                if min_len > 0:
                    all_true_labels.extend(true_tags[:min_len])
                    all_pred_labels.extend(pred_tags[:min_len])
                    valid_sequences += 1
        
        # Calculate metrics
        if all_true_labels and all_pred_labels:
            # Token-level accuracy
            token_accuracy = accuracy_score(all_true_labels, all_pred_labels)
            
            # Per-label metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average='weighted', zero_division=0
            )
            
            # Create detailed classification report
            try:
                report = classification_report(
                    all_true_labels, all_pred_labels, 
                    target_names=self.config.valid_labels,
                    output_dict=True,
                    zero_division=0
                )
                
                # Build metrics data
                metrics_data = [
                    {"Metric": "Token-Level Accuracy", "Value": token_accuracy},
                    {"Metric": "Weighted Precision", "Value": precision},
                    {"Metric": "Weighted Recall", "Value": recall},
                    {"Metric": "Weighted F1 Score", "Value": f1},
                    {"Metric": "Valid Sequences", "Value": valid_sequences},
                    {"Metric": "Total Sequences", "Value": len(results_df)},
                    {"Metric": "Sequence Success Rate", "Value": valid_sequences / len(results_df) if len(results_df) > 0 else 0},
                    {"Metric": "Total Tokens", "Value": len(all_true_labels)},
                ]
                
                # Add per-label metrics
                for label in self.config.valid_labels:
                    if label in report:
                        metrics_data.extend([
                            {"Metric": f"Precision {label}", "Value": report[label]['precision']},
                            {"Metric": f"Recall {label}", "Value": report[label]['recall']},
                            {"Metric": f"F1 {label}", "Value": report[label]['f1-score']},
                            {"Metric": f"Support {label}", "Value": int(report[label]['support'])},
                        ])
                        
            except Exception as e:
                logger.warning(f"Error creating classification report: {e}")
                metrics_data = [
                    {"Metric": "Token-Level Accuracy", "Value": token_accuracy},
                    {"Metric": "Weighted Precision", "Value": precision},
                    {"Metric": "Weighted Recall", "Value": recall},
                    {"Metric": "Weighted F1 Score", "Value": f1},
                ]
        else:
            # No valid predictions
            metrics_data = [
                {"Metric": "Token-Level Accuracy", "Value": 0.0},
                {"Metric": "Weighted Precision", "Value": 0.0},
                {"Metric": "Weighted Recall", "Value": 0.0},
                {"Metric": "Weighted F1 Score", "Value": 0.0},
                {"Metric": "Valid Sequences", "Value": 0},
                {"Metric": "Total Sequences", "Value": len(results_df)},
                {"Metric": "Sequence Success Rate", "Value": 0.0},
            ]
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Log key metrics
        logger.info("Sequence Labeling Evaluation Results:")
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
        """Format results in FLAME-compatible format for sequence labeling."""
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

            # Extract tokens and tags
            tokens = sample.get(self.config.text_field, [])
            actual_tags = sample.get(self.config.label_field, [])

            # Extract labels if not already extracted
            if extracted is None and response_text:
                extracted = self.extract_label_from_response(response_text, tokens)

            # FLAME-compatible result format
            result = {
                "index": i,
                "tokens": tokens,  # FLAME field for sequence labeling
                "actual_tags": actual_tags,  # FLAME field for sequence labeling
                "predicted_tags": extracted,  # FLAME field for sequence labeling
                "llm_responses": response_text,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # BenchForge alias
                # Metadata
                "prompt": prompt,
                "sample": sample,
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Log extraction statistics
        total = len(df)
        successful = df["predicted_tags"].notna().sum()
        success_rate = (successful / total * 100) if total > 0 else 0
        logger.info(
            f"Sequence extraction success rate: {successful}/{total} ({success_rate:.1f}%)"
        )

        return df

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract causal detection labels from response."""
        self._stats["responses_extracted"] += 1

        tokens = sample.get(self.config.text_field, []) if sample else []
        extracted = self.extract_label_from_response(raw_response, tokens)

        if extracted is None:
            self._stats["extraction_failures"] += 1
            logger.debug(f"Failed to extract labels from: {raw_response[:100]}...")

        return extracted


# Register the task
def register_causal_detection_task():
    """Register Causal Detection task."""
    from bench_forge.tasks.registry import get_registry

    registry = get_registry()
    registry.register("causal_detection", CausalDetectionTask)
    logger.info("Registered Causal Detection task with BIO sequence labeling")


# Auto-register when imported
try:
    register_causal_detection_task()
except Exception as e:
    logger.warning(f"Could not auto-register Causal Detection task: {e}")