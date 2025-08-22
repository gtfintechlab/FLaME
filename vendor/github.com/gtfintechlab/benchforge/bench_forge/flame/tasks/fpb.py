"""Financial Phrase Bank (FPB) sentiment analysis task."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class FPBConfig(FLAMEConfig):
    """Configuration for Financial Phrase Bank task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/financial_phrasebank_sentences_allagree"
    dataset_name: str = "5768"
    text_field: str = "sentence"
    label_field: str = "label"
    dataset_split: str = "test"

    # FPB-specific fields
    valid_labels: List[str] = field(default_factory=lambda: ["NEGATIVE", "NEUTRAL", "POSITIVE"])
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "NEGATIVE": 0,
            "NEUTRAL": 1, 
            "POSITIVE": 2
        }
    )

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "fpb"
        super().__post_init__()


@flame_task("fpb")
class FPBTask(FLAMETask):
    """Financial Phrase Bank sentiment classification task.
    
    This task classifies financial sentences into three sentiment categories:
    - POSITIVE: Sentences expressing positive financial sentiment
    - NEGATIVE: Sentences expressing negative financial sentiment  
    - NEUTRAL: Sentences with neutral financial sentiment
    
    Features:
    - Uses exact FLAME prompt for consistency
    - Robust multi-strategy label extraction
    - FLAME-compatible column names and output format
    - Support for both zero-shot and few-shot prompting
    """

    def __init__(self, config: Optional[FPBConfig] = None):
        """Initialize FPB task."""
        if config is None:
            config = FPBConfig(name="fpb")
        elif not isinstance(config, FPBConfig):
            fpb_config = FPBConfig(**config.__dict__)
            config = fpb_config

        super().__init__(config)
        self.config: FPBConfig = config

        logger.info("Initialized FPB task with 3-class sentiment classification")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for FPB sentiment classification using exact FLAME prompt."""
        format = format or self.config.prompt_format

        # Extract sentence from sample
        sentence = sample.get(self.config.text_field, "")
        if not sentence:
            sentence = sample.get("text", sample.get("content", ""))

        if format == PromptFormat.ZERO_SHOT:
            # Use exact FLAME prompt
            prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence classifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide the label in the first line and provide a short explanation in the second line. This is the sentence: {sentence}"""

        elif format == PromptFormat.FEW_SHOT:
            # Create few-shot prompt with examples
            prompt = f"""Classify financial sentences into sentiment categories.

Example 1:
Sentence: The company reported strong quarterly earnings exceeding analyst expectations.
Classification: POSITIVE
Explanation: The sentence expresses positive sentiment about company performance.

Example 2:
Sentence: The stock price declined following news of regulatory investigation.
Classification: NEGATIVE  
Explanation: The sentence conveys negative sentiment due to declining stock price and regulatory concerns.

Example 3:
Sentence: The quarterly report will be released next Tuesday.
Classification: NEUTRAL
Explanation: The sentence is factual without expressing positive or negative sentiment.

Now classify this sentence:
Sentence: {sentence}
Classification:"""

        else:
            # Default to zero-shot
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract sentiment label from response using multi-strategy approach."""
        self._stats["responses_extracted"] += 1

        if not raw_response:
            return None

        # Clean the response
        response = raw_response.strip()

        # Strategy 1: Extract from first line (FLAME format)
        lines = response.split('\n')
        if lines:
            first_line = lines[0].strip().upper()
            for label in self.config.valid_labels:
                if label.upper() in first_line:
                    return label

        # Strategy 2: Look for labels anywhere in response
        response_upper = response.upper()
        for label in self.config.valid_labels:
            if label.upper() in response_upper:
                return label

        # Strategy 3: Pattern matching for common formats
        patterns = [
            r"(?:classification|label|sentiment):\s*(\w+)",
            r"(?:answer|result):\s*(\w+)",
            r"^(\w+)(?:\s*[:\-\.]|\s*$)",  # Label at start of line
            r"'\s*(\w+)\s*'",  # Label in quotes
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_upper, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                match_clean = match.strip()
                if match_clean in [label.upper() for label in self.config.valid_labels]:
                    # Return the properly cased version
                    for label in self.config.valid_labels:
                        if label.upper() == match_clean:
                            return label

        # Strategy 4: Partial matching for variations
        if "POSIT" in response_upper or "POS " in response_upper:
            return "POSITIVE"
        elif "NEGAT" in response_upper or "NEG " in response_upper:
            return "NEGATIVE"
        elif "NEUTR" in response_upper or "NEUT" in response_upper:
            return "NEUTRAL"

        logger.debug(f"Failed to extract sentiment from response: {raw_response[:100]}...")
        self._stats["extraction_failures"] += 1
        return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth label from sample."""
        label = sample.get(self.config.label_field)
        
        # Handle different label formats
        if isinstance(label, int):
            # Convert numeric to string label
            reverse_mapping = {v: k for k, v in self.config.label_mapping.items()}
            return reverse_mapping.get(label, label)
        elif isinstance(label, str):
            # Normalize string label
            label_upper = label.strip().upper()
            for valid_label in self.config.valid_labels:
                if valid_label.upper() == label_upper:
                    return valid_label
            return label
        
        return label

    def format_results(
        self,
        samples: List[Dict[str, Any]], 
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results with FLAME-compatible column names."""
        results = []

        for i, (sample, prompt, raw_response, extracted) in enumerate(
            zip(samples, prompts, raw_responses, extracted_responses)
        ):
            # Handle raw response format
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

            # Extract if not already done
            if extracted is None and response_text:
                extracted = self.extract_response(response_text, sample)

            # Get sample data
            sentence = sample.get(self.config.text_field, "")
            ground_truth = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields
                "sentences": sentence,  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field
                "actual_labels": ground_truth,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # FLAME primary field
                # FPB-specific fields
                "sentence": sentence,
                "sentiment": extracted,
                "confidence": 1.0 if extracted else 0.0,
                # Standard BenchForge fields
                "prompt": prompt,
                "input": sentence,
                "ground_truth": ground_truth,
                "raw_response": response_text,
                "extracted_response": extracted,
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Log extraction statistics
        total = len(df)
        successful = df["extracted_labels"].notna().sum()
        success_rate = (successful / total * 100) if total > 0 else 0
        
        logger.info(
            f"FPB extraction results: {successful}/{total} successful extractions ({success_rate:.1f}%)"
        )

        # Log label distribution
        if successful > 0:
            label_counts = df["extracted_labels"].value_counts()
            logger.info(f"Label distribution: {label_counts.to_dict()}")

        return df