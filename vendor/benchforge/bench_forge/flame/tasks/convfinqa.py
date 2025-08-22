"""ConvFinQA task with multi-turn conversation handling for BenchForge."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat
from bench_forge.prompts.extractor import ExtractionStrategy

logger = logging.getLogger(__name__)


@dataclass
class ConvFinQAConfig(FLAMEConfig):
    """Configuration for ConvFinQA task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/convfinqa"
    text_field: str = "combined_text"
    label_field: str = "answer_1"
    dataset_split: str = "dev"  # ConvFinQA uses 'dev' split for testing

    # ConvFinQA-specific fields
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.REGEX
    financial_domain: str = "conversation_qa"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "convfinqa"
        super().__post_init__()


@flame_task("convfinqa")
class ConvFinQATask(FLAMETask):
    """ConvFinQA task for financial conversation QA.

    Features:
    - Multi-turn conversation context handling
    - Financial table and text integration
    - Numeric answer extraction and validation
    - Conversation flow understanding
    - Complex financial reasoning

    Input format:
    - pre_text: List of text segments before table
    - post_text: List of text segments after table
    - table_ori: Original table data (list of lists)
    - question_0: First question in conversation
    - answer_0: First answer in conversation
    - question_1: Follow-up question (target)
    - answer_1: Target answer (ground truth)

    The task creates a combined context including all conversation elements
    and asks the model to answer the follow-up question.
    """

    def __init__(self, config: Optional[ConvFinQAConfig] = None):
        """Initialize ConvFinQA task."""
        if config is None:
            config = ConvFinQAConfig(name="convfinqa")
        elif not isinstance(config, ConvFinQAConfig):
            convfinqa_config = ConvFinQAConfig(**config.__dict__)
            config = convfinqa_config

        super().__init__(config)
        self.config: ConvFinQAConfig = config

        logger.info("Initialized ConvFinQA task with multi-turn conversation handling")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for ConvFinQA with multi-turn conversation context."""
        format = format or self.config.prompt_format

        # Extract conversation components
        pre_text = self._extract_text_list(sample.get("pre_text", []))
        post_text = self._extract_text_list(sample.get("post_text", []))
        table_text = self._format_table(sample.get("table_ori", []))

        question_0 = (
            str(sample.get("question_0", "")) if sample.get("question_0") else ""
        )
        answer_0 = str(sample.get("answer_0", "")) if sample.get("answer_0") else ""
        question_1 = (
            str(sample.get("question_1", "")) if sample.get("question_1") else ""
        )

        # Build conversation context
        context_parts = []
        if pre_text:
            context_parts.append(f"Context: {pre_text}")
        if table_text:
            context_parts.append(f"Table: {table_text}")
        if post_text:
            context_parts.append(f"Additional Information: {post_text}")

        full_context = " ".join(context_parts)

        if format == PromptFormat.ZERO_SHOT:
            prompt = f"""Analyze the following financial information and conversation to answer the follow-up question.

{full_context}

Conversation:
Q1: {question_0}
A1: {answer_0}

Follow-up Question: {question_1}

Provide a precise answer based on the financial information and conversation context. If the answer is numerical, provide only the number and unit (if applicable).

Answer:"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""Analyze financial conversations to answer follow-up questions based on context and previous Q&A pairs.

Example:
Context: Company ABC reported revenue of $100M in Q1 and $120M in Q2.
Q1: What was the revenue in Q1?
A1: $100M
Follow-up Question: What was the percentage increase from Q1 to Q2?
Answer: 20%

Now analyze:
{full_context}

Conversation:
Q1: {question_0}
A1: {answer_0}

Follow-up Question: {question_1}

Answer:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def _extract_text_list(self, text_data: Any) -> str:
        """Extract and join text from various list formats."""
        if isinstance(text_data, list):
            return " ".join(str(item) for item in text_data)
        elif isinstance(text_data, str):
            return text_data
        else:
            return str(text_data) if text_data else ""

    def _format_table(self, table_data: Any) -> str:
        """Format table data into readable text."""
        if not table_data:
            return ""

        try:
            if isinstance(table_data, list):
                # List of lists (rows)
                formatted_rows = []
                for row in table_data:
                    if isinstance(row, list):
                        row_text = " | ".join(str(cell) for cell in row)
                        formatted_rows.append(row_text)
                    else:
                        formatted_rows.append(str(row))
                return " ".join(formatted_rows)
            else:
                return str(table_data)
        except Exception as e:
            logger.debug(f"Error formatting table: {e}")
            return str(table_data)

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract answer from ConvFinQA response with numeric handling."""
        self._stats["responses_extracted"] += 1

        try:
            extracted = self._extract_numeric_answer(raw_response)

            if extracted is None:
                self._stats["extraction_failures"] += 1
                logger.debug(f"Failed to extract answer from: {raw_response[:100]}...")

            return extracted

        except Exception as e:
            self._stats["extraction_failures"] += 1
            logger.error(f"Extraction error: {e}")
            return None

    def _extract_numeric_answer(self, response: str) -> Optional[str]:
        """Extract numeric answer using multiple strategies."""
        if not response:
            return None

        response = response.strip()

        # Strategy 1: Direct numeric extraction (handles percentages, currency, etc.)
        numeric_patterns = [
            r"^\$?[\d,]+\.?\d*%?$",  # Simple numbers with optional $ and %
            r"(?:^|\s)(\$?[\d,]+\.?\d*%?)(?:\s|$)",  # Numbers in text
            r"(?:Answer:?\s*)?(\$?[\d,]+\.?\d*%?)(?:\s|\.|\!|$)",  # After "Answer:"
        ]

        for pattern in numeric_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                if pattern.startswith("(?:^|\\s)"):
                    extracted = match.group(1)
                else:
                    extracted = (
                        match.group(0) if match.groups() == () else match.group(1)
                    )

                # Clean up the extracted value
                extracted = extracted.strip(".,!?")
                if extracted:
                    logger.debug(
                        f"Extracted numeric answer: '{extracted}' using pattern: {pattern}"
                    )
                    return extracted

        # Strategy 2: Look for numbers in structured format
        answer_patterns = [
            r"(?:answer|result|conclusion|final answer):?\s*([^\n\r]+)",
            r"(?:the answer is|answer:)\s*([^\n\r]+)",
            r"(?:solution|response):?\s*([^\n\r]+)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                # Try to extract number from the answer text
                num_match = re.search(r"\$?[\d,]+\.?\d*%?", answer_text)
                if num_match:
                    extracted = num_match.group(0)
                    logger.debug(f"Extracted from structured answer: '{extracted}'")
                    return extracted

        # Strategy 3: First clean number found in response
        clean_numbers = re.findall(r"\$?[\d,]+\.?\d*%?", response)
        if clean_numbers:
            extracted = clean_numbers[0]
            logger.debug(f"Extracted first number: '{extracted}'")
            return extracted

        # Strategy 4: Handle word responses that might be valid
        if len(response.split()) <= 5:  # Short responses might be valid answers
            # Clean and return if it's a reasonable short answer
            cleaned = re.sub(r"[^\w\s\$%.,]", "", response).strip()
            if cleaned:
                logger.debug(f"Extracted short response: '{cleaned}'")
                return cleaned

        logger.warning(f"Could not extract answer from response: {response[:100]}...")
        return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth from ConvFinQA sample."""
        # ConvFinQA uses answer_1 as the target answer
        return sample.get("answer_1") or sample.get(self.config.label_field)

    def format_results(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results with ConvFinQA-specific fields."""
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

            # Build conversation context for reference
            combined_text = self._build_combined_text(sample)

            result = {
                "index": i,
                # FLAME-compatible fields
                "context": combined_text,
                "response": response_text,
                "actual_label": self.get_ground_truth(sample),
                "complete_responses": complete_response,
                # ConvFinQA-specific fields
                "pre_text": sample.get("pre_text", []),
                "post_text": sample.get("post_text", []),
                "table_ori": sample.get("table_ori", []),
                "question_0": sample.get("question_0", ""),
                "answer_0": sample.get("answer_0", ""),
                "question_1": sample.get("question_1", ""),
                "answer_1": sample.get("answer_1", ""),
                "extracted_answer": extracted,
                # Standard BenchForge fields
                "prompt": prompt,
                "input": combined_text,
                "ground_truth": self.get_ground_truth(sample),
                "raw_response": response_text,
                "extracted_response": extracted,
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Log extraction statistics
        total = len(df)
        successful = df["extracted_answer"].notna().sum()
        success_rate = (successful / total * 100) if total > 0 else 0
        logger.info(
            f"ConvFinQA extraction success rate: {successful}/{total} ({success_rate:.1f}%)"
        )

        return df

    def _build_combined_text(self, sample: Dict[str, Any]) -> str:
        """Build combined text context for FLAME compatibility."""
        pre_text = self._extract_text_list(sample.get("pre_text", []))
        post_text = self._extract_text_list(sample.get("post_text", []))
        table_text = self._format_table(sample.get("table_ori", []))
        question_0 = (
            str(sample.get("question_0", "")) if sample.get("question_0") else ""
        )
        answer_0 = str(sample.get("answer_0", "")) if sample.get("answer_0") else ""
        question_1 = (
            str(sample.get("question_1", "")) if sample.get("question_1") else ""
        )

        return f"{pre_text} {post_text} {table_text} Question 0: {question_0} Answer: {answer_0}. Now answer the following question: {question_1}"
