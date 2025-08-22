"""TATQA (Table and Text QA) task implementation for BenchForge."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class TATQAConfig(FLAMEConfig):
    """Configuration for TATQA task."""

    # Dataset configuration  
    huggingface_dataset: str = "gtfintechlab/TATQA"
    text_field: str = "text"
    question_field: str = "query"
    label_field: str = "answer"
    dataset_split: str = "test"

    # TATQA-specific fields
    financial_domain: str = "table_text_qa"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "tatqa"
        super().__post_init__()


@flame_task("tatqa")
class TATQATask(FLAMETask):
    """TATQA task for table-and-text question answering.
    
    This task requires models to answer questions by extracting information
    from both tables and text in the provided context. Often involves
    arithmetic reasoning and numerical answer extraction.
    
    Features:
    - Combined table and text reasoning
    - Arithmetic computation support
    - Numerical answer extraction
    - FLAME-compatible evaluation
    - Multi-strategy extraction for various response formats
    
    Input format:
    - text: Context containing both tables and text
    - query: The question to answer 
    - answer: Ground truth answer (usually numerical)
    """

    def __init__(self, config: Optional[TATQAConfig] = None):
        """Initialize TATQA task."""
        if config is None:
            config = TATQAConfig(name="tatqa")
        elif not isinstance(config, TATQAConfig):
            tatqa_config = TATQAConfig(**config.__dict__)
            config = tatqa_config

        super().__init__(config)
        self.config: TATQAConfig = config

        logger.info("Initialized TATQA task with table-and-text QA capabilities")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for TATQA using exact FLAME prompt."""
        format = format or self.config.prompt_format

        # Extract components
        context_text = sample.get(self.config.text_field, "")
        question = sample.get(self.config.question_field, "")
        
        # Combine context and question as FLAME does
        combined_context = f"{context_text} {question}"

        if format == PromptFormat.ZERO_SHOT:
            # Use exact FLAME prompt
            prompt = f"""Discard all previous instructions. Behave like an expert in table-and-text-based financial question answering.
Your task is to answer a question by extracting relevant information from both tables and text
provided in the context. Ensure that you use both sources comprehensively to generate an accurate response. Repeat your final answer at the
end of your response.
The context: {combined_context}"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""You are an expert in table-and-text-based question answering. Analyze the provided context to answer questions.

Example:
Context: Table: Year | Revenue | Profit
2020 | $100M | $20M  
2021 | $120M | $25M
Text: The company showed strong growth in 2021.
Question: What was the profit increase from 2020 to 2021?
Answer: The profit increased from $20M to $25M, which is an increase of $5M.

Now analyze this context and answer the question:
Context: {combined_context}
"""

        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            prompt = f"""Let's solve this table-and-text question step by step.

Context: {combined_context}

Let me analyze this systematically:
1. First, I'll identify the relevant information from both tables and text
2. Then, I'll determine what calculation or reasoning is needed
3. Finally, I'll provide the answer

Step 1 - Relevant information:
Step 2 - Required reasoning:
Step 3 - Final answer:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract numerical answer using multi-strategy approach."""
        self._stats["responses_extracted"] += 1

        if not raw_response:
            return None

        # Clean the response
        response = raw_response.strip()

        # Strategy 1: Extract final answer if explicitly stated
        extracted = self._extract_final_answer(response)
        if extracted is not None:
            return extracted

        # Strategy 2: Extract from structured formats
        extracted = self._extract_structured_answer(response)
        if extracted is not None:
            return extracted

        # Strategy 3: Extract last number mentioned
        extracted = self._extract_last_number(response)
        if extracted is not None:
            return extracted

        # Strategy 4: Extract percentage if applicable
        extracted = self._extract_percentage(response)
        if extracted is not None:
            return extracted

        # Strategy 5: Extract currency amounts
        extracted = self._extract_currency(response)
        if extracted is not None:
            return extracted

        # Strategy 6: Extract any numerical value
        extracted = self._extract_any_number(response)
        if extracted is not None:
            return extracted

        logger.debug(f"Failed to extract answer from response: {raw_response[:100]}...")
        self._stats["extraction_failures"] += 1
        return None

    def _extract_final_answer(self, response: str) -> Optional[str]:
        """Extract explicitly stated final answer."""
        # Look for patterns like "Final answer: X", "The answer is X", etc.
        final_patterns = [
            r'final answer[:\s]+([^\n.]+)',
            r'the answer is[:\s]+([^\n.]+)',
            r'answer[:\s]+([^\n.]+)',
            r'result[:\s]+([^\n.]+)',
        ]
        
        for pattern in final_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()  # Take the last match
                # Clean and extract number from answer
                cleaned = self._clean_numerical_answer(answer)
                if cleaned:
                    return cleaned
        
        return None

    def _extract_structured_answer(self, response: str) -> Optional[str]:
        """Extract from structured response formats."""
        # Look for JSON-like structures
        try:
            # Simple JSON extraction
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*:([^{}]+)\}', response, re.IGNORECASE)
            if json_match:
                answer = json_match.group(1).strip().strip('"')
                cleaned = self._clean_numerical_answer(answer)
                if cleaned:
                    return cleaned
        except Exception:
            pass

        # Look for key-value pairs
        kv_patterns = [
            r'answer\s*[:=]\s*([^\n,]+)',
            r'result\s*[:=]\s*([^\n,]+)',
            r'value\s*[:=]\s*([^\n,]+)',
        ]
        
        for pattern in kv_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                cleaned = self._clean_numerical_answer(answer)
                if cleaned:
                    return cleaned
        
        return None

    def _extract_last_number(self, response: str) -> Optional[str]:
        """Extract the last numerical value mentioned."""
        # Find all numbers (including percentages, currencies, decimals)
        number_patterns = [
            r'\$?[\d,]+\.?\d*%?',  # Currency, percentages, decimals
            r'[\d,]+\.?\d*\s*(?:million|billion|trillion|M|B|T)',  # Large numbers
            r'[\d,]+\.?\d*',  # Basic numbers
        ]
        
        all_numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            all_numbers.extend(matches)
        
        if all_numbers:
            # Return the last number found
            last_number = all_numbers[-1]
            cleaned = self._clean_numerical_answer(last_number)
            if cleaned:
                return cleaned
        
        return None

    def _extract_percentage(self, response: str) -> Optional[str]:
        """Extract percentage values."""
        percentage_patterns = [
            r'([\d,]*\.?\d+)%',
            r'([\d,]*\.?\d+)\s*percent',
        ]
        
        for pattern in percentage_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Return the last percentage found
                percentage = matches[-1].strip()
                return f"{percentage}%"
        
        return None

    def _extract_currency(self, response: str) -> Optional[str]:
        """Extract currency amounts."""
        currency_patterns = [
            r'\$?([\d,]+\.?\d*)\s*(?:million|billion|trillion|M|B|T)?',
            r'([\d,]+\.?\d*)\s*dollars?',
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                amount = matches[-1].strip()
                return self._clean_numerical_answer(amount)
        
        return None

    def _extract_any_number(self, response: str) -> Optional[str]:
        """Extract any numerical value as fallback."""
        # Very broad number extraction
        numbers = re.findall(r'[\d,]*\.?\d+', response)
        if numbers:
            # Return the last number found
            number = numbers[-1]
            return self._clean_numerical_answer(number)
        
        return None

    def _clean_numerical_answer(self, answer: str) -> Optional[str]:
        """Clean and validate numerical answer."""
        if not answer:
            return None
            
        # Remove common prefixes/suffixes
        answer = answer.strip()
        answer = re.sub(r'^(is|was|are|were|equals?|=)\s*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\s*(dollars?|usd|\$)\s*$', '', answer, flags=re.IGNORECASE)
        
        # Clean up formatting
        answer = answer.replace(',', '')  # Remove commas
        answer = answer.strip()
        
        # Validate it contains numbers
        if re.search(r'\d', answer):
            return answer
        
        return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth answer from sample."""
        return sample.get(self.config.label_field, "")

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
            context = sample.get(self.config.text_field, "")
            question = sample.get(self.config.question_field, "")
            combined_context = f"{context} {question}"
            actual_answer = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields for TATQA
                "context": combined_context,  # FLAME primary field
                "response": response_text,  # FLAME primary field  
                "actual_answer": actual_answer,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # FLAME primary field
                # TATQA-specific fields
                "text": context,
                "query": question,
                "answer": actual_answer,
                # Standard BenchForge fields
                "prompt": prompt,
                "input": combined_context,
                "ground_truth": actual_answer,
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
            f"TATQA extraction results: {successful}/{total} successful extractions ({success_rate:.1f}%)"
        )

        return df