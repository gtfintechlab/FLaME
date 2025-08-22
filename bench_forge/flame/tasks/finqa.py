"""FinQA task with financial table processing for BenchForge."""

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
class FinQAConfig(FLAMEConfig):
    """Configuration for FinQA task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/finqa"
    text_field: str = "combined_text"
    label_field: str = "answer"
    dataset_split: str = "test"  # FinQA uses 'test' split

    # FinQA-specific fields
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.REGEX
    financial_domain: str = "financial_qa"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "finqa"
        super().__post_init__()


@flame_task("finqa")
class FinQATask(FLAMETask):
    """FinQA task for financial question answering with table reasoning.
    
    Features:
    - Financial table and text integration
    - Multi-hop reasoning over financial data
    - Numeric answer extraction and validation
    - Complex financial calculations
    - Program synthesis for computation chains
    
    Input format:
    - pre_text: List of text segments before table
    - post_text: List of text segments after table  
    - table_ori: Original table data (list of lists)
    - question: The question to answer
    - answer: Target answer (ground truth)
    
    The task requires reasoning over both textual and tabular financial data
    to answer questions that often involve calculations.
    """

    def __init__(self, config: Optional[FinQAConfig] = None):
        """Initialize FinQA task."""
        if config is None:
            config = FinQAConfig(name="finqa")
        elif not isinstance(config, FinQAConfig):
            finqa_config = FinQAConfig(**config.__dict__)
            config = finqa_config

        super().__init__(config)
        self.config: FinQAConfig = config

        logger.info("Initialized FinQA task with financial table processing")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for FinQA with financial table context."""
        format = format or self.config.prompt_format

        # Extract financial components
        pre_text = self._extract_text_list(sample.get("pre_text", []))
        post_text = self._extract_text_list(sample.get("post_text", []))
        table_text = self._format_financial_table(sample.get("table_ori", []))
        question = str(sample.get("question", ""))

        # Build financial context
        context_parts = []
        if pre_text:
            context_parts.append(f"Financial Report: {pre_text}")
        if table_text:
            context_parts.append(f"Financial Table:\n{table_text}")
        if post_text:
            context_parts.append(f"Additional Information: {post_text}")

        full_context = "\n\n".join(context_parts)

        if format == PromptFormat.ZERO_SHOT:
            prompt = f"""Analyze the financial information and answer the question. Provide calculations if needed.

{full_context}

Question: {question}

Instructions:
- Use the provided financial data to answer the question
- If calculations are required, show your reasoning
- Provide the final answer as a number with appropriate units
- If the answer cannot be determined, state "Cannot be determined"

Answer:"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""Analyze financial data to answer questions. Show calculations when needed.

Example:
Financial Report: Company revenue increased from $100M to $120M.
Financial Table:
Year | Revenue
2021 | $100M
2022 | $120M

Question: What is the percentage increase in revenue?
Answer: The percentage increase is (120-100)/100 * 100 = 20%

Now analyze:
{full_context}

Question: {question}

Answer:"""

        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            prompt = f"""Let's analyze this financial question step by step.

{full_context}

Question: {question}

Let me break this down:
1. First, I'll identify the relevant data
2. Then, I'll determine what calculation is needed
3. Finally, I'll compute the answer

Step 1 - Relevant data:
Step 2 - Required calculation:
Step 3 - Final answer:"""

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

    def _format_financial_table(self, table_data: Any) -> str:
        """Format financial table data into readable format with proper alignment."""
        if not table_data:
            return ""
        
        try:
            if isinstance(table_data, list) and len(table_data) > 0:
                # List of lists (rows)
                formatted_table = []
                
                # Process first row as potential header
                if table_data:
                    first_row = table_data[0]
                    if isinstance(first_row, list):
                        header = " | ".join(str(cell).strip() for cell in first_row)
                        formatted_table.append(header)
                        
                        # Add separator line
                        separator = " | ".join("-" * len(str(cell).strip()) for cell in first_row)
                        formatted_table.append(separator)
                        
                        # Process remaining rows
                        for row in table_data[1:]:
                            if isinstance(row, list):
                                row_text = " | ".join(str(cell).strip() for cell in row)
                                formatted_table.append(row_text)
                
                return "\n".join(formatted_table)
            else:
                return str(table_data)
        except Exception as e:
            logger.debug(f"Error formatting financial table: {e}")
            return str(table_data)

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract answer from FinQA response with financial numeric handling."""
        self._stats["responses_extracted"] += 1

        try:
            extracted = self._extract_financial_answer(raw_response)
            
            if extracted is None:
                self._stats["extraction_failures"] += 1
                logger.debug(f"Failed to extract financial answer from: {raw_response[:100]}...")

            return extracted

        except Exception as e:
            self._stats["extraction_failures"] += 1
            logger.error(f"Financial extraction error: {e}")
            return None

    def _extract_financial_answer(self, response: str) -> Optional[str]:
        """Extract financial answer using specialized financial patterns."""
        if not response:
            return None

        response = response.strip()

        # Strategy 1: Financial-specific patterns (currency, percentages, ratios)
        financial_patterns = [
            r'\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|M|B))?',  # Currency with scale
            r'[\d,]+(?:\.\d+)?%',  # Percentages
            r'[\d,]+(?:\.\d+)?\s*(?:times|x)',  # Ratios (e.g., "2.5 times")
            r'[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|M|B|K)',  # Numbers with scale
            r'\$?[\d,]+(?:\.\d+)?',  # Basic numbers with optional currency
        ]
        
        for pattern in financial_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Take the last match as it's often the final answer
                extracted = matches[-1].strip()
                logger.debug(f"Extracted financial answer: '{extracted}' using pattern: {pattern}")
                return extracted

        # Strategy 2: Look for structured financial answers
        financial_answer_patterns = [
            r'(?:final answer|answer|result|conclusion):?\s*([^\n\r]+)',
            r'(?:the answer is|answer:)\s*([^\n\r]+)',
            r'(?:calculation result|computed value):?\s*([^\n\r]+)',
            r'(?:step 3|final calculation|final answer):.*?([^\n\r]+)',
        ]
        
        for pattern in financial_answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                
                # Extract financial number from the answer text
                for fin_pattern in financial_patterns:
                    fin_match = re.search(fin_pattern, answer_text, re.IGNORECASE)
                    if fin_match:
                        extracted = fin_match.group(0).strip()
                        logger.debug(f"Extracted from structured financial answer: '{extracted}'")
                        return extracted

        # Strategy 3: Mathematical expression evaluation (simple cases)
        math_patterns = [
            r'=\s*([\d,]+(?:\.\d+)?(?:\s*[%])?)',  # After equals sign
            r'(?:equals?|results? in)\s*([\d,]+(?:\.\d+)?(?:\s*[%])?)',
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                logger.debug(f"Extracted from mathematical expression: '{extracted}'")
                return extracted

        # Strategy 4: Fallback to any number in the response
        all_numbers = re.findall(r'[\d,]+(?:\.\d+)?', response)
        if all_numbers:
            # Prefer numbers that look like final answers (typically larger or with decimals)
            for number in reversed(all_numbers):  # Start from the end
                if '.' in number or len(number.replace(',', '')) >= 3:
                    logger.debug(f"Extracted fallback number: '{number}'")
                    return number
            
            # If no preferred number, take the last one
            extracted = all_numbers[-1]
            logger.debug(f"Extracted last number: '{extracted}'")
            return extracted

        # Strategy 5: Handle text answers for qualitative questions
        if len(response.split()) <= 10:  # Short responses might be valid
            # Clean and return if it's a reasonable short answer
            cleaned = re.sub(r'[^\w\s\$%.,]', '', response).strip()
            if cleaned and not cleaned.lower().startswith(('i ', 'the ', 'this ', 'sorry')):
                logger.debug(f"Extracted short text answer: '{cleaned}'")
                return cleaned

        logger.warning(f"Could not extract financial answer from response: {response[:100]}...")
        return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth from FinQA sample."""
        return sample.get("answer") or sample.get(self.config.label_field)

    def format_results(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results with FinQA-specific fields."""
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

            # Build combined text for FLAME compatibility
            combined_text = self._build_combined_text(sample)

            result = {
                "index": i,
                # FLAME-compatible fields
                "context": combined_text,
                "response": response_text,
                "actual_label": self.get_ground_truth(sample),
                "complete_responses": complete_response,
                # FinQA-specific fields
                "pre_text": sample.get("pre_text", []),
                "post_text": sample.get("post_text", []),
                "table_ori": sample.get("table_ori", []),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
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
            f"FinQA extraction success rate: {successful}/{total} ({success_rate:.1f}%)"
        )

        return df

    def _build_combined_text(self, sample: Dict[str, Any]) -> str:
        """Build combined text context for FLAME compatibility."""
        pre_text = self._extract_text_list(sample.get("pre_text", []))
        post_text = self._extract_text_list(sample.get("post_text", []))
        table_text = " ".join([" ".join(str(cell) for cell in row) for row in sample.get("table_ori", [])])
        question = str(sample.get("question", ""))

        return f"{pre_text} {post_text} {table_text} {question}"