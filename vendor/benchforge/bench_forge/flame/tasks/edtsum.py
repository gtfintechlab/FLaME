"""EDTSum QA task with earnings call question answering for BenchForge."""

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
class EDTSumConfig(FLAMEConfig):
    """Configuration for EDTSum QA task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/edtsum"
    text_field: str = "summary"
    label_field: str = "question"
    dataset_split: str = "test"

    # EDTSum-specific fields
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.REGEX
    financial_domain: str = "earnings_qa"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "edtsum"
        super().__post_init__()


@flame_task("edtsum")
class EDTSumTask(FLAMETask):
    """EDTSum task for earnings call question answering.

    Features:
    - Earnings call transcript analysis
    - Financial Q&A generation and answering
    - Summary-based question generation
    - Financial communication understanding
    - Executive communication analysis

    Input format:
    - summary: Summary of earnings call segment
    - question: Expected question to be generated/answered
    - transcript: Full transcript context (if available)
    - company: Company name
    - quarter: Financial quarter

    The task generates relevant questions about earnings call summaries
    or answers questions based on earnings call content.
    """

    def __init__(self, config: Optional[EDTSumConfig] = None):
        """Initialize EDTSum task."""
        if config is None:
            config = EDTSumConfig(name="edtsum")
        elif not isinstance(config, EDTSumConfig):
            edtsum_config = EDTSumConfig(**config.__dict__)
            config = edtsum_config

        super().__init__(config)
        self.config: EDTSumConfig = config

        logger.info("Initialized EDTSum task with earnings call QA")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for EDTSum QA task."""
        format = format or self.config.prompt_format

        # Extract earnings call components
        summary = sample.get("summary", "")
        question = sample.get("question", "")
        company = sample.get("company", "")
        quarter = sample.get("quarter", "")
        transcript = sample.get("transcript", "")

        # Determine task type based on available data
        if question and not transcript:
            # Question generation task
            task_type = "generation"
        else:
            # Question answering task
            task_type = "answering"

        if format == PromptFormat.ZERO_SHOT:
            if task_type == "generation":
                prompt = f"""Generate a relevant question about the following earnings call summary.

Company: {company}
Quarter: {quarter}

Earnings Call Summary:
{summary}

Generate a specific, insightful question that an analyst or investor might ask based on this summary. The question should focus on financial performance, business strategy, or market outlook.

Question:"""
            else:
                context = transcript or summary
                prompt = f"""Answer the following question based on the earnings call information.

Company: {company}
Quarter: {quarter}

Earnings Call Context:
{context}

Question: {question}

Provide a comprehensive answer based on the earnings call information. Include specific details and financial metrics when available.

Answer:"""

        elif format == PromptFormat.FEW_SHOT:
            if task_type == "generation":
                prompt = f"""Generate questions about earnings call summaries.

Example:
Company: Apple Inc.
Summary: Apple reported record iPhone sales with revenue growth of 15% year-over-year despite supply chain challenges.
Question: What specific factors contributed to the strong iPhone performance despite ongoing supply chain issues?

Now generate:
Company: {company}
Quarter: {quarter}
Summary: {summary}

Question:"""
            else:
                context = transcript or summary
                prompt = f"""Answer earnings call questions based on provided context.

Example:
Context: Management discussed strong Q3 performance with 20% revenue growth driven by new product launches.
Question: What drove the revenue growth this quarter?
Answer: The 20% revenue growth was primarily driven by new product launches, as mentioned by management.

Now answer:
Company: {company}
Context: {context}
Question: {question}

Answer:"""

        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            if task_type == "generation":
                prompt = f"""Let me analyze this earnings call summary to generate a relevant question.

Company: {company}
Quarter: {quarter}
Summary: {summary}

Step 1: Key information analysis
Step 2: Identify areas of interest for investors
Step 3: Formulate a specific question

Analysis:
Step 1 - Key information:
Step 2 - Investor interests:
Step 3 - Generated question:"""
            else:
                context = transcript or summary
                prompt = f"""Let me analyze the earnings call to answer this question.

Company: {company}
Context: {context}
Question: {question}

Step 1: Understand the question
Step 2: Find relevant information in the context
Step 3: Formulate a comprehensive answer

Analysis:
Step 1 - Question analysis:
Step 2 - Relevant information:
Step 3 - Answer:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract answer or question from EDTSum response."""
        self._stats["responses_extracted"] += 1

        try:
            extracted = self._extract_qa_content(raw_response)

            if extracted is None:
                self._stats["extraction_failures"] += 1
                logger.debug(
                    f"Failed to extract QA content from: {raw_response[:100]}..."
                )

            return extracted

        except Exception as e:
            self._stats["extraction_failures"] += 1
            logger.error(f"EDTSum extraction error: {e}")
            return None

    def _extract_qa_content(self, response: str) -> Optional[str]:
        """Extract question or answer content from response."""
        if not response:
            return None

        response = response.strip()

        # Strategy 1: Extract after markers
        qa_patterns = [
            r"(?:Question:?\s*)(.*?)(?:\n|$)",
            r"(?:Answer:?\s*)(.*?)(?:\n|$)",
            r"(?:Generated question:?\s*)(.*?)(?:\n|$)",
            r"(?:Step 3[^:]*:?\s*)(.*?)(?:\n|$)",
            r"(?:Final answer:?\s*)(.*?)(?:\n|$)",
        ]

        for pattern in qa_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted and len(extracted.split()) >= 3:  # Reasonable length
                    logger.debug(
                        f"Extracted QA content using pattern: {pattern[:20]}..."
                    )
                    return extracted

        # Strategy 2: Clean the entire response if it looks like a direct answer
        if not any(
            marker in response.lower()
            for marker in ["step 1", "step 2", "analysis:", "example:"]
        ):
            # Likely a direct answer/question
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line and len(line.split()) >= 3:
                    # Remove common prefixes
                    prefixes = [
                        "Question:",
                        "Answer:",
                        "Generated question:",
                        "Final answer:",
                    ]
                    for prefix in prefixes:
                        if line.startswith(prefix):
                            line = line[len(prefix) :].strip()

                    if line:
                        logger.debug(f"Extracted cleaned response: {line[:50]}...")
                        return line

        # Strategy 3: Extract the longest meaningful sentence
        sentences = re.split(r"[.!?]\s+", response)
        best_sentence = ""
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip sentences with analysis markers
            if any(
                marker in sentence.lower()
                for marker in ["step", "analysis", "example", "context"]
            ):
                continue

            if (
                len(sentence.split()) > len(best_sentence.split())
                and len(sentence.split()) >= 5
            ):
                best_sentence = sentence

        if best_sentence:
            logger.debug(f"Extracted best sentence: {best_sentence[:50]}...")
            return best_sentence

        # Strategy 4: Return first substantial line
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if (
                line
                and len(line.split()) >= 5
                and not line.lower().startswith(("step", "analysis", "example"))
            ):
                logger.debug(f"Extracted first substantial line: {line[:50]}...")
                return line

        logger.warning(
            f"Could not extract QA content from response: {response[:100]}..."
        )
        return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth from EDTSum sample."""
        # EDTSum might have different target fields depending on task type
        return (
            sample.get("question")
            or sample.get("answer")
            or sample.get("expected_question")
            or sample.get("expected_answer")
            or sample.get(self.config.label_field)
        )

    def format_results(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results with EDTSum-specific fields."""
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
            summary = sample.get("summary", "")
            company = sample.get("company", "")
            quarter = sample.get("quarter", "")
            ground_truth = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields
                "context": summary,
                "response": response_text,
                "actual_label": ground_truth,
                "complete_responses": complete_response,
                # EDTSum-specific fields
                "summary": summary,
                "company": company,
                "quarter": quarter,
                "expected_question": sample.get("question", ""),
                "expected_answer": sample.get("answer", ""),
                "extracted_content": extracted,
                "transcript": sample.get("transcript", ""),
                # Standard BenchForge fields
                "prompt": prompt,
                "input": summary,
                "ground_truth": ground_truth,
                "raw_response": response_text,
                "extracted_response": extracted,
            }

            results.append(result)

        df = pd.DataFrame(results)

        # Log extraction statistics
        total = len(df)
        successful = df["extracted_content"].notna().sum()
        success_rate = (successful / total * 100) if total > 0 else 0
        logger.info(
            f"EDTSum extraction success rate: {successful}/{total} ({success_rate:.1f}%)"
        )

        return df
