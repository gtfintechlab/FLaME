"""ECTSum (Earnings Call Transcript Summarization) task implementation for BenchForge."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class ECTSumConfig(FLAMEConfig):
    """Configuration for ECTSum summarization task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/ECTSum"
    text_field: str = "context"
    label_field: str = "response"
    dataset_split: str = "test"

    # ECTSum-specific fields
    financial_domain: str = "earnings_summarization"
    max_summary_words: int = 50
    summary_format: str = "bullet_points"

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "ectsum"
        super().__post_init__()


@flame_task("ectsum")
class ECTSumTask(FLAMETask):
    """ECTSum task for earnings call transcript summarization.

    This task summarizes earnings call transcripts from Russell 3000 Index companies
    into concise bullet-point summaries following telegram-style format used in
    Reuters articles.

    Features:
    - Earnings call transcript summarization
    - Extractive + abstractive approach
    - Bullet-point format output (max 50 words)
    - Financial domain-specific understanding
    - FLAME-compatible evaluation with BERTScore metrics
    - Multi-strategy summary extraction

    Input format:
    - context: Earnings call transcript
    - response: Target summary (ground truth)
    """

    def __init__(self, config: Optional[ECTSumConfig] = None):
        """Initialize ECTSum task."""
        if config is None:
            config = ECTSumConfig(name="ectsum")
        elif not isinstance(config, ECTSumConfig):
            ectsum_config = ECTSumConfig(**config.__dict__)
            config = ectsum_config

        super().__init__(config)
        self.config: ECTSumConfig = config

        logger.info(
            "Initialized ECTSum task with earnings call summarization capabilities"
        )

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for ECTSum using exact FLAME prompt."""
        format = format or self.config.prompt_format

        # Extract transcript
        document = sample.get(self.config.text_field, "")

        if format == PromptFormat.ZERO_SHOT:
            # Use exact FLAME prompt
            prompt = f"""Discard all the previous instructions.
Behave like you are an expert at summarization tasks.
Below an earnings call transcript of a Russell 3000 Index company
is provided. Perform extractive summarization followed by
paraphrasing the transcript in bullet point format according to the
experts-written short telegram-style bullet point summaries
derived from corresponding Reuters articles. The target length of
the summary should be at most 50 words.

The document:
{document}"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""You are an expert at financial summarization. Summarize earnings call transcripts into concise bullet points.

Example:
Transcript: "Good morning everyone. Thank you for joining our Q3 earnings call. We're pleased to report record quarterly revenue of $2.5 billion, up 15% year-over-year. Our net income was $400 million, or $2.10 per share, beating analyst estimates..."

Summary:
• Q3 revenue $2.5B, +15% YoY
• Net income $400M, $2.10/share, beat estimates
• Strong performance across all segments

Now summarize this earnings call transcript (max 50 words, bullet format):
Transcript: {document}

Summary:"""

        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            prompt = f"""Let's analyze this earnings call transcript step by step for summarization.

Transcript: {document}

Step 1: Identify key financial metrics (revenue, profit, EPS, etc.)
Step 2: Note significant events or changes (YoY growth, guidance, etc.)
Step 3: Create concise bullet points highlighting the most important information
Step 4: Ensure summary is under 50 words and uses telegram style

Final summary in bullet points:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract summary using multi-strategy approach."""
        self._stats["responses_extracted"] += 1

        if not raw_response:
            return None

        # Clean the response
        response = raw_response.strip()

        # Strategy 1: Extract bullet points if present
        extracted = self._extract_bullet_points(response)
        if extracted:
            return extracted

        # Strategy 2: Extract from structured format
        extracted = self._extract_structured_summary(response)
        if extracted:
            return extracted

        # Strategy 3: Extract summary content (non-bullet format)
        extracted = self._extract_summary_content(response)
        if extracted:
            return extracted

        # Strategy 4: Clean up and return as-is if it looks like summary
        extracted = self._extract_clean_summary(response)
        if extracted:
            return extracted

        # Strategy 5: Extract first coherent paragraph
        extracted = self._extract_first_paragraph(response)
        if extracted:
            return extracted

        logger.debug(
            f"Failed to extract summary from response: {raw_response[:100]}..."
        )
        self._stats["extraction_failures"] += 1
        return None

    def _extract_bullet_points(self, response: str) -> Optional[str]:
        """Extract bullet-point formatted summaries."""
        # Look for bullet point patterns
        bullet_patterns = [
            r"[•·*-]\s*([^\n•·*-]+)",  # Bullet points
            r"^\s*[•·*-]\s*(.+)$",  # Line starting with bullet
            r"^\s*\d+\.\s*(.+)$",  # Numbered points
        ]

        bullet_points = []
        lines = response.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in bullet_patterns:
                matches = re.findall(pattern, line, re.MULTILINE)
                for match in matches:
                    if match.strip():
                        bullet_points.append(match.strip())

        if bullet_points:
            # Join bullet points with newlines and bullet markers
            summary = "\n".join([f"• {point}" for point in bullet_points])
            # Check word count
            if (
                self._count_words(summary) <= self.config.max_summary_words + 10
            ):  # Allow slight overage
                return summary

        return None

    def _extract_structured_summary(self, response: str) -> Optional[str]:
        """Extract from structured response formats."""
        # Look for "Summary:", "Key points:", etc.
        summary_patterns = [
            r"summary\s*:?\s*(.+?)(?:\n\n|\Z)",
            r"key points?\s*:?\s*(.+?)(?:\n\n|\Z)",
            r"main points?\s*:?\s*(.+?)(?:\n\n|\Z)",
            r"highlights?\s*:?\s*(.+?)(?:\n\n|\Z)",
            r"takeaways?\s*:?\s*(.+?)(?:\n\n|\Z)",
        ]

        for pattern in summary_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                summary = matches[0].strip()
                if (
                    summary
                    and self._count_words(summary) <= self.config.max_summary_words + 10
                ):
                    return self._format_as_bullets(summary)

        return None

    def _extract_summary_content(self, response: str) -> Optional[str]:
        """Extract main summary content."""
        # Remove common prefixes and get main content
        prefixes_to_remove = [
            r"^(summary|here is|here\'s|the summary|key points?|main points?|highlights?)\s*:?\s*",
            r"^(based on|according to|from the transcript)\s*,?\s*",
        ]

        content = response
        for prefix in prefixes_to_remove:
            content = re.sub(prefix, "", content, flags=re.IGNORECASE)

        content = content.strip()

        # Check if it's a reasonable summary length
        if content and self._count_words(content) <= self.config.max_summary_words + 15:
            return self._format_as_bullets(content)

        return None

    def _extract_clean_summary(self, response: str) -> Optional[str]:
        """Clean up response and check if it's a valid summary."""
        # Remove extra whitespace and clean up
        cleaned = re.sub(r"\s+", " ", response.strip())

        # Remove common AI response patterns
        patterns_to_remove = [
            r"I would|I\'d|I can|I will",
            r"Here is|Here\'s|This is",
            r"The following|Based on",
            r"In summary|To summarize",
        ]

        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = cleaned.strip()

        # Check if it looks like financial content and is reasonable length
        financial_keywords = [
            "revenue",
            "profit",
            "earnings",
            "income",
            "loss",
            "growth",
            "decline",
            "margin",
            "eps",
            "guidance",
        ]
        has_financial_content = any(
            keyword.lower() in cleaned.lower() for keyword in financial_keywords
        )

        if (
            has_financial_content
            and self._count_words(cleaned) <= self.config.max_summary_words + 20
        ):
            return self._format_as_bullets(cleaned)

        return None

    def _extract_first_paragraph(self, response: str) -> Optional[str]:
        """Extract the first coherent paragraph as summary."""
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]

        if paragraphs:
            first_para = paragraphs[0]
            if self._count_words(first_para) <= self.config.max_summary_words + 10:
                return self._format_as_bullets(first_para)

        return None

    def _format_as_bullets(self, text: str) -> str:
        """Format text as bullet points if not already formatted."""
        # If already has bullet points, return as-is
        if re.search(r"[•·*-]\s", text):
            return text

        # Split into sentences and create bullet points
        sentences = re.split(r"[.!?]+", text)
        bullets = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                bullets.append(f"• {sentence}")

        return "\n".join(bullets)

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        if not text:
            return 0
        return len(text.split())

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth summary from sample."""
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
            transcript = sample.get(self.config.text_field, "")
            actual_summary = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields for ECTSum
                "documents": transcript,  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field
                "actual_labels": actual_summary,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # FLAME primary field
                # ECTSum-specific fields
                "context": transcript,
                "transcript": transcript,
                "response": actual_summary,
                "summary": extracted,
                "word_count": self._count_words(extracted) if extracted else 0,
                # Standard BenchForge fields
                "prompt": prompt,
                "input": transcript,
                "ground_truth": actual_summary,
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
            f"ECTSum extraction results: {successful}/{total} successful extractions ({success_rate:.1f}%)"
        )

        # Log summary statistics
        if successful > 0:
            word_counts = df[df["extracted_labels"].notna()]["word_count"]
            avg_words = word_counts.mean()
            max_words = word_counts.max()
            logger.info(
                f"Summary statistics: avg={avg_words:.1f} words, max={max_words} words"
            )

        return df
