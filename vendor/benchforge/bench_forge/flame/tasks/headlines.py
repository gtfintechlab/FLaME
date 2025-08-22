"""Headlines multi-attribute news classification task."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class HeadlinesConfig(FLAMEConfig):
    """Configuration for Headlines multi-attribute classification task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/Headlines"
    dataset_name: str = "5768"
    text_field: str = "News"
    dataset_split: str = "test"

    # Headlines-specific fields
    attributes: List[str] = field(
        default_factory=lambda: [
            "Price_or_Not",
            "Direction_Up",
            "Direction_Down",
            "Direction_Constant",
            "Past_Price",
            "Future_Price",
            "Past_News",
        ]
    )

    # Attribute mapping from dataset fields to output fields
    dataset_fields: List[str] = field(
        default_factory=lambda: [
            "PriceOrNot",
            "DirectionUp",
            "DirectionDown",
            "DirectionConstant",
            "PastPrice",
            "FuturePrice",
            "PastNews",
        ]
    )

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "headlines"
        super().__post_init__()


@flame_task("headlines")
class HeadlinesTask(FLAMETask):
    """Headlines multi-attribute news classification task.

    This task analyzes news headlines for 7 binary attributes:
    - Price_or_Not: Does the news item talk about price
    - Direction_Up: Does it talk about price going up
    - Direction_Down: Does it talk about price going down
    - Direction_Constant: Does it talk about price remaining constant
    - Past_Price: Does it talk about an event in the past
    - Future_Price: Does it talk about an event in the future
    - Past_News: Does it talk about a general event (apart from prices) in the past

    Features:
    - Uses exact FLAME prompt for consistency
    - Robust multi-strategy JSON extraction
    - FLAME-compatible column names and output format
    - Support for 7 binary attribute scoring
    """

    def __init__(self, config: Optional[HeadlinesConfig] = None):
        """Initialize Headlines task."""
        if config is None:
            config = HeadlinesConfig(name="headlines")
        elif not isinstance(config, HeadlinesConfig):
            headlines_config = HeadlinesConfig(**config.__dict__)
            config = headlines_config

        super().__init__(config)
        self.config: HeadlinesConfig = config

        logger.info("Initialized Headlines task with 7 binary attributes")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for Headlines classification using exact FLAME prompt."""
        format = format or self.config.prompt_format

        # Extract news headline from sample
        sentence = sample.get(self.config.text_field, "")
        if not sentence:
            sentence = sample.get("text", sample.get("content", ""))

        if format == PromptFormat.ZERO_SHOT:
            # Use exact FLAME prompt
            prompt = f"""Discard all the previous instructions. Behave like you are an expert at analyzing headlines.
Give a score of 0 for each of the following attributes if the news headline does not contain the following information or 1 if it does.
Price or Not: Does the news item talk about price or not.
Direction Up: Does the news headline talk about price going up or not?
Direction Down: Does the news headline talk about price going down or not?
Direction Constant: Does the news headline talk about price remaining constant or not?
Past Price: Does the news headline talk about an event in the past?
Future Price: Does the news headline talk about an event in the future?
Past News: Does the news headline talk about a general event (apart from prices) in the past?
The news headline is:
{sentence}"""

        elif format == PromptFormat.FEW_SHOT:
            # Create few-shot prompt with examples
            prompt = f"""Analyze news headlines for multiple attributes. Give a score of 0 or 1 for each attribute.

Example 1:
Headline: "Apple stock surged 15% after strong earnings report yesterday"
Price or Not: 1 (talks about stock price)
Direction Up: 1 (surged = going up)
Direction Down: 0 (not going down)
Direction Constant: 0 (not constant)
Past Price: 1 (yesterday = past event)
Future Price: 0 (not about future)
Past News: 1 (earnings report was past event)

Example 2:
Headline: "Company announces new product launch next quarter"
Price or Not: 0 (no price mentioned)
Direction Up: 0 (no price direction)
Direction Down: 0 (no price direction)
Direction Constant: 0 (no price direction)
Past Price: 0 (not past price event)
Future Price: 0 (not price-related future event)
Past News: 0 (announcement about future, not past event)

Now analyze this headline:
Headline: {sentence}
"""

        else:
            # Default to zero-shot
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract 7 binary attributes using multi-strategy approach."""
        self._stats["responses_extracted"] += 1

        if not raw_response:
            return None

        # Clean the response
        response = raw_response.strip()

        # Strategy 1: JSON extraction
        extracted = self._extract_json_format(response)
        if extracted is not None:
            return extracted

        # Strategy 2: Structured text parsing
        extracted = self._extract_structured_text(response)
        if extracted is not None:
            return extracted

        # Strategy 3: Line-by-line attribute search
        extracted = self._extract_line_by_line(response)
        if extracted is not None:
            return extracted

        # Strategy 4: Keyword-based extraction
        extracted = self._extract_keyword_based(response)
        if extracted is not None:
            return extracted

        # Strategy 5: Pattern matching for lists/arrays
        extracted = self._extract_list_pattern(response)
        if extracted is not None:
            return extracted

        logger.debug(
            f"Failed to extract attributes from response: {raw_response[:100]}..."
        )
        self._stats["extraction_failures"] += 1
        return None

    def _extract_json_format(self, response: str) -> Optional[List[int]]:
        """Extract from JSON-like format."""
        try:
            # Look for JSON objects
            json_patterns = [
                r"\{[^{}]*\}",  # Simple JSON object
                r"\{[^{}]*\{[^{}]*\}[^{}]*\}",  # Nested JSON
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        result = []
                        for attr in self.config.attributes:
                            value = data.get(attr, data.get(attr.replace("_", " "), 0))
                            result.append(self._normalize_binary(value))

                        if any(x != 0 for x in result):  # At least one non-zero value
                            return result
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug(f"JSON extraction failed: {e}")

        return None

    def _extract_structured_text(self, response: str) -> Optional[List[int]]:
        """Extract from structured text format."""
        result = [0] * 7
        found_any = False

        # Look for "Attribute: Value" patterns
        patterns = [
            (r"Price\s*(?:or\s*)?Not\s*:?\s*([01])", 0),
            (r"Direction\s*Up\s*:?\s*([01])", 1),
            (r"Direction\s*Down\s*:?\s*([01])", 2),
            (r"Direction\s*Constant\s*:?\s*([01])", 3),
            (r"Past\s*Price\s*:?\s*([01])", 4),
            (r"Future\s*Price\s*:?\s*([01])", 5),
            (r"Past\s*News\s*:?\s*([01])", 6),
        ]

        for pattern, idx in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                result[idx] = self._normalize_binary(matches[0])
                found_any = True

        return result if found_any else None

    def _extract_line_by_line(self, response: str) -> Optional[List[int]]:
        """Extract by analyzing each line for attribute information."""
        lines = response.split("\n")
        result = [0] * 7
        found_any = False

        attribute_keywords = [
            (["price or not", "price_or_not", "priceornot"], 0),
            (
                ["direction up", "direction_up", "directionup", "going up", "price up"],
                1,
            ),
            (
                [
                    "direction down",
                    "direction_down",
                    "directiondown",
                    "going down",
                    "price down",
                ],
                2,
            ),
            (
                [
                    "direction constant",
                    "direction_constant",
                    "directionconstant",
                    "constant price",
                ],
                3,
            ),
            (["past price", "past_price", "pastprice", "price past"], 4),
            (["future price", "future_price", "futureprice", "price future"], 5),
            (["past news", "past_news", "pastnews", "news past", "general past"], 6),
        ]

        for line in lines:
            line_lower = line.lower().strip()

            # Look for binary values in each line
            binary_values = re.findall(r"\b[01]\b", line)

            for keywords, idx in attribute_keywords:
                if any(kw in line_lower for kw in keywords) and binary_values:
                    result[idx] = int(binary_values[0])
                    found_any = True
                    break

        return result if found_any else None

    def _extract_keyword_based(self, response: str) -> Optional[List[int]]:
        """Extract based on keyword presence and context."""
        response_lower = response.lower()
        result = [0] * 7

        # Simple heuristics based on content
        # Price or Not
        if any(
            word in response_lower
            for word in ["price", "cost", "dollar", "$", "expensive", "cheap"]
        ):
            result[0] = 1

        # Direction indicators
        if any(
            word in response_lower
            for word in ["up", "rise", "increase", "surge", "gain", "higher"]
        ):
            result[1] = 1
        if any(
            word in response_lower
            for word in ["down", "fall", "decrease", "drop", "decline", "lower"]
        ):
            result[2] = 1
        if any(
            word in response_lower
            for word in ["stable", "unchanged", "constant", "steady", "flat"]
        ):
            result[3] = 1

        # Temporal indicators
        past_words = [
            "yesterday",
            "last",
            "previous",
            "ago",
            "earlier",
            "was",
            "reported",
        ]
        future_words = [
            "will",
            "next",
            "future",
            "expected",
            "forecast",
            "projected",
            "tomorrow",
        ]

        if any(word in response_lower for word in past_words):
            if result[0] == 1:  # If price-related and past
                result[4] = 1
            result[6] = 1  # Past news

        if any(word in response_lower for word in future_words) and result[0] == 1:
            result[5] = 1

        return result if any(result) else None

    def _extract_list_pattern(self, response: str) -> Optional[List[int]]:
        """Extract from list/array-like patterns."""
        # Look for sequences of 7 binary numbers
        binary_sequences = re.findall(
            r"(?:\[([01](?:\s*,\s*[01]){6})\])|(?:([01](?:\s+[01]){6}))", response
        )

        for seq_tuple in binary_sequences:
            seq = seq_tuple[0] or seq_tuple[1]
            if seq:
                try:
                    # Parse the sequence
                    numbers = [int(x.strip()) for x in re.findall(r"[01]", seq)]
                    if len(numbers) == 7:
                        return numbers
                except ValueError:
                    continue

        return None

    def _normalize_binary(self, value) -> int:
        """Normalize various representations to 0 or 1."""
        if isinstance(value, (int, float)):
            return 1 if value > 0 else 0
        elif isinstance(value, str):
            value_clean = str(value).strip().lower()
            if value_clean in ["1", "true", "yes", "y"]:
                return 1
            elif value_clean in ["0", "false", "no", "n"]:
                return 0
        return 0

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth labels from sample."""
        # Extract ground truth attributes from dataset fields
        result = []
        for field_name in self.config.dataset_fields:
            value = sample.get(field_name, 0)
            result.append(self._normalize_binary(value))
        return result

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
            news_text = sample.get(self.config.text_field, "")
            ground_truth = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields
                "news": news_text,  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field
                "actual_labels": ground_truth,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # FLAME primary field
                # Headlines-specific fields
                "headline": news_text,
                "attributes": extracted,
                "confidence": 1.0 if extracted else 0.0,
                # Standard BenchForge fields
                "prompt": prompt,
                "input": news_text,
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
            f"Headlines extraction results: {successful}/{total} successful extractions ({success_rate:.1f}%)"
        )

        # Log attribute distribution if successful extractions exist
        if successful > 0:
            extracted_attrs = [
                attr for attr in df["extracted_labels"] if attr is not None
            ]
            if extracted_attrs:
                attr_sums = [sum(attr[i] for attr in extracted_attrs) for i in range(7)]
                attr_names = self.config.attributes
                attr_dist = {name: count for name, count in zip(attr_names, attr_sums)}
                logger.info(f"Attribute distribution: {attr_dist}")

        return df
