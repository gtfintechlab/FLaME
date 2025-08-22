"""Professional response extraction system for LLM outputs.

This module provides sophisticated extraction strategies for parsing
and structuring LLM responses across different task types.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Pattern

logger = logging.getLogger(__name__)


class ExtractionStrategy(Enum):
    """Response extraction strategies."""

    KEYWORD = "keyword"  # Extract based on keywords
    REGEX = "regex"  # Extract using regex patterns
    JSON = "json"  # Extract from JSON responses
    FIRST_LINE = "first_line"  # Extract first line only
    LAST_LINE = "last_line"  # Extract last line only
    FUNCTION = "function"  # Custom extraction function
    CHAIN_OF_THOUGHT = "cot"  # Extract from chain-of-thought
    STRUCTURED = "structured"  # Extract structured data
    FUZZY = "fuzzy"  # Fuzzy matching for labels
    CONFIDENCE = "confidence"  # Extract with confidence scores
    LLM_BASED = "llm_based"  # Use LLM for extraction (FLAME parity)


@dataclass
class ExtractionResult:
    """Result of response extraction.

    Attributes:
        value: Extracted value
        confidence: Confidence score (0-1)
        strategy: Strategy used for extraction
        raw_response: Original response
        metadata: Additional extraction metadata
    """

    value: Any
    confidence: float = 1.0
    strategy: ExtractionStrategy = ExtractionStrategy.KEYWORD
    raw_response: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResponseExtractor:
    """Professional response extraction with multiple strategies.

    This extractor provides:
    - Multiple extraction strategies
    - Confidence scoring
    - Error recovery
    - Pattern caching
    - Custom extractors
    """

    def __init__(
        self,
        default_strategy: ExtractionStrategy = ExtractionStrategy.KEYWORD,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        normalize_unicode: bool = True,
        cache_patterns: bool = True,
    ):
        """Initialize response extractor.

        Args:
            default_strategy: Default extraction strategy
            case_sensitive: Whether extraction is case-sensitive
            strip_whitespace: Whether to strip whitespace
            normalize_unicode: Whether to normalize unicode
            cache_patterns: Whether to cache compiled patterns
        """
        self.default_strategy = default_strategy
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.normalize_unicode = normalize_unicode
        self.cache_patterns = cache_patterns

        # Pattern cache
        self._pattern_cache = {}

        # Custom extractors
        self._custom_extractors = {}

        # Statistics
        self._stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "strategy_usage": {},
        }

        # Common patterns
        self._init_common_patterns()

    def _init_common_patterns(self):
        """Initialize common extraction patterns."""
        self._common_patterns = {
            # Classification patterns
            "label_colon": re.compile(
                r"(?:answer|label|classification|sentiment|category)\s*:\s*(\w+)",
                re.IGNORECASE,
            ),
            "label_quoted": re.compile(
                r'(?:answer|label|classification|sentiment|category)\s*:\s*["\']([^"\']+)["\']',
                re.IGNORECASE,
            ),
            "label_parentheses": re.compile(r"\(([A-Z][A-Z0-9_]*)\)", re.IGNORECASE),
            # Numeric patterns
            "number": re.compile(r"-?\d+\.?\d*"),
            "percentage": re.compile(r"(\d+\.?\d*)\s*%"),
            "currency": re.compile(r"\$\s*(\d+\.?\d*)"),
            # JSON patterns
            "json_object": re.compile(r"\{[^}]+\}"),
            "json_array": re.compile(r"\[[^\]]+\]"),
            # List patterns
            "bullet_list": re.compile(r"^\s*[-*•]\s+(.+)$", re.MULTILINE),
            "numbered_list": re.compile(r"^\s*\d+[.)]\s+(.+)$", re.MULTILINE),
            # Answer patterns
            "final_answer": re.compile(
                r"(?:final answer|answer|result|conclusion)\s*:\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        }

    def extract(
        self, response: str, strategy: Optional[ExtractionStrategy] = None, **kwargs
    ) -> ExtractionResult:
        """Extract structured data from response.

        Args:
            response: Raw LLM response
            strategy: Extraction strategy to use
            **kwargs: Strategy-specific arguments

        Returns:
            ExtractionResult with extracted value
        """
        if not response:
            return ExtractionResult(value=None, confidence=0.0, raw_response=response)

        # Preprocess response
        processed = self._preprocess(response)

        # Use default strategy if not specified
        strategy = strategy or self.default_strategy

        # Update statistics
        self._stats["total_extractions"] += 1
        self._stats["strategy_usage"][strategy.value] = (
            self._stats["strategy_usage"].get(strategy.value, 0) + 1
        )

        try:
            # Route to appropriate extraction method
            if strategy == ExtractionStrategy.KEYWORD:
                result = self._extract_keyword(processed, **kwargs)
            elif strategy == ExtractionStrategy.REGEX:
                result = self._extract_regex(processed, **kwargs)
            elif strategy == ExtractionStrategy.JSON:
                result = self._extract_json(processed, **kwargs)
            elif strategy == ExtractionStrategy.FIRST_LINE:
                result = self._extract_first_line(processed, **kwargs)
            elif strategy == ExtractionStrategy.LAST_LINE:
                result = self._extract_last_line(processed, **kwargs)
            elif strategy == ExtractionStrategy.FUNCTION:
                result = self._extract_function(processed, **kwargs)
            elif strategy == ExtractionStrategy.CHAIN_OF_THOUGHT:
                result = self._extract_cot(processed, **kwargs)
            elif strategy == ExtractionStrategy.STRUCTURED:
                result = self._extract_structured(processed, **kwargs)
            elif strategy == ExtractionStrategy.FUZZY:
                result = self._extract_fuzzy(processed, **kwargs)
            elif strategy == ExtractionStrategy.CONFIDENCE:
                result = self._extract_with_confidence(processed, **kwargs)
            elif strategy == ExtractionStrategy.LLM_BASED:
                result = self._extract_llm_based(processed, **kwargs)
            else:
                raise ValueError(f"Unknown extraction strategy: {strategy}")

            result.raw_response = response
            result.strategy = strategy

            self._stats["successful_extractions"] += 1

            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            self._stats["failed_extractions"] += 1

            return ExtractionResult(
                value=None,
                confidence=0.0,
                strategy=strategy,
                raw_response=response,
                metadata={"error": str(e)},
            )

    def extract_label(
        self,
        response: str,
        valid_labels: List[str],
        strategy: Optional[ExtractionStrategy] = None,
        fuzzy_threshold: float = 0.8,
    ) -> str:
        """Extract classification label from response.

        Args:
            response: Raw LLM response
            valid_labels: List of valid label values
            strategy: Extraction strategy
            fuzzy_threshold: Threshold for fuzzy matching

        Returns:
            Extracted label or None
        """
        if not response or not valid_labels:
            return None

        # Try multiple strategies in order of reliability
        strategies = [
            ExtractionStrategy.KEYWORD,
            ExtractionStrategy.REGEX,
            ExtractionStrategy.FUZZY,
        ]

        if strategy:
            strategies = [strategy] + [s for s in strategies if s != strategy]

        for strat in strategies:
            if strat == ExtractionStrategy.KEYWORD:
                result = self._extract_label_keyword(response, valid_labels)
            elif strat == ExtractionStrategy.REGEX:
                result = self._extract_label_regex(response, valid_labels)
            elif strat == ExtractionStrategy.FUZZY:
                result = self._extract_label_fuzzy(
                    response, valid_labels, fuzzy_threshold
                )
            else:
                continue

            if result:
                return result

        # Fallback: check if any valid label appears in response
        response_lower = response.lower()
        for label in valid_labels:
            if label.lower() in response_lower:
                return label

        return None

    def _preprocess(self, text: str) -> str:
        """Preprocess text for extraction."""
        if not text:
            return ""

        # Strip whitespace if requested
        if self.strip_whitespace:
            text = text.strip()

        # Normalize unicode if requested
        if self.normalize_unicode:
            import unicodedata

            text = unicodedata.normalize("NFKD", text)

        return text

    def _extract_keyword(
        self, text: str, keywords: Optional[List[str]] = None, **kwargs
    ) -> ExtractionResult:
        """Extract based on keywords."""
        if not keywords:
            # Try common patterns
            for pattern_name, pattern in self._common_patterns.items():
                if "label" in pattern_name or "answer" in pattern_name:
                    match = pattern.search(text)
                    if match:
                        return ExtractionResult(
                            value=match.group(1),
                            confidence=0.9,
                            metadata={"pattern": pattern_name},
                        )
        else:
            # Look for specific keywords
            text_to_search = text if self.case_sensitive else text.lower()

            for keyword in keywords:
                keyword_to_search = keyword if self.case_sensitive else keyword.lower()

                if keyword_to_search in text_to_search:
                    return ExtractionResult(
                        value=keyword,
                        confidence=1.0,
                        metadata={"matched_keyword": keyword},
                    )

        return ExtractionResult(value=None, confidence=0.0)

    def _extract_regex(
        self, text: str, pattern: Union[str, Pattern], **kwargs
    ) -> ExtractionResult:
        """Extract using regex pattern."""
        if isinstance(pattern, str):
            # Compile and cache pattern
            if self.cache_patterns and pattern in self._pattern_cache:
                compiled = self._pattern_cache[pattern]
            else:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                compiled = re.compile(pattern, flags)
                if self.cache_patterns:
                    self._pattern_cache[pattern] = compiled
        else:
            compiled = pattern

        match = compiled.search(text)
        if match:
            # Extract first group if exists, otherwise whole match
            value = match.group(1) if match.groups() else match.group(0)
            return ExtractionResult(
                value=value, confidence=0.95, metadata={"match_span": match.span()}
            )

        return ExtractionResult(value=None, confidence=0.0)

    def _extract_json(
        self, text: str, key: Optional[str] = None, **kwargs
    ) -> ExtractionResult:
        """Extract from JSON response."""
        # Find JSON in text
        json_match = self._common_patterns["json_object"].search(text)
        if not json_match:
            json_match = self._common_patterns["json_array"].search(text)

        if not json_match:
            # Try to parse entire text as JSON
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return ExtractionResult(value=None, confidence=0.0)
        else:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return ExtractionResult(value=None, confidence=0.0)

        # Extract specific key if provided
        if key and isinstance(data, dict):
            value = data.get(key)
        else:
            value = data

        return ExtractionResult(
            value=value,
            confidence=0.95 if value is not None else 0.0,
            metadata={"json_found": True},
        )

    def _extract_first_line(self, text: str, **kwargs) -> ExtractionResult:
        """Extract first non-empty line."""
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line:
                return ExtractionResult(
                    value=line, confidence=0.8, metadata={"line_number": 1}
                )

        return ExtractionResult(value=None, confidence=0.0)

    def _extract_last_line(self, text: str, **kwargs) -> ExtractionResult:
        """Extract last non-empty line."""
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line:
                return ExtractionResult(
                    value=line, confidence=0.8, metadata={"line_number": len(lines)}
                )

        return ExtractionResult(value=None, confidence=0.0)

    def _extract_function(
        self, text: str, func: Callable, **kwargs
    ) -> ExtractionResult:
        """Extract using custom function."""
        if not func:
            raise ValueError("Extraction function not provided")

        try:
            value = func(text, **kwargs)
            return ExtractionResult(
                value=value,
                confidence=0.9 if value is not None else 0.0,
                metadata={"function": func.__name__},
            )
        except Exception as e:
            logger.error(f"Custom extraction function failed: {e}")
            return ExtractionResult(value=None, confidence=0.0)

    def _extract_cot(self, text: str, **kwargs) -> ExtractionResult:
        """Extract from chain-of-thought response."""
        # Look for final answer patterns
        final_answer = self._common_patterns["final_answer"].search(text)
        if final_answer:
            return ExtractionResult(
                value=final_answer.group(1).strip(),
                confidence=0.95,
                metadata={"cot_detected": True},
            )

        # Fallback to last paragraph
        paragraphs = text.strip().split("\n\n")
        if paragraphs:
            last_para = paragraphs[-1].strip()
            # Check if it looks like an answer
            if len(last_para) < 200:  # Reasonable answer length
                return ExtractionResult(
                    value=last_para,
                    confidence=0.7,
                    metadata={"fallback": "last_paragraph"},
                )

        return ExtractionResult(value=None, confidence=0.0)

    def _extract_structured(
        self, text: str, schema: Optional[Dict] = None, **kwargs
    ) -> ExtractionResult:
        """Extract structured data based on schema."""
        if not schema:
            # Try to extract any structured format
            # Check for JSON
            json_result = self._extract_json(text)
            if json_result.value:
                return json_result

            # Check for lists
            bullet_items = self._common_patterns["bullet_list"].findall(text)
            if bullet_items:
                return ExtractionResult(
                    value=bullet_items,
                    confidence=0.85,
                    metadata={"format": "bullet_list"},
                )

            numbered_items = self._common_patterns["numbered_list"].findall(text)
            if numbered_items:
                return ExtractionResult(
                    value=numbered_items,
                    confidence=0.85,
                    metadata={"format": "numbered_list"},
                )
        else:
            # Extract based on schema
            extracted = {}
            confidence_sum = 0

            for field, field_schema in schema.items():
                field_pattern = field_schema.get("pattern")
                field_type = field_schema.get("type", "string")

                if field_pattern:
                    match = re.search(field_pattern, text, re.IGNORECASE)
                    if match:
                        value = match.group(1) if match.groups() else match.group(0)

                        # Type conversion
                        if field_type == "int":
                            value = int(value)
                        elif field_type == "float":
                            value = float(value)
                        elif field_type == "bool":
                            value = value.lower() in ("true", "yes", "1")

                        extracted[field] = value
                        confidence_sum += 0.9
                    else:
                        extracted[field] = None

            avg_confidence = confidence_sum / len(schema) if schema else 0

            return ExtractionResult(
                value=extracted,
                confidence=avg_confidence,
                metadata={"schema_fields": len(schema)},
            )

        return ExtractionResult(value=None, confidence=0.0)

    def _extract_fuzzy(
        self, text: str, options: List[str], threshold: float = 0.8, **kwargs
    ) -> ExtractionResult:
        """Extract using fuzzy matching."""
        try:
            from difflib import SequenceMatcher
        except ImportError:
            logger.warning("difflib not available for fuzzy matching")
            return ExtractionResult(value=None, confidence=0.0)

        if not options:
            return ExtractionResult(value=None, confidence=0.0)

        text_lower = text.lower()
        best_match = None
        best_score = 0.0

        for option in options:
            option_lower = option.lower()

            # Check exact match first
            if option_lower in text_lower:
                return ExtractionResult(
                    value=option, confidence=1.0, metadata={"fuzzy_match": False}
                )

            # Fuzzy match on tokens
            tokens = text_lower.split()
            for token in tokens:
                score = SequenceMatcher(None, token, option_lower).ratio()
                if score > best_score:
                    best_score = score
                    best_match = option

        if best_score >= threshold:
            return ExtractionResult(
                value=best_match,
                confidence=best_score,
                metadata={"fuzzy_match": True, "match_score": best_score},
            )

        return ExtractionResult(value=None, confidence=0.0)

    def _extract_with_confidence(self, text: str, **kwargs) -> ExtractionResult:
        """Extract value with confidence score."""
        # Look for patterns like "Answer: X (confidence: 0.9)"
        confidence_pattern = re.compile(
            r"(?:answer|prediction|result)\s*:\s*([^(]+)\s*\((?:confidence|probability|certainty)\s*:\s*(\d+\.?\d*)\)",
            re.IGNORECASE,
        )

        match = confidence_pattern.search(text)
        if match:
            value = match.group(1).strip()
            confidence = float(match.group(2))

            return ExtractionResult(
                value=value,
                confidence=min(confidence, 1.0),
                metadata={"confidence_explicit": True},
            )

        # Fallback to other extraction with estimated confidence
        result = self._extract_keyword(text, **kwargs)
        if not result.value:
            result = self._extract_regex(
                text, pattern=self._common_patterns["final_answer"]
            )

        return result

    def _extract_label_keyword(
        self, text: str, valid_labels: List[str]
    ) -> Optional[str]:
        """Extract label using keyword matching."""
        text_to_search = text if self.case_sensitive else text.lower()

        for label in valid_labels:
            label_to_search = label if self.case_sensitive else label.lower()

            # Check exact match with word boundaries
            pattern = r"\b" + re.escape(label_to_search) + r"\b"
            if re.search(pattern, text_to_search):
                return label

        return None

    def _extract_label_regex(self, text: str, valid_labels: List[str]) -> Optional[str]:
        """Extract label using regex patterns."""
        # Try common label patterns
        for pattern in [
            self._common_patterns["label_colon"],
            self._common_patterns["label_quoted"],
            self._common_patterns["label_parentheses"],
        ]:
            match = pattern.search(text)
            if match:
                extracted = match.group(1)
                # Check if extracted value is valid
                extracted_lower = extracted.lower()
                for label in valid_labels:
                    if label.lower() == extracted_lower:
                        return label

        return None

    def _extract_label_fuzzy(
        self, text: str, valid_labels: List[str], threshold: float
    ) -> Optional[str]:
        """Extract label using fuzzy matching."""
        result = self._extract_fuzzy(text, valid_labels, threshold)
        return result.value if result.confidence >= threshold else None

    def _extract_llm_based(
        self,
        text: str,
        llm_client=None,
        prompt_template: Optional[str] = None,
        valid_labels: Optional[List[str]] = None,
        **kwargs,
    ) -> ExtractionResult:
        """Extract using LLM with extraction prompt (FLAME parity).

        This method implements FLAME's approach of using a language model
        to extract structured information from messy LLM responses.

        Args:
            text: Raw LLM response to extract from
            llm_client: LLM client for making extraction calls
            prompt_template: Template for extraction prompt
            valid_labels: List of valid labels for classification tasks
            **kwargs: Additional arguments for LLM call

        Returns:
            ExtractionResult with extracted value
        """
        if not llm_client:
            logger.warning("LLM client not provided for LLM-based extraction")
            # Fallback to fuzzy matching if we have valid labels
            if valid_labels:
                return self._extract_fuzzy(text, valid_labels, threshold=0.8)
            return ExtractionResult(
                value=None, confidence=0.0, metadata={"error": "No LLM client provided"}
            )

        # Create extraction prompt
        if not prompt_template:
            if valid_labels:
                # Default classification extraction prompt
                labels_str = ", ".join([f"'{label}'" for label in valid_labels])
                prompt_template = f"""Extract the classification label from the following LLM response. 
                The label should be one of the following: {labels_str}.
                
                Here is the LLM response to analyze:
                "{{llm_response}}"
                
                Provide only the label that best matches the response, exactly as it appears in the list.
                Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
            else:
                # Generic extraction prompt
                prompt_template = """Extract the main answer or value from the following LLM response.
                
                LLM Response: "{llm_response}"
                
                Provide only the extracted value without any additional text or explanation."""

        # Format the prompt with the text
        extraction_prompt = prompt_template.format(llm_response=text)

        try:
            # Make LLM call for extraction
            # Note: The actual call interface will depend on the llm_client implementation
            if hasattr(llm_client, "complete"):
                # For clients with complete method
                extracted_response = llm_client.complete(
                    prompt=extraction_prompt,
                    max_tokens=kwargs.get("max_tokens", 50),
                    temperature=kwargs.get(
                        "temperature", 0.0
                    ),  # Low temperature for extraction
                    **kwargs,
                )
            elif hasattr(llm_client, "chat"):
                # For chat-based clients
                messages = [{"role": "user", "content": extraction_prompt}]
                extracted_response = llm_client.chat(
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 50),
                    temperature=kwargs.get("temperature", 0.0),
                    **kwargs,
                )
            else:
                # Generic call
                extracted_response = llm_client(extraction_prompt, **kwargs)

            # Extract the actual text from the response
            if hasattr(extracted_response, "choices"):
                # OpenAI-style response
                extracted_text = extracted_response.choices[0].message.content
            elif isinstance(extracted_response, dict):
                # Dictionary response
                extracted_text = extracted_response.get(
                    "content", ""
                ) or extracted_response.get("text", "")
            else:
                # String response
                extracted_text = str(extracted_response)

            # Clean up the extracted text
            extracted_text = extracted_text.strip()

            # If we have valid labels, validate the extraction
            if valid_labels:
                # Try to match with valid labels
                extracted_upper = extracted_text.upper()
                for label in valid_labels:
                    if (
                        label.upper() == extracted_upper
                        or label.upper() in extracted_upper
                    ):
                        return ExtractionResult(
                            value=label,
                            confidence=0.95,
                            metadata={
                                "llm_extracted": True,
                                "extraction_method": "llm_based",
                            },
                        )

                # If no exact match, try fuzzy matching on the extracted text
                fuzzy_result = self._extract_fuzzy(
                    extracted_text, valid_labels, threshold=0.8
                )
                if fuzzy_result.value:
                    fuzzy_result.metadata["llm_extracted"] = True
                    fuzzy_result.metadata["extraction_method"] = "llm_based_fuzzy"
                    return fuzzy_result

                # No valid label found
                return ExtractionResult(
                    value=None,
                    confidence=0.0,
                    metadata={
                        "llm_extracted": True,
                        "extracted_text": extracted_text,
                        "error": "Extracted text does not match valid labels",
                    },
                )
            else:
                # Return the extracted text as-is for non-classification tasks
                return ExtractionResult(
                    value=extracted_text,
                    confidence=0.9,
                    metadata={"llm_extracted": True, "extraction_method": "llm_based"},
                )

        except Exception as e:
            logger.error(f"LLM-based extraction failed: {e}")
            # Fallback to other strategies
            if valid_labels:
                return self._extract_fuzzy(text, valid_labels, threshold=0.8)
            return ExtractionResult(
                value=None,
                confidence=0.0,
                metadata={"error": f"LLM extraction failed: {str(e)}"},
            )

    def register_extractor(self, name: str, func: Callable):
        """Register a custom extraction function.

        Args:
            name: Name for the custom extractor
            func: Extraction function
        """
        self._custom_extractors[name] = func
        logger.info(f"Registered custom extractor: {name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        stats = self._stats.copy()

        if stats["total_extractions"] > 0:
            stats["success_rate"] = (
                stats["successful_extractions"] / stats["total_extractions"]
            )
            stats["failure_rate"] = (
                stats["failed_extractions"] / stats["total_extractions"]
            )

        return stats

    def reset_stats(self):
        """Reset extraction statistics."""
        self._stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "strategy_usage": {},
        }


# Convenience function
def get_extractor(
    strategy: ExtractionStrategy = ExtractionStrategy.KEYWORD, **kwargs
) -> ResponseExtractor:
    """Get a configured response extractor.

    Args:
        strategy: Default extraction strategy
        **kwargs: Additional configuration

    Returns:
        Configured ResponseExtractor
    """
    return ResponseExtractor(default_strategy=strategy, **kwargs)
