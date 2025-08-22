"""FiNER task with financial named entity recognition for BenchForge."""

import json
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
class FiNERConfig(FLAMEConfig):
    """Configuration for FiNER task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/finer-ord-bio"
    text_field: str = "tokens"
    label_field: str = "tags"
    dataset_split: str = "test"

    # FiNER-specific fields
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.REGEX
    financial_domain: str = "named_entity_recognition"
    
    # NER tag set (BIO format for financial entities)
    valid_labels: List[str] = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "finer"
        
        # Define financial NER tags
        if self.valid_labels is None:
            self.valid_labels = [
                "O",  # Outside
                "B-PER", "I-PER",  # Person
                "B-ORG", "I-ORG",  # Organization
                "B-LOC", "I-LOC",  # Location
                "B-MISC", "I-MISC",  # Miscellaneous
                # Financial-specific entities
                "B-MONEY", "I-MONEY",  # Monetary amounts
                "B-PERCENT", "I-PERCENT",  # Percentages
                "B-DATE", "I-DATE",  # Dates
                "B-TIME", "I-TIME",  # Time expressions
            ]
        
        super().__post_init__()


@flame_task("finer")
class FiNERTask(FLAMETask):
    """FiNER task for financial named entity recognition.
    
    Features:
    - Financial entity recognition in context
    - BIO tagging scheme handling
    - Token-level entity boundary detection
    - Financial-specific entity types
    - Sequence labeling evaluation
    
    Input format:
    - tokens: List of word tokens
    - tags: List of BIO tags corresponding to tokens
    
    The task identifies and labels financial entities in text,
    including organizations, monetary amounts, percentages, and dates.
    """

    def __init__(self, config: Optional[FiNERConfig] = None):
        """Initialize FiNER task."""
        if config is None:
            config = FiNERConfig(name="finer")
        elif not isinstance(config, FiNERConfig):
            finer_config = FiNERConfig(**config.__dict__)
            config = finer_config

        super().__init__(config)
        self.config: FiNERConfig = config

        logger.info("Initialized FiNER task with financial entity recognition")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for FiNER with entity recognition instructions."""
        format = format or self.config.prompt_format

        # Extract tokens
        tokens = sample.get("tokens", [])
        if isinstance(tokens, str):
            tokens = tokens.split()
        
        sentence = " ".join(str(token) for token in tokens)

        if format == PromptFormat.ZERO_SHOT:
            prompt = f"""Identify and label financial entities in the following sentence using BIO tagging.

Entity Types:
- PER: Person names
- ORG: Organizations, companies
- LOC: Locations, places
- MISC: Miscellaneous entities
- MONEY: Monetary amounts ($100, dollars)
- PERCENT: Percentages (5%, percent)
- DATE: Dates (2021, January)
- TIME: Time expressions

BIO Format:
- B-TYPE: Beginning of entity type
- I-TYPE: Inside entity type  
- O: Outside any entity

Sentence: {sentence}

Provide the BIO tags for each token in order, separated by spaces.

Tags:"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""Label financial entities using BIO tagging format.

Example:
Sentence: Apple Inc. reported $5.2 billion revenue in Q3 2021.
Tags: B-ORG I-ORG O B-MONEY I-MONEY I-MONEY O O B-DATE I-DATE

Entity Types: PER, ORG, LOC, MISC, MONEY, PERCENT, DATE, TIME
BIO Format: B-TYPE (beginning), I-TYPE (inside), O (outside)

Sentence: {sentence}

Tags:"""

        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            prompt = f"""Let me identify financial entities step by step.

Sentence: {sentence}

Step 1: Identify potential entities
Step 2: Classify entity types
Step 3: Apply BIO tagging

Entity Types: PER, ORG, LOC, MISC, MONEY, PERCENT, DATE, TIME

Analysis:
Step 1 - Potential entities:
Step 2 - Entity classifications:
Step 3 - BIO tags:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract BIO tags from NER response."""
        self._stats["responses_extracted"] += 1

        try:
            extracted = self._extract_bio_tags(raw_response, sample)
            
            if extracted is None:
                self._stats["extraction_failures"] += 1
                logger.debug(f"Failed to extract BIO tags from: {raw_response[:100]}...")

            return extracted

        except Exception as e:
            self._stats["extraction_failures"] += 1
            logger.error(f"NER extraction error: {e}")
            return None

    def _extract_bio_tags(self, response: str, sample: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
        """Extract BIO tags using multiple strategies."""
        if not response:
            return None

        response = response.strip()
        
        # Get expected number of tokens
        expected_length = None
        if sample and "tokens" in sample:
            tokens = sample["tokens"]
            if isinstance(tokens, list):
                expected_length = len(tokens)
            elif isinstance(tokens, str):
                expected_length = len(tokens.split())

        # Strategy 1: Direct tag sequence extraction
        tag_patterns = [
            r'(?:Tags?:?\s*)((?:[BOI]-[A-Z]+|O)(?:\s+(?:[BOI]-[A-Z]+|O))*)',
            r'(?:Step 3[^:]*:?\s*)((?:[BOI]-[A-Z]+|O)(?:\s+(?:[BOI]-[A-Z]+|O))*)',
            r'^((?:[BOI]-[A-Z]+|O)(?:\s+(?:[BOI]-[A-Z]+|O))*)$',
            r'(?:BIO tags?:?\s*)((?:[BOI]-[A-Z]+|O)(?:\s+(?:[BOI]-[A-Z]+|O))*)',
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                tag_string = match.group(1).strip()
                tags = self._parse_tag_string(tag_string)
                if tags and self._validate_bio_tags(tags):
                    if expected_length is None or len(tags) == expected_length:
                        logger.debug(f"Extracted BIO tags: {tags[:10]}...")
                        return tags

        # Strategy 2: Extract from structured format
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(marker in line.lower() for marker in ['tags:', 'bio tags', 'step 3']):
                # Extract tags from this line
                tag_candidates = re.findall(r'[BOI]-[A-Z]+|O', line)
                if tag_candidates and self._validate_bio_tags(tag_candidates):
                    if expected_length is None or len(tag_candidates) == expected_length:
                        logger.debug(f"Extracted from line: {tag_candidates[:10]}...")
                        return tag_candidates

        # Strategy 3: Find the longest valid tag sequence
        all_tags = re.findall(r'[BOI]-[A-Z]+|O', response)
        if all_tags:
            # Try different subsets
            for start in range(len(all_tags)):
                for end in range(start + 1, len(all_tags) + 1):
                    candidate = all_tags[start:end]
                    if self._validate_bio_tags(candidate):
                        if expected_length is None or len(candidate) == expected_length:
                            logger.debug(f"Extracted subset: {candidate[:10]}...")
                            return candidate

        # Strategy 4: Fallback to all O tags if we know the length
        if expected_length:
            fallback_tags = ["O"] * expected_length
            logger.debug(f"Using fallback O tags, length: {expected_length}")
            return fallback_tags

        logger.warning(f"Could not extract valid BIO tags from response: {response[:100]}...")
        return None

    def _parse_tag_string(self, tag_string: str) -> List[str]:
        """Parse a string of BIO tags into a list."""
        # Handle both space and comma separated
        tags = re.split(r'[,\s]+', tag_string.strip())
        # Filter out empty strings and normalize
        tags = [tag.strip().upper() for tag in tags if tag.strip()]
        return tags

    def _validate_bio_tags(self, tags: List[str]) -> bool:
        """Validate that BIO tags follow proper format and constraints."""
        if not tags:
            return False
        
        valid_tag_pattern = re.compile(r'^[BOI]-[A-Z]+$|^O$')
        
        for tag in tags:
            if not valid_tag_pattern.match(tag):
                return False
        
        # Check BIO constraints (I- must follow B- or I- of same type)
        prev_type = None
        for tag in tags:
            if tag.startswith('I-'):
                entity_type = tag[2:]
                if prev_type != entity_type:
                    # I- tag without proper B- prefix
                    return False
            elif tag.startswith('B-'):
                prev_type = tag[2:]
            else:  # O tag
                prev_type = None
        
        return True

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth BIO tags from sample."""
        tags = sample.get("tags") or sample.get(self.config.label_field)
        
        # Handle different formats
        if isinstance(tags, str):
            try:
                # Try to parse as JSON
                tags = json.loads(tags)
            except:
                # Try to parse as space-separated
                tags = tags.split()
        
        return tags

    def format_results(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results with NER-specific evaluation fields."""
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

            # Get tokens and ground truth
            tokens = sample.get("tokens", [])
            if isinstance(tokens, str):
                tokens = tokens.split()
                
            ground_truth = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields (using expected column names)
                "sentences": tokens,  # FLAME expects this field name
                "llm_responses": response_text,  # FLAME expects this field name
                "actual_labels": ground_truth,  # FLAME expects this field name
                "complete_responses": complete_response,  # FLAME expects this field name
                "extracted_labels": extracted,  # FLAME expects this field name
                # NER-specific fields
                "tokens": tokens,
                "ground_truth_tags": ground_truth,
                "predicted_tags": extracted,
                "sentence_text": " ".join(str(token) for token in tokens),
                # Standard BenchForge fields
                "prompt": prompt,
                "input": " ".join(str(token) for token in tokens),
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
            f"FiNER extraction success rate: {successful}/{total} ({success_rate:.1f}%)"
        )

        return df