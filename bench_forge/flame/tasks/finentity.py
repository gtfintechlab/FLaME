"""FinEntity task with entity+sentiment extraction for BenchForge.

This implementation matches the native FLAME FinEntity task:
- Extracts company/organization entities from financial text
- Classifies sentiment (Positive/Negative/Neutral) for each entity  
- Provides character boundaries (start/end indices) for each entity
"""

import logging
import re
import json
import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat
from bench_forge.prompts.extractor import ExtractionStrategy

logger = logging.getLogger(__name__)


@dataclass
class FinEntityConfig(FLAMEConfig):
    """Configuration for FinEntity task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/finentity"
    dataset_name: str = "5768"
    text_field: str = "content"
    label_field: str = "annotations"
    dataset_split: str = "test"

    # FinEntity-specific fields
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.JSON
    financial_domain: str = "entity_sentiment_extraction"
    
    # Sentiment labels (not entity types!)
    valid_labels: List[str] = field(default_factory=lambda: [
        "Positive",
        "Negative", 
        "Neutral"
    ])
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "finentity"
        super().__post_init__()


@flame_task("finentity")
class FinEntityTask(FLAMETask):
    """FinEntity task for company/organization entity + sentiment extraction.
    
    This matches the original FLAME implementation:
    - Identifies company/organization entities in financial text
    - Classifies sentiment for each entity (Positive/Negative/Neutral)
    - Provides exact character boundaries (start/end indices)
    - Returns JSON list format
    
    Input format:
    - content: Financial text paragraph
    - annotations: Ground truth entity list
    
    Output format:
    JSON list with structure:
    [{"value": "EntityName", "tag": "Sentiment", "label": "Sentiment", 
      "start": X, "end": Y}]
    """

    def __init__(self, config: Optional[FinEntityConfig] = None):
        """Initialize FinEntity task."""
        if config is None:
            config = FinEntityConfig(name="finentity")
        elif not isinstance(config, FinEntityConfig):
            finentity_config = FinEntityConfig(**config.__dict__)
            config = finentity_config

        super().__init__(config)
        self.config: FinEntityConfig = config

        logger.info("Initialized FinEntity task with entity+sentiment extraction")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for FinEntity extraction using EXACT FLAME prompt."""
        format = format or self.config.prompt_format

        # Extract sentence from sample
        sentence = sample.get("content", sample.get("sentence", ""))
        
        if not sentence:
            # Try other common field names
            sentence = sample.get("text", sample.get("paragraph", ""))

        # Use the EXACT prompt from FLAME's finentity_zeroshot_prompt
        prompt = f"""Discard all the previous instructions. Behave like you are an expert entity recognizer and sentiment classifier. Identify the entities which are companies or organizations from the following content and classify the sentiment of the corresponding entities into 'Neutral' 'Positive' or 'Negative' classes. Considering every paragraph as a String in Python, provide the entities with the start and end index to mark the boundaries of it including spaces and punctuation using zero-based indexing. In the output,
    Tag means sentiment; value means entity name. If no entity is found in the paragraph,
    the response should be empty. Only give the output, not python code. The output should be a list that looks like:
    [{{'end': 210,
   'label': 'Neutral',
   'start': 207,
   'tag': 'Neutral',
   'value': 'FAA'}},
  {{'end': 7, 'label': 'Neutral', 'start': 4, 'tag': 'Neutral', 'value': 'FAA'}},
  {{'end': 298,
   'label': 'Neutral',
   'start': 295,
   'tag': 'Neutral',
   'value': 'FAA'}},
  {{'end': 105,
   'label': 'Neutral',
   'start': 99,
   'tag': 'Neutral',
   'value': 'Boeing'}}]
   Do not repeat any JSON object in the list. Evey JSON object should be unique.
   The paragraph:
                {sentence}"""

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract entity list from response using BenchForge JSON extractor."""
        self._stats["responses_extracted"] += 1

        try:
            # Try BenchForge's JSON extractor first
            extraction_result = self.extractor.extract(
                raw_response,
                strategy=ExtractionStrategy.JSON
            )
            
            if extraction_result.value is not None:
                entities = extraction_result.value
                if isinstance(entities, list):
                    # Validate entity format
                    validated_entities = self._validate_entities(entities, sample)
                    if validated_entities:
                        return validated_entities
            
            # Fallback to custom extraction
            entities = self._extract_entity_list(raw_response)
            
            if entities is None:
                self._stats["extraction_failures"] += 1
                logger.debug(f"Failed to extract entities from: {raw_response[:100]}...")

            return entities

        except Exception as e:
            self._stats["extraction_failures"] += 1
            logger.error(f"Entity extraction error: {e}")
            return []

    def _extract_entity_list(self, response: str) -> Optional[List[Dict]]:
        """Extract entity list using multiple strategies."""
        if not response:
            return []

        response = response.strip()

        # Remove markdown code blocks
        if "```" in response:
            response = re.sub(r'```(?:json)?\s*\n?', '', response)
            response = re.sub(r'\n?```', '', response)

        # Strategy 1: Direct JSON parse
        try:
            if response.startswith('[') and response.endswith(']'):
                data = json.loads(response)
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            pass

        # Strategy 2: Python literal eval (handles single quotes)
        try:
            data = ast.literal_eval(response)
            if isinstance(data, list):
                return data
        except (ValueError, SyntaxError):
            pass

        # Strategy 3: Find JSON array in response
        json_pattern = r'\[\s*\{[^]]+\}\s*\]'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                # Clean up the matched string
                json_str = match.group(0)
                # Replace single quotes with double quotes for JSON
                json_str = json_str.replace("'", '"')
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                try:
                    # Try ast.literal_eval on original match
                    data = ast.literal_eval(match.group(0))
                    if isinstance(data, list):
                        return data
                except:
                    pass

        # Strategy 4: Check for empty indicators
        empty_indicators = ['[]', 'empty', 'no entities', 'none', 'not found']
        if any(indicator in response.lower() for indicator in empty_indicators):
            return []

        logger.warning(f"Could not extract entity list from response: {response[:200]}...")
        return []

    def _validate_entities(self, entities: List[Dict], sample: Optional[Dict] = None) -> List[Dict]:
        """Validate extracted entities."""
        if not entities:
            return []

        validated = []
        source_text = ""
        
        if sample:
            source_text = sample.get("content", sample.get("sentence", sample.get("text", "")))

        for entity in entities:
            if not isinstance(entity, dict):
                continue
                
            # Check required fields
            required_fields = ['value', 'tag', 'label']
            if not all(field in entity for field in required_fields):
                continue
                
            # Validate sentiment labels
            sentiment = entity.get('tag', '').strip()
            if sentiment not in self.config.valid_labels:
                # Try to map common variations
                sentiment_map = {
                    'positive': 'Positive',
                    'negative': 'Negative', 
                    'neutral': 'Neutral',
                    'pos': 'Positive',
                    'neg': 'Negative',
                    'neu': 'Neutral'
                }
                sentiment = sentiment_map.get(sentiment.lower(), sentiment)
                
                if sentiment not in self.config.valid_labels:
                    continue
                    
                # Update the entity with corrected sentiment
                entity['tag'] = sentiment
                entity['label'] = sentiment
            
            # Validate boundaries if present and source text available
            if 'start' in entity and 'end' in entity and source_text:
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                
                if isinstance(start, int) and isinstance(end, int) and start >= 0 and end <= len(source_text):
                    # Optionally validate that extracted text matches entity value
                    if start < end:
                        extracted_text = source_text[start:end]
                        entity_value = entity.get('value', '')
                        
                        # Allow some flexibility in matching
                        if entity_value.lower() in extracted_text.lower() or extracted_text.lower() in entity_value.lower():
                            validated.append(entity)
                        else:
                            # Still add but log the mismatch
                            logger.debug(f"Boundary mismatch: '{entity_value}' vs '{extracted_text}'")
                            validated.append(entity)
                    else:
                        # Invalid boundaries, skip start/end
                        entity_clean = {k: v for k, v in entity.items() if k not in ['start', 'end']}
                        validated.append(entity_clean)
                else:
                    # Invalid boundaries, skip start/end
                    entity_clean = {k: v for k, v in entity.items() if k not in ['start', 'end']}
                    validated.append(entity_clean)
            else:
                # No boundaries to validate
                validated.append(entity)

        return validated

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth entity list from sample."""
        # Try multiple field names for ground truth
        for field_name in ["annotations", "entities", "labels", self.config.label_field]:
            if field_name in sample:
                ground_truth = sample[field_name]
                
                # Handle different formats
                if isinstance(ground_truth, str):
                    try:
                        # Try to parse as JSON
                        return json.loads(ground_truth)
                    except json.JSONDecodeError:
                        try:
                            # Try literal eval
                            return ast.literal_eval(ground_truth)
                        except:
                            return ground_truth
                elif isinstance(ground_truth, list):
                    return ground_truth
                else:
                    return ground_truth
        
        return []

    def format_results(
        self,
        samples: List[Dict[str, Any]],
        prompts: List[str],
        raw_responses: List[Any],
        extracted_responses: List[Any],
    ) -> pd.DataFrame:
        """Format results with entity+sentiment extraction fields."""
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
            sentence = sample.get("content", sample.get("sentence", ""))
            ground_truth = self.get_ground_truth(sample)

            # Convert extracted entities to JSON string for compatibility
            extracted_json = json.dumps(extracted) if extracted else "[]"
            ground_truth_json = json.dumps(ground_truth) if isinstance(ground_truth, list) else str(ground_truth)

            result = {
                "index": i,
                # FLAME-compatible fields
                "sentences": sentence,  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field  
                "actual_labels": ground_truth_json,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted_json,  # FLAME primary field
                # FinEntity-specific fields
                "sentence": sentence,
                "content": sentence,
                "entities": extracted,
                "num_entities": len(extracted) if extracted else 0,
                "entity_names": [e.get('value', '') for e in extracted] if extracted else [],
                "sentiments": [e.get('tag', '') for e in extracted] if extracted else [],
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
        successful = df["extracted_labels"].apply(lambda x: len(json.loads(x)) > 0 if x != "[]" else False).sum()
        success_rate = (successful / total * 100) if total > 0 else 0
        
        avg_entities = df["num_entities"].mean()
        
        logger.info(
            f"FinEntity extraction results: {successful}/{total} samples with entities ({success_rate:.1f}%)"
        )
        logger.info(f"Average entities per sample: {avg_entities:.2f}")

        return df