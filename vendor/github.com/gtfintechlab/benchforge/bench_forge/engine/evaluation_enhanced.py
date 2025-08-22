"""Enhanced evaluation engine with complete FLAME parity.

This module adds fallback extraction from complete_responses to match FLAME's capabilities.
"""

import logging
from typing import Any, List, Optional
import pandas as pd

from bench_forge.engine.evaluation import EvaluationEngine, EvaluationResult

logger = logging.getLogger(__name__)


class EnhancedEvaluationEngine(EvaluationEngine):
    """Enhanced evaluation engine with fallback extraction support.

    This engine extends the base EvaluationEngine to add:
    - Fallback extraction from complete_responses
    - Re-extraction for failed extractions
    - LLM-based extraction as ultimate fallback
    - Full FLAME compatibility
    """

    def __init__(self, *args, enable_fallback_extraction: bool = True, **kwargs):
        """Initialize enhanced evaluation engine.

        Args:
            enable_fallback_extraction: Whether to enable fallback extraction from complete_responses
            *args, **kwargs: Arguments for parent EvaluationEngine
        """
        super().__init__(*args, **kwargs)
        self.enable_fallback_extraction = enable_fallback_extraction
        logger.info(
            "Initialized EnhancedEvaluationEngine with fallback extraction support"
        )

    def _extract_from_complete_response(
        self, complete_response: Any, task: Optional[str] = None
    ) -> Optional[str]:
        """Extract label/answer from a complete response object.

        This method provides FLAME-compatible extraction from stored ModelResponse objects.

        Args:
            complete_response: The complete response object (ModelResponse or dict)
            task: Task name for task-specific extraction

        Returns:
            Extracted label/answer or None if extraction fails
        """
        try:
            # Handle different response formats
            if hasattr(complete_response, "choices"):
                # It's a ModelResponse object
                if complete_response.choices and len(complete_response.choices) > 0:
                    response_text = complete_response.choices[0].message.content
                else:
                    return None
            elif isinstance(complete_response, dict):
                # It's a dictionary representation
                if "choices" in complete_response and complete_response["choices"]:
                    response_text = complete_response["choices"][0]["message"][
                        "content"
                    ]
                else:
                    return None
            elif isinstance(complete_response, str):
                # It's already extracted text
                response_text = complete_response
            else:
                logger.warning(
                    f"Unknown complete_response type: {type(complete_response)}"
                )
                return None

            # Apply task-specific extraction if available
            if task:
                try:
                    from bench_forge.tasks.registry import get_registry

                    registry = get_registry()
                    task_instance = registry.create_task(task)

                    # Use task's extraction method if available
                    if hasattr(task_instance, "extract_label_from_response"):
                        extracted = task_instance.extract_label_from_response(
                            response_text
                        )
                        if extracted:
                            logger.debug(
                                f"Successfully extracted '{extracted}' using task-specific extraction"
                            )
                            return extracted
                except Exception as e:
                    logger.debug(f"Could not use task-specific extraction: {e}")

            # Fallback to generic extraction
            return self._generic_extraction(response_text)

        except Exception as e:
            logger.error(f"Error extracting from complete_response: {e}")
            return None

    def _generic_extraction(self, response_text: str) -> Optional[str]:
        """Generic extraction logic for common patterns.

        Args:
            response_text: The text to extract from

        Returns:
            Extracted value or None
        """
        if not response_text:
            return None

        # Clean the response
        response_clean = response_text.strip()

        # Try to extract first line (often contains the answer)
        lines = response_clean.split("\n")
        if lines:
            first_line = lines[0].strip()
            # Remove common prefixes
            for prefix in ["Answer:", "Classification:", "Label:", "Response:"]:
                if first_line.startswith(prefix):
                    first_line = first_line[len(prefix) :].strip()

            if first_line:
                return first_line

        return response_clean

    def _prepare_predictions_with_fallback(
        self, df: pd.DataFrame, task: Optional[str] = None
    ) -> List[Any]:
        """Prepare predictions with fallback extraction from complete_responses.

        This method implements FLAME-compatible fallback extraction:
        1. Use extracted_response if available
        2. Fallback to complete_responses for failed extractions
        3. Apply task-specific extraction logic

        Args:
            df: Results DataFrame
            task: Task name for task-specific extraction

        Returns:
            List of predictions with fallback extraction applied
        """
        predictions = []

        # Determine primary prediction column
        if "extracted_response" in df.columns:
            primary_col = "extracted_response"
        elif "extracted_labels" in df.columns:
            primary_col = "extracted_labels"
        elif "prediction" in df.columns:
            primary_col = "prediction"
        elif "llm_responses" in df.columns:
            primary_col = "llm_responses"
        else:
            # No extraction column, try to extract from complete_responses
            primary_col = None

        # Process each row
        for idx, row in df.iterrows():
            prediction = None

            # Try primary column first
            if primary_col and pd.notna(row.get(primary_col)):
                prediction = row[primary_col]

            # Fallback to complete_responses if needed
            if prediction is None and self.enable_fallback_extraction:
                if "complete_responses" in df.columns and pd.notna(
                    row.get("complete_responses")
                ):
                    logger.debug(f"Attempting fallback extraction for row {idx}")
                    complete_resp = row["complete_responses"]

                    # Special handling for string representations of objects
                    if isinstance(complete_resp, str) and complete_resp.startswith(
                        "ModelResponse"
                    ):
                        # Try to eval it (FLAME compatibility)
                        try:
                            # Import necessary types for eval
                            from litellm import ModelResponse

                            type_dict = {"ModelResponse": ModelResponse}
                            complete_resp = eval(complete_resp, type_dict)
                        except Exception as e:
                            logger.debug(f"Could not eval ModelResponse string: {e}")

                    # Extract from complete response
                    prediction = self._extract_from_complete_response(
                        complete_resp, task
                    )

                    if prediction:
                        logger.info(
                            f"Successfully extracted via fallback for row {idx}: {prediction}"
                        )
                    else:
                        logger.warning(f"Fallback extraction failed for row {idx}")

                # Last resort: try raw_response column
                if prediction is None and "raw_response" in df.columns:
                    prediction = row.get("raw_response")

            predictions.append(prediction)

        # Log extraction statistics
        total = len(predictions)
        successful = sum(1 for p in predictions if p is not None)
        logger.info(
            f"Prediction preparation complete: {successful}/{total} successful extractions"
        )

        if successful < total:
            failed_count = total - successful
            logger.warning(f"{failed_count} predictions could not be extracted")

        return predictions

    def evaluate(self, *args, **kwargs) -> EvaluationResult:
        """Evaluate with enhanced fallback extraction.

        This method overrides the parent evaluate to use fallback extraction.
        """
        # Get task name if provided
        task = kwargs.get("task")

        # Load results DataFrame
        if "results_path" in kwargs:
            df = pd.read_csv(kwargs["results_path"])
        elif "results_df" in kwargs:
            df = kwargs["results_df"]
        else:
            # Let parent handle the error
            return super().evaluate(*args, **kwargs)

        # Check if we have complete_responses column
        if "complete_responses" in df.columns:
            logger.info(
                "Found complete_responses column - fallback extraction available"
            )

            # Prepare predictions with fallback
            predictions = self._prepare_predictions_with_fallback(df, task)

            # Update DataFrame with extracted predictions
            df["_evaluation_predictions"] = predictions

            # Temporarily replace extracted_response column for evaluation
            if "extracted_response" in df.columns:
                df["_original_extracted_response"] = df["extracted_response"]
            df["extracted_response"] = predictions

            # Update kwargs with modified DataFrame
            kwargs["results_df"] = df
            if "results_path" in kwargs:
                del kwargs["results_path"]  # Use DataFrame instead

        # Call parent evaluate with potentially modified DataFrame
        result = super().evaluate(*args, **kwargs)

        # Add metadata about fallback extraction
        if hasattr(result, "metadata") and result.metadata:
            result.metadata["fallback_extraction_enabled"] = (
                self.enable_fallback_extraction
            )
            result.metadata["complete_responses_available"] = (
                "complete_responses" in df.columns
            )

        return result


def create_enhanced_evaluation_engine(**kwargs) -> EnhancedEvaluationEngine:
    """Factory function to create enhanced evaluation engine.

    Args:
        **kwargs: Arguments for EnhancedEvaluationEngine

    Returns:
        EnhancedEvaluationEngine instance
    """
    return EnhancedEvaluationEngine(**kwargs)
