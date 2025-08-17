"""Financial PhraseBank (FPB) task implementation using BenchForge.

Financial sentiment classification task using the Financial PhraseBank dataset.
"""

from typing import Any, Dict, List, Optional

from flame.benchforge import (
    flame_task,
    FLAMEConfig,
    PromptFormat,
    ExtractionStrategy,
)
from flame.tasks.base_flame_task import BaseFLAMETask


@flame_task("fpb")
class FPBTask(BaseFLAMETask):
    """Financial PhraseBank sentiment classification task.

    This task classifies financial news sentences as POSITIVE, NEGATIVE, or NEUTRAL
    based on sentiment towards the company or financial performance.
    """

    def __init__(self, config: Optional[FLAMEConfig] = None):
        """Initialize FPB task."""
        if config is None:
            config = FLAMEConfig(
                name="fpb",
                dataset="fpb",
                huggingface_dataset="financial_phrasebank",
                metrics=["accuracy", "f1_weighted"],
                prompt_format=PromptFormat.ZERO_SHOT,
                max_tokens=10,
                batch_size=20,
                # FLAME-specific
                text_field="sentence",
                label_field="label",
                valid_labels=["POSITIVE", "NEGATIVE", "NEUTRAL"],
                extraction_strategy=ExtractionStrategy.KEYWORD,
                financial_domain="sentiment_analysis",
            )

        super().__init__(config)

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create FPB sentiment classification prompt.

        Args:
            sample: Dataset sample
            format: Prompt format

        Returns:
            Formatted prompt
        """
        format = format or self.config.prompt_format

        # Preprocess sample
        sample = self.preprocess_sample(sample)
        text = sample.get(self.config.text_field, sample.get("text", ""))

        if format == PromptFormat.ZERO_SHOT:
            return self._create_zero_shot_prompt(text)
        elif format == PromptFormat.FEW_SHOT:
            return self._create_few_shot_prompt(text)
        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            return self._create_cot_prompt(text)
        else:
            # Default fallback
            return f"Sentiment of '{text}' (POSITIVE/NEGATIVE/NEUTRAL):"

    def _create_zero_shot_prompt(self, text: str) -> str:
        """Create zero-shot prompt for FPB."""
        return f"""Analyze the sentiment of the following financial text and classify it as POSITIVE, NEGATIVE, or NEUTRAL.

Consider the financial implications and investor perspective when determining sentiment.

Text: {text}

Sentiment (POSITIVE/NEGATIVE/NEUTRAL):"""

    def _create_few_shot_prompt(self, text: str) -> str:
        """Create few-shot prompt for FPB."""
        examples = """Example 1:
Text: The company's profits exceeded analyst expectations by 15%, driven by strong sales growth.
Sentiment: POSITIVE

Example 2:
Text: Sales declined sharply in the third quarter, falling 20% below forecasts.
Sentiment: NEGATIVE

Example 3:
Text: The board meeting is scheduled for next Tuesday to discuss quarterly results.
Sentiment: NEUTRAL"""

        return f"""Analyze financial sentiment for investment decision-making.

{examples}

Now analyze:
Text: {text}
Sentiment (POSITIVE/NEGATIVE/NEUTRAL):"""

    def _create_cot_prompt(self, text: str) -> str:
        """Create chain-of-thought prompt for FPB."""
        return f"""Analyze the following financial text and determine its sentiment.

Think step-by-step:
1. Identify key financial indicators and metrics mentioned
2. Assess whether these indicate positive or negative performance
3. Consider the implications for investors and stakeholders
4. Determine the overall sentiment

Text: {text}

Let's analyze step by step:"""

    def get_default_examples(self) -> List[Dict[str, Any]]:
        """Get default examples for few-shot prompting."""
        return [
            {
                "sentence": "The company's profits exceeded expectations.",
                "label": "POSITIVE",
            },
            {
                "sentence": "Sales declined sharply in the third quarter.",
                "label": "NEGATIVE",
            },
            {
                "sentence": "The board meeting is scheduled for next Tuesday.",
                "label": "NEUTRAL",
            },
        ]

    def postprocess_response(self, response: str) -> str:
        """Postprocess FPB response."""
        response = super().postprocess_response(response)

        # Normalize common variations
        response_upper = response.upper()

        # Handle common variations
        if any(
            word in response_upper
            for word in ["POSITIVE", "BULLISH", "GOOD", "FAVORABLE"]
        ):
            return "POSITIVE"
        elif any(
            word in response_upper
            for word in ["NEGATIVE", "BEARISH", "BAD", "UNFAVORABLE"]
        ):
            return "NEGATIVE"
        elif any(word in response_upper for word in ["NEUTRAL", "MIXED", "UNCERTAIN"]):
            return "NEUTRAL"

        # Check for numeric labels (sometimes models output 0, 1, 2)
        if "2" in response or "two" in response.lower():
            return "POSITIVE"
        elif "0" in response or "zero" in response.lower():
            return "NEGATIVE"
        elif "1" in response or "one" in response.lower():
            return "NEUTRAL"

        return response

    def compute_task_metrics(self, results_df) -> Dict[str, float]:
        """Compute FPB-specific metrics."""
        metrics = super().compute_task_metrics(results_df)

        # Add sentiment-specific metrics
        if "extracted_response" in results_df.columns:
            sentiment_dist = (
                results_df["extracted_response"].value_counts(normalize=True).to_dict()
            )

            for sentiment, proportion in sentiment_dist.items():
                metrics[f"proportion_{sentiment.lower()}"] = proportion

        return metrics
