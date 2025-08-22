"""FinBench task implementation for BenchForge."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


@dataclass
class FinBenchConfig(FLAMEConfig):
    """Configuration for FinBench loan risk assessment task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/finbench"
    text_field: str = "X_profile"
    label_field: str = "y"
    dataset_split: str = "test"

    # FinBench-specific fields
    valid_labels: List[str] = field(default_factory=lambda: ["LOW RISK", "HIGH RISK"])
    financial_domain: str = "loan_risk_assessment"

    # Risk assessment mapping
    risk_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            # Direct mappings
            "low risk": "LOW RISK",
            "low_risk": "LOW RISK",
            "lowrisk": "LOW RISK",
            "low": "LOW RISK",
            "safe": "LOW RISK",
            "good": "LOW RISK",
            "approve": "LOW RISK",
            "approved": "LOW RISK",
            "accept": "LOW RISK",
            "high risk": "HIGH RISK",
            "high_risk": "HIGH RISK",
            "highrisk": "HIGH RISK",
            "high": "HIGH RISK",
            "risky": "HIGH RISK",
            "dangerous": "HIGH RISK",
            "reject": "HIGH RISK",
            "rejected": "HIGH RISK",
            "deny": "HIGH RISK",
            "denied": "HIGH RISK",
            # Alternative phrasings
            "unlikely to default": "LOW RISK",
            "will pay back": "LOW RISK",
            "creditworthy": "LOW RISK",
            "reliable": "LOW RISK",
            "stable": "LOW RISK",
            "likely to default": "HIGH RISK",
            "will not pay back": "HIGH RISK",
            "unreliable": "HIGH RISK",
            "unstable": "HIGH RISK",
            "default risk": "HIGH RISK",
        }
    )

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "finbench"
        super().__post_init__()


@flame_task("finbench")
class FinBenchTask(FLAMETask):
    """FinBench loan risk assessment task.

    This task evaluates loan applicant profiles to classify them as either
    LOW RISK or HIGH RISK for loan approval. It analyzes financial and
    demographic information to predict the likelihood of loan default.

    Features:
    - Binary risk classification (LOW RISK / HIGH RISK)
    - Financial profile analysis
    - Loan default risk assessment
    - Multi-strategy risk label extraction
    - FLAME-compatible evaluation
    - Comprehensive risk assessment reasoning

    Input format:
    - X_profile: Applicant profile data (financial and demographic info)
    - y: Ground truth risk label (LOW RISK or HIGH RISK)
    """

    def __init__(self, config: Optional[FinBenchConfig] = None):
        """Initialize FinBench task."""
        if config is None:
            config = FinBenchConfig(name="finbench")
        elif not isinstance(config, FinBenchConfig):
            finbench_config = FinBenchConfig(**config.__dict__)
            config = finbench_config

        super().__init__(config)
        self.config: FinBenchConfig = config

        logger.info("Initialized FinBench task with loan risk assessment capabilities")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for FinBench using exact FLAME prompt."""
        format = format or self.config.prompt_format

        # Extract profile data
        profile = sample.get(self.config.text_field, "")

        if format == PromptFormat.ZERO_SHOT:
            # Use exact FLAME prompt
            prompt = f"""Discard all the previous instructions. Behave like you are an expect risk assessor.
Classify the following individual as either 'LOW RISK' or 'HIGH RISK' for approving a loan for.
Categorize the person as 'HIGH RISK' if their profile indicates that they will likely default on
the loan and not pay it back, and 'LOW RISK' if it is unlikely that they will fail to pay the loan back in full.
Provide the label in the first line and provide a short explanation in the second line. Explain how you came to your classification decision and output the label that you chose. Do not write any code, simply think and provide your decision.
Here is the information about the person:
Profile data: {profile}
Predict the risk category of this person:"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""You are an expert loan risk assessor. Classify loan applicants as LOW RISK or HIGH RISK based on their profile.

Examples:

Profile: Age: 45, Income: $85000, Debt: $15000, Credit Score: 750, Employment: Full-time
LOW RISK
This applicant has a strong financial profile with high income, manageable debt, and excellent credit.

Profile: Age: 22, Income: $25000, Debt: $45000, Credit Score: 580, Employment: Part-time  
HIGH RISK
This applicant has high debt relative to income, poor credit score, and unstable employment.

Profile: Age: 35, Income: $65000, Debt: $8000, Credit Score: 720, Employment: Full-time
LOW RISK
This applicant shows financial stability with good income-to-debt ratio and strong credit.

Now classify this applicant:
Profile data: {profile}
Predict the risk category of this person:"""

        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            prompt = f"""Let's analyze this loan applicant's risk profile step by step.

Profile data: {profile}

Step 1: Analyze income and employment stability
Step 2: Evaluate debt-to-income ratio and existing obligations  
Step 3: Assess credit history and score
Step 4: Consider demographic factors and loan history
Step 5: Determine overall risk level

Final risk assessment (LOW RISK or HIGH RISK):"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract risk label using multi-strategy approach."""
        self._stats["responses_extracted"] += 1

        if not raw_response:
            return None

        # Clean the response
        response = raw_response.strip()

        # Strategy 1: Extract from first line (FLAME format)
        extracted = self._extract_first_line_risk(response)
        if extracted:
            return extracted

        # Strategy 2: Direct risk label match
        extracted = self._extract_direct_risk_label(response)
        if extracted:
            return extracted

        # Strategy 3: Extract from structured format
        extracted = self._extract_structured_risk(response)
        if extracted:
            return extracted

        # Strategy 4: Risk keyword analysis
        extracted = self._extract_keyword_based_risk(response)
        if extracted:
            return extracted

        # Strategy 5: Pattern-based extraction
        extracted = self._extract_pattern_based_risk(response)
        if extracted:
            return extracted

        # Strategy 6: Context-based risk assessment
        extracted = self._extract_contextual_risk(response)
        if extracted:
            return extracted

        logger.debug(
            f"Failed to extract risk label from response: {raw_response[:100]}..."
        )
        self._stats["extraction_failures"] += 1
        return None

    def _extract_first_line_risk(self, response: str) -> Optional[str]:
        """Extract risk label from first line (FLAME format)."""
        lines = response.split("\n")
        if not lines:
            return None

        first_line = lines[0].strip().upper()

        # Check for direct matches
        for label in self.config.valid_labels:
            if label.upper() in first_line:
                return label

        # Check mapping dictionary
        for key, value in self.config.risk_mapping.items():
            if key.upper() in first_line:
                return value

        return None

    def _extract_direct_risk_label(self, response: str) -> Optional[str]:
        """Extract risk label through direct matching."""
        response_upper = response.upper()

        # Check for exact label matches first
        for label in self.config.valid_labels:
            if label.upper() in response_upper:
                return label

        return None

    def _extract_structured_risk(self, response: str) -> Optional[str]:
        """Extract from structured response formats."""
        # Look for patterns like "Risk: HIGH RISK", "Classification: LOW RISK", etc.
        structured_patterns = [
            r"(?:risk|classification|category|assessment|decision|label)\s*[:=]\s*([^\n,]+)",
            r"(?:predict|classify|determine)\s+(?:as|to be)?\s*:?\s*([^\n,]+)",
            r"(?:result|conclusion|verdict)\s*[:=]\s*([^\n,]+)",
        ]

        for pattern in structured_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                risk_candidate = matches[0].strip()
                validated_risk = self._validate_risk_label(risk_candidate)
                if validated_risk:
                    return validated_risk

        return None

    def _extract_keyword_based_risk(self, response: str) -> Optional[str]:
        """Extract risk based on keyword analysis."""
        response_lower = response.lower()

        # Count indicators for each risk level
        low_risk_indicators = [
            "low risk",
            "safe",
            "good",
            "stable",
            "reliable",
            "creditworthy",
            "unlikely to default",
            "will pay back",
            "approve",
            "accept",
            "strong financial",
            "excellent credit",
            "high income",
            "stable employment",
        ]

        high_risk_indicators = [
            "high risk",
            "risky",
            "dangerous",
            "unstable",
            "unreliable",
            "likely to default",
            "will not pay back",
            "reject",
            "deny",
            "poor credit",
            "low income",
            "high debt",
            "unstable employment",
            "default risk",
            "financial instability",
        ]

        low_score = sum(
            1 for indicator in low_risk_indicators if indicator in response_lower
        )
        high_score = sum(
            1 for indicator in high_risk_indicators if indicator in response_lower
        )

        if low_score > high_score and low_score > 0:
            return "LOW RISK"
        elif high_score > low_score and high_score > 0:
            return "HIGH RISK"

        return None

    def _extract_pattern_based_risk(self, response: str) -> Optional[str]:
        """Extract using pattern matching."""
        # Look for quoted or bracketed labels
        quote_patterns = [
            r'["\']([^"\']+)["\']',
            r"\[([^\]]+)\]",
            r"\(([^\)]+)\)",
        ]

        for pattern in quote_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                validated_risk = self._validate_risk_label(match)
                if validated_risk:
                    return validated_risk

        return None

    def _extract_contextual_risk(self, response: str) -> Optional[str]:
        """Extract based on overall context and sentiment."""
        response_lower = response.lower()

        # Positive financial indicators
        positive_indicators = [
            "good",
            "strong",
            "stable",
            "reliable",
            "excellent",
            "high income",
            "low debt",
            "full-time",
            "employed",
            "creditworthy",
            "approve",
        ]

        # Negative financial indicators
        negative_indicators = [
            "bad",
            "poor",
            "weak",
            "unstable",
            "unreliable",
            "low income",
            "high debt",
            "unemployed",
            "part-time",
            "reject",
            "default",
        ]

        positive_score = sum(
            1 for indicator in positive_indicators if indicator in response_lower
        )
        negative_score = sum(
            1 for indicator in negative_indicators if indicator in response_lower
        )

        # Only return if there's a clear indication
        if positive_score >= 2 and positive_score > negative_score:
            return "LOW RISK"
        elif negative_score >= 2 and negative_score > positive_score:
            return "HIGH RISK"

        return None

    def _validate_risk_label(self, candidate: str) -> Optional[str]:
        """Validate if candidate is a valid risk label."""
        if not candidate:
            return None

        candidate_clean = candidate.strip().lower()

        # Direct match with valid labels
        for label in self.config.valid_labels:
            if candidate_clean == label.lower():
                return label

        # Check mapping dictionary
        if candidate_clean in self.config.risk_mapping:
            return self.config.risk_mapping[candidate_clean]

        # Partial matching
        if "low" in candidate_clean and "risk" in candidate_clean:
            return "LOW RISK"
        elif "high" in candidate_clean and "risk" in candidate_clean:
            return "HIGH RISK"

        return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth risk label from sample."""
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
            profile_data = sample.get(self.config.text_field, "")
            actual_risk = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields for FinBench
                "X_profile": profile_data,  # FLAME primary field
                "y": actual_risk,  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # FLAME primary field
                # FinBench-specific fields
                "profile": profile_data,
                "profile_data": profile_data,
                "risk_label": actual_risk,
                "predicted_risk": extracted,
                # Standard BenchForge fields
                "prompt": prompt,
                "input": profile_data,
                "ground_truth": actual_risk,
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
            f"FinBench extraction results: {successful}/{total} successful extractions ({success_rate:.1f}%)"
        )

        # Log risk distribution if successful extractions exist
        if successful > 0:
            extracted_risks = df["extracted_labels"].dropna()
            risk_distribution = extracted_risks.value_counts()
            logger.info(f"Risk distribution: {dict(risk_distribution.items())}")

        return df
