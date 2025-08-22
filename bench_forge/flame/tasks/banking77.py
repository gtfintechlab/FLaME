"""Banking77 task implementation for BenchForge."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from difflib import get_close_matches

from bench_forge.flame.adapter import FLAMEConfig, FLAMETask, flame_task
from bench_forge.tasks.config import PromptFormat

logger = logging.getLogger(__name__)


# Banking77 categories (77 intent classes)
BANKING77_CATEGORIES = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "refund_not_showing_up",
    "request_refund",
    "reverted_card_payment",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


@dataclass
class Banking77Config(FLAMEConfig):
    """Configuration for Banking77 intent classification task."""

    # Dataset configuration
    huggingface_dataset: str = "gtfintechlab/banking77"
    text_field: str = "text"
    label_field: str = "label"
    dataset_split: str = "test"

    # Banking77-specific fields
    valid_labels: List[str] = field(default_factory=lambda: BANKING77_CATEGORIES)
    financial_domain: str = "banking_intent"
    
    # Intent mapping for alternative phrases
    intent_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Card-related intents
        "card_activation": "activate_my_card",
        "activate_card": "activate_my_card",
        "card_expired": "card_about_to_expire",
        "card_expiry": "card_about_to_expire", 
        "card_not_accepted": "card_acceptance",
        "card_declined": "declined_card_payment",
        "card_stolen": "lost_or_stolen_card",
        "card_lost": "lost_or_stolen_card",
        "compromised": "compromised_card",
        "card_blocked": "pin_blocked",
        
        # Transfer-related intents
        "money_transfer": "transfer_into_account",
        "transfer_money": "transfer_into_account",
        "send_money": "transfer_into_account",
        "transfer_cancelled": "cancel_transfer",
        "transfer_declined": "declined_transfer",
        "transfer_pending": "pending_transfer",
        "transfer_failed": "failed_transfer",
        "transfer_not_received": "transfer_not_received_by_recipient",
        
        # Top-up related intents
        "add_money": "topping_up_by_card",
        "top_up": "topping_up_by_card",
        "deposit": "top_up_by_cash_or_cheque",
        "add_funds": "topping_up_by_card",
        
        # Payment related intents
        "payment_declined": "declined_card_payment",
        "payment_failed": "declined_card_payment",
        "payment_pending": "pending_card_payment",
        "payment_not_recognised": "card_payment_not_recognised",
        
        # Account related intents
        "close_account": "terminate_account",
        "delete_account": "terminate_account",
        "change_details": "edit_personal_details",
        "update_details": "edit_personal_details",
        "forgot_password": "passcode_forgotten",
        "forgot_pin": "passcode_forgotten",
        
        # Verification intents
        "identity_verification": "verify_my_identity",
        "verify_identity": "verify_my_identity",
        "source_of_funds": "verify_source_of_funds",
        "verify_funds": "verify_source_of_funds",
        
        # Support related intents
        "exchange_rates": "exchange_rate",
        "currency_exchange": "exchange_rate",
        "atm_locations": "atm_support",
        "supported_countries": "country_support",
    })

    def __post_init__(self):
        """Post-initialization setup."""
        if self.name == "unknown":
            self.name = "banking77"
        super().__post_init__()


@flame_task("banking77")
class Banking77Task(FLAMETask):
    """Banking77 intent classification task.
    
    This task classifies banking customer service queries into 77 specific intent categories.
    The categories cover the full spectrum of banking operations including payments, transfers,
    cards, account management, verification, and support queries.
    
    Features:
    - 77-class intent classification
    - Banking domain-specific understanding
    - Multi-strategy intent extraction with fallbacks
    - Fuzzy matching for intent variations
    - FLAME-compatible evaluation
    
    Input format:
    - text: Customer query/sentence
    - label: Ground truth intent category
    """

    def __init__(self, config: Optional[Banking77Config] = None):
        """Initialize Banking77 task."""
        if config is None:
            config = Banking77Config(name="banking77")
        elif not isinstance(config, Banking77Config):
            banking77_config = Banking77Config(**config.__dict__)
            config = banking77_config

        super().__init__(config)
        self.config: Banking77Config = config

        logger.info("Initialized Banking77 task with 77 banking intent categories")

    def create_prompt(
        self, sample: Dict[str, Any], format: Optional[PromptFormat] = None
    ) -> str:
        """Create prompt for Banking77 using exact FLAME prompt."""
        format = format or self.config.prompt_format

        # Extract query
        sentence = sample.get(self.config.text_field, "")

        if format == PromptFormat.ZERO_SHOT:
            # Use exact FLAME prompt with all 77 categories
            categories_str = ", ".join(BANKING77_CATEGORIES)
            prompt = f"""Discard all the previous instructions. Behave like you are an expert at
fine-grained single-domain intent detection. From the following list: {categories_str}, identify
which category the following sentence belongs to.
{sentence}"""

        elif format == PromptFormat.FEW_SHOT:
            prompt = f"""You are an expert at banking intent classification. Classify the query into one of the banking intent categories.

Examples:
Query: "I can't find my card anywhere, I think I lost it"
Intent: lost_or_stolen_card

Query: "Why was my payment declined at the store?"
Intent: declined_card_payment

Query: "How do I add money to my account?"
Intent: topping_up_by_card

Query: "I need to verify my identity"
Intent: verify_my_identity

Query: "What's the current exchange rate?"
Intent: exchange_rate

Available categories: {", ".join(BANKING77_CATEGORIES)}

Now classify this query:
Query: {sentence}
Intent:"""

        elif format == PromptFormat.CHAIN_OF_THOUGHT:
            prompt = f"""Let's analyze this banking query step by step to determine the correct intent.

Query: {sentence}

Let me think through this:
1. What is the main topic or concern?
2. What action or information is the customer seeking?
3. Which banking intent category best matches this?

Available categories: {", ".join(BANKING77_CATEGORIES)}

Step 1 - Main topic:
Step 2 - Customer need:
Step 3 - Best matching intent:"""

        else:
            prompt = self.create_prompt(sample, PromptFormat.ZERO_SHOT)

        self._stats["prompts_created"] += 1
        return prompt

    def extract_response(
        self, raw_response: str, sample: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Extract intent using multi-strategy approach."""
        self._stats["responses_extracted"] += 1

        if not raw_response:
            return None

        # Clean the response
        response = raw_response.strip()

        # Strategy 1: Direct intent match
        extracted = self._extract_direct_intent(response)
        if extracted is not None:
            return extracted

        # Strategy 2: Extract from structured format
        extracted = self._extract_structured_intent(response)
        if extracted is not None:
            return extracted

        # Strategy 3: Find intent in response text
        extracted = self._extract_intent_from_text(response)
        if extracted is not None:
            return extracted

        # Strategy 4: Fuzzy matching with banking intents
        extracted = self._extract_fuzzy_match(response)
        if extracted is not None:
            return extracted

        # Strategy 5: Keyword-based intent detection
        extracted = self._extract_keyword_based_intent(response)
        if extracted is not None:
            return extracted

        # Strategy 6: Pattern-based extraction
        extracted = self._extract_pattern_based(response)
        if extracted is not None:
            return extracted

        logger.debug(f"Failed to extract intent from response: {raw_response[:100]}...")
        self._stats["extraction_failures"] += 1
        return None

    def _extract_direct_intent(self, response: str) -> Optional[str]:
        """Extract if response starts with or is exactly an intent."""
        response_clean = response.lower().strip()
        
        # Check if response is exactly one of our intents
        for intent in self.config.valid_labels:
            if response_clean == intent.lower():
                return intent
        
        # Check if response starts with an intent
        for intent in self.config.valid_labels:
            if response_clean.startswith(intent.lower()):
                return intent
        
        return None

    def _extract_structured_intent(self, response: str) -> Optional[str]:
        """Extract intent from structured responses."""
        # Look for "Intent: X", "Category: X", "Answer: X" patterns
        structured_patterns = [
            r'intent\s*[:=]\s*([^\n,]+)',
            r'category\s*[:=]\s*([^\n,]+)', 
            r'answer\s*[:=]\s*([^\n,]+)',
            r'classification\s*[:=]\s*([^\n,]+)',
            r'result\s*[:=]\s*([^\n,]+)',
        ]
        
        for pattern in structured_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                intent_candidate = matches[0].strip()
                # Check if it's a valid intent
                valid_intent = self._validate_intent(intent_candidate)
                if valid_intent:
                    return valid_intent
        
        return None

    def _extract_intent_from_text(self, response: str) -> Optional[str]:
        """Find any valid intent mentioned in the response."""
        response_lower = response.lower()
        
        # Look for each intent in the response
        found_intents = []
        for intent in self.config.valid_labels:
            if intent.lower() in response_lower:
                found_intents.append(intent)
        
        # Return the first (or most specific) intent found
        if found_intents:
            return found_intents[0]
        
        return None

    def _extract_fuzzy_match(self, response: str) -> Optional[str]:
        """Use fuzzy string matching to find closest intent."""
        # Get the first word or phrase from response
        words = response.lower().split()
        if not words:
            return None
        
        # Try fuzzy matching with different phrase lengths
        for length in [1, 2, 3]:
            if length <= len(words):
                phrase = "_".join(words[:length])
                matches = get_close_matches(phrase, [intent.lower() for intent in self.config.valid_labels], 
                                          n=1, cutoff=0.7)
                if matches:
                    # Find the original intent that matches
                    for intent in self.config.valid_labels:
                        if intent.lower() == matches[0]:
                            return intent
        
        return None

    def _extract_keyword_based_intent(self, response: str) -> Optional[str]:
        """Extract intent based on keywords and context."""
        response_lower = response.lower()
        
        # Define keyword patterns for common intents
        keyword_patterns = {
            "activate_my_card": ["activate", "card", "enable"],
            "lost_or_stolen_card": ["lost", "stolen", "missing", "can't find"],
            "declined_card_payment": ["declined", "rejected", "refused", "blocked"],
            "card_not_working": ["not working", "broken", "faulty", "doesn't work"],
            "transfer_into_account": ["transfer", "send money", "move money"],
            "topping_up_by_card": ["top up", "add money", "deposit", "load"],
            "balance_not_updated_after_bank_transfer": ["balance", "updated", "transfer"],
            "exchange_rate": ["exchange", "rate", "currency"],
            "verify_my_identity": ["verify", "identity", "verification"],
            "terminate_account": ["close", "delete", "cancel", "account"],
            "edit_personal_details": ["change", "update", "edit", "details"],
            "passcode_forgotten": ["forgot", "forgotten", "pin", "password"],
            "refund_not_showing_up": ["refund", "not showing", "missing"],
            "card_payment_fee_charged": ["fee", "charged", "charge"],
        }
        
        # Score each intent based on keyword presence
        intent_scores = {}
        for intent, keywords in keyword_patterns.items():
            score = sum(1 for keyword in keywords if keyword in response_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            return best_intent
        
        return None

    def _extract_pattern_based(self, response: str) -> Optional[str]:
        """Extract using pattern matching for common response formats."""
        # Look for quoted intents
        quote_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'\[([^\]]+)\]',
            r'\(([^\)]+)\)',
        ]
        
        for pattern in quote_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                valid_intent = self._validate_intent(match)
                if valid_intent:
                    return valid_intent
        
        return None

    def _validate_intent(self, candidate: str) -> Optional[str]:
        """Validate if candidate is a valid intent."""
        if not candidate:
            return None
        
        candidate_clean = candidate.lower().strip()
        
        # Direct match
        for intent in self.config.valid_labels:
            if candidate_clean == intent.lower():
                return intent
        
        # Check mapping dictionary
        if candidate_clean in self.config.intent_mapping:
            return self.config.intent_mapping[candidate_clean]
        
        # Check if candidate is substring of valid intent
        for intent in self.config.valid_labels:
            if candidate_clean in intent.lower() or intent.lower() in candidate_clean:
                return intent
        
        return None

    def get_ground_truth(self, sample: Dict[str, Any]) -> Any:
        """Get ground truth intent from sample."""
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
            query_text = sample.get(self.config.text_field, "")
            actual_intent = self.get_ground_truth(sample)

            result = {
                "index": i,
                # FLAME-compatible fields for Banking77
                "documents": query_text,  # FLAME primary field
                "llm_responses": response_text,  # FLAME primary field
                "actual_labels": actual_intent,  # FLAME primary field
                "complete_responses": complete_response,  # FLAME primary field
                "extracted_labels": extracted,  # FLAME primary field
                # Banking77-specific fields
                "text": query_text,
                "query": query_text,
                "intent": actual_intent,
                "predicted_intent": extracted,
                # Standard BenchForge fields
                "prompt": prompt,
                "input": query_text,
                "ground_truth": actual_intent,
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
            f"Banking77 extraction results: {successful}/{total} successful extractions ({success_rate:.1f}%)"
        )

        # Log intent distribution if successful extractions exist
        if successful > 0:
            extracted_intents = df["extracted_labels"].dropna()
            intent_counts = extracted_intents.value_counts()
            logger.info(f"Top 5 predicted intents: {dict(intent_counts.head().items())}")

        return df