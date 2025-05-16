"""
Few-Shot Prompt Functions

This module contains few-shot prompt functions for various FLaME tasks.
Few-shot prompts include examples to help guide the model's behavior.
"""

from .registry import register_prompt, PromptFormat

# Most few-shot prompts are currently stubs for future implementation
# Only banking77_fewshot_prompt is fully implemented


@register_prompt("banking77", PromptFormat.FEW_SHOT)
def banking77_fewshot_prompt(sentence: str) -> str:
    """Generate a few-shot prompt for the Banking77 task.

    This prompt asks the model to classify a banking-related query into one of 77 intent categories.
    Implemented as a zero-shot prompt for now - will be updated with examples in future.

    Args:
        sentence: The banking query to classify

    Returns:
        Formatted prompt string
    """
    banking77_list = [
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
        "Refund_not_showing_up",
        "request_refund",
        "reverted_card_payment?",
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

    prompt = f"""Discard all the previous instructions. Behave like you are an expert at
                fine-grained single-domain intent detection. From the following list: {banking77_list}, identify
                which category the following sentence belongs to.
                {sentence}"""
    return prompt


# The following are stub functions that will be implemented in the future
# They are included here for completeness and to maintain registry consistency


@register_prompt("numclaim", PromptFormat.FEW_SHOT)
def numclaim_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot NumClaim prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fpb", PromptFormat.FEW_SHOT)
def fpb_fewshot_prompt(sentence: str, prompt_format: str = None) -> str:
    """Stub for few-shot FPB prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The financial sentence to classify
        prompt_format: Optional format specifier

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fomc", PromptFormat.FEW_SHOT)
def fomc_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FOMC prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The FOMC sentence to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("headlines", PromptFormat.FEW_SHOT)
def headlines_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot Headlines prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The headline to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fiqa_task1", PromptFormat.FEW_SHOT)
def fiqa_task1_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FiQA Task 1 prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The financial sentence to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fiqa_task2", PromptFormat.FEW_SHOT)
def fiqa_task2_fewshot_prompt(question: str) -> str:
    """Stub for few-shot FiQA Task 2 prompt.

    This is a placeholder for future implementation.

    Args:
        question: The financial question to answer

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("edtsum", PromptFormat.FEW_SHOT)
def edtsum_fewshot_prompt(document: str) -> str:
    """Stub for few-shot EDTSum prompt.

    This is a placeholder for future implementation.

    Args:
        document: The document to summarize

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("ectsum", PromptFormat.FEW_SHOT)
def ectsum_fewshot_prompt(document: str) -> str:
    """Stub for few-shot ECTSum prompt.

    This is a placeholder for future implementation.

    Args:
        document: The document to summarize

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finqa", PromptFormat.FEW_SHOT)
def finqa_fewshot_prompt(document: str) -> str:
    """Stub for few-shot FinQA prompt.

    This is a placeholder for future implementation.

    Args:
        document: The financial document to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("convfinqa", PromptFormat.FEW_SHOT)
def convfinqa_fewshot_prompt(document: str) -> str:
    """Stub for few-shot ConvFinQA prompt.

    This is a placeholder for future implementation.

    Args:
        document: The conversational financial document to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("causal_classification", PromptFormat.FEW_SHOT)
def causal_classification_fewshot_prompt(text: str) -> str:
    """Stub for few-shot Causal Classification prompt.

    This is a placeholder for future implementation.

    Args:
        text: The text to classify

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finred", PromptFormat.FEW_SHOT)
def finred_fewshot_prompt(sentence: str, entity1: str, entity2: str) -> str:
    """Stub for few-shot FinRED prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze
        entity1: The first entity
        entity2: The second entity

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("causal_detection", PromptFormat.FEW_SHOT)
def causal_detection_fewshot_prompt(tokens: list) -> str:
    """Stub for few-shot Causal Detection prompt.

    This is a placeholder for future implementation.

    Args:
        tokens: The list of tokens to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("subjectiveqa", PromptFormat.FEW_SHOT)
def subjectiveqa_fewshot_prompt(feature, definition, question, answer) -> str:
    """Stub for few-shot SubjectiveQA prompt.

    This is a placeholder for future implementation.

    Args:
        feature: The feature to analyze
        definition: The definition of the feature
        question: The question to answer
        answer: The answer to evaluate

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("fnxl", PromptFormat.FEW_SHOT)
def fnxl_fewshot_prompt(sentence, company, doc_type) -> str:
    """Stub for few-shot FNXL prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze
        company: The company name
        doc_type: The document type

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("refind", PromptFormat.FEW_SHOT)
def refind_fewshot_prompt(entities) -> str:
    """Stub for few-shot ReFinD prompt.

    This is a placeholder for future implementation.

    Args:
        entities: The entities to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finentity", PromptFormat.FEW_SHOT)
def finentity_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FinEntity prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finer", PromptFormat.FEW_SHOT)
def finer_fewshot_prompt(sentence: str) -> str:
    """Stub for few-shot FinER prompt.

    This is a placeholder for future implementation.

    Args:
        sentence: The sentence to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None


@register_prompt("finbench", PromptFormat.FEW_SHOT)
def finbench_fewshot_prompt(profile: str) -> str:
    """Stub for few-shot FinBench prompt.

    This is a placeholder for future implementation.

    Args:
        profile: The profile to analyze

    Returns:
        None (will be implemented in future)
    """
    # placeholder for the prompt
    return None
