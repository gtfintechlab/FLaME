"""
Zero-Shot Prompt Functions

This module contains zero-shot prompt functions for various FLaME tasks.
These prompts do not include examples and expect the model to perform the task
based solely on instructions.
"""

from .registry import register_prompt, PromptFormat


@register_prompt("headlines", PromptFormat.ZERO_SHOT)
def headlines_zeroshot_prompt(sentence: str) -> str:
    """Generate a zero-shot prompt for the Headlines task.

    This prompt asks the model to analyze a news headline and rate it on various attributes.

    Args:
        sentence: The news headline to analyze

    Returns:
        Formatted prompt string
    """
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

    return prompt


@register_prompt("numclaim", PromptFormat.ZERO_SHOT)
def numclaim_zeroshot_prompt(sentence: str) -> str:
    """Generate a zero-shot prompt for the NumClaim task.

    This prompt asks the model to classify a sentence as containing a claim or just factual information.

    Args:
        sentence: The sentence to classify

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence senti-
            ment classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class.
            Label 'INCLAIM' if consist of a claim and not just factual past or present information, or
            'OUTOFCLAIM' if it has just factual past or present information. Provide the label in the
            first line and provide a short explanation in the second line. The sentence:{sentence}"""

    return prompt


@register_prompt("fomc", PromptFormat.ZERO_SHOT)
def fomc_zeroshot_prompt(sentence: str) -> str:
    """Generate a zero-shot prompt for the FOMC task.

    This prompt asks the model to classify a sentence from FOMC minutes as hawkish, dovish, or neutral.

    Args:
        sentence: The FOMC sentence to classify

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence clas-
                sifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', or 'NEU-
                TRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the monetary policy,
                'DOVISH' if it is corresponding to easing of the monetary policy, or 'NEUTRAL' if the
                stance is neutral. Provide the label in the first line and provide a short explanation in the
                second line. This is the sentence: {sentence}"""

    return prompt


@register_prompt("fpb", PromptFormat.ZERO_SHOT)
def fpb_zeroshot_prompt(sentence: str, prompt_format: str = None) -> str:
    """Generate a zero-shot prompt for the Financial Phrase Bank sentiment classification task.

    This prompt asks the model to classify a financial sentence as positive, negative, or neutral.
    It supports multiple prompt formats for experimentation.

    Args:
        sentence: The financial sentence to classify
        prompt_format: The specific prompt format to use (flame, finben_icl, etc.)

    Returns:
        Formatted prompt string
    """
    # Default prompt format if none is specified
    if prompt_format is None or prompt_format not in [
        "flame",
        "finben_icl",
        "finben_noicl",
        "flame_icl",
        "flame_cot",
    ]:
        prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence clas-
                    sifier. Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL'
                    class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is
                    corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide
                    the label in the first line and provide a short explanation in the second line. This is the sentence: {sentence}"""
        return prompt

    # Handle different prompt formats
    if prompt_format == "flame":
        system_prompt = """ Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier"""

        user_msg = f""" Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL'
                    class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is
                    corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. 
                    Provide the label in the first line and provide a short explanation in the second line.
                    Explain how you came to your classification decision. This is the sentence: {sentence}."""

    elif prompt_format == "finben_icl":
        system_prompt = """"""
        user_msg = f""" Analyze the sentiment of this statement extracted from a financial news article.
                        Provide your answer as either NEGATIVE, POSITIVE or NEUTRAL.
                        For instance, 'The company's stocks plummeted following the scandal.' would be classified as negative. This is the sentence: {sentence}"""

    elif prompt_format == "finben_noicl":
        system_prompt = """"""
        user_msg = f""" Analyze the sentiment of this statement extracted from a financial news article.
                        Provide your answer as either NEGATIVE, POSITIVE or NEUTRAL.
                        This is the sentence: {sentence}"""

    elif prompt_format == "flame_icl":
        system_prompt = """Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier """
        user_msg = f""" Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL'
                        class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is
                        corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Provide
                        the label in the first line and provide a short explanation in the second line.
                        For instance: 
                        "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing" would be classified as 'NEUTRAL.
                        "When this investment is in place , Atria plans to expand into the Moscow market" would be classified as 'NEUTRAL'.
                        "With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability" would be classified as 'POSITIVE'.
                        "For the last quarter of 2010 , Componenta's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m" would be classified as 'POSITIVE'.
                        "Aspocomp has a large factory in China and a factory building project in India that was halted due to financing problems" would be classified as 'NEGATIVE'.
                        "The low capacity utilisation rate in steel production considerably increases the fixed costs per unit of steel produced" would be classified as 'NEGATIVE'.
                        This is the sentence: {sentence}"""

    elif prompt_format == "flame_cot":
        system_prompt = """Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier """
        user_msg = f""" Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL'
                        class. Label 'NEGATIVE' if it is corresponding to negative sentiment, 'POSITIVE' if it is
                        corresponding to positive sentiment, or 'NEUTRAL' if the sentiment is neutral. Let's think about this sentiment classification task step by step.
                        First, generate your reasoning steps for the classification. After your reasoning, end the response with the label that fits your reasoning.
                        This is the sentence: {sentence}"""

    prompt = f"""{system_prompt}\n{user_msg}"""
    return prompt


@register_prompt("edtsum", PromptFormat.ZERO_SHOT)
def edtsum_zeroshot_prompt(document: str) -> str:
    """Generate a zero-shot prompt for the EDTSum task.

    This prompt asks the model to perform abstractive summarization on a document.

    Args:
        document: The document to summarize

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all the previous instructions. Behave like you are an expert at summarization tasks.	
        You are given a text that consists of multiple sentences. Your task is to perform abstractive summarization 
        on this text. Use your understanding of the content to express the main ideas and crucial details in a shorter, coherent, and natural sounding text.
        \nThe text:\n{document}.\nOutput your concise summary below. Try to keep your summary to one sentence and a maximum of 50 words, preferably around 25 words."""
    return prompt


@register_prompt("banking77", PromptFormat.ZERO_SHOT)
def banking77_zeroshot_prompt(sentence: str) -> str:
    """Generate a zero-shot prompt for the Banking77 task.

    This prompt asks the model to classify a banking-related query into one of 77 intent categories.

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
