def subjectiveqa_extraction_prompt(llm_response, feature):
    """Prompt to extract a valid label for SubjectiveQA."""
    return f"""The LLM output provided below contains the predicted rating for the feature '{feature}'.
    Extract the rating as one of the following numbers: 0, 1, or 2, without any explanation or additional text.
    If the rating is missing or the format is invalid, return 'error'.

    LLM Response: "{llm_response}" """
    
    
def numclaim_extraction_prompt(llm_response: str):
    prompt = f"""Based on the provided response, extract the following information:
                - Label the response as 'INCLAIM' if it contains the word INCLAIM or any numeric value or quantitative assertion.
                - Label the response as 'OUTCOFLAIM' if it contains the word OUTOFCLAIM or any qualitative assertion.
                ONLY PROVIDE THE LABEL WITHOUT ANY ADDITIONAL TEXT.
                The response: "{llm_response}"."""
    return prompt

def fnxl_extraction_prompt(raw_response: str):
    """
    Prompt to transform the raw LLM output into a standard JSON dict:
      {
        "us-gaap:SomeTag": [1.0, 2.0],
        "other": [3.0]
      }
    The LLM might have used a different format or extra text, so we
    ask it to re-extract in a consistent way.
    """
    prompt = f"""An LLM previously gave the following response about numerals and XBRL tags:
    ---
    {raw_response}

    Please convert that into valid JSON of the form:
    {{
      "xbrl_tag": [list_of_numerical_values],
      "other_tag": [list_of_numerical_values]
    }}
    If you have no numerals for a certain tag, omit that tag.
    Only return the JSON. Do not include any extra text.
    """
    return prompt


def finentity_extraction_prompt(model_response: str):
    """Generate a prompt to reformat extracted entity lists into structured JSON."""
    prompt = f"""Reformat the following extracted entity list into a structured JSON array.
                Use the exact format below, ensuring each entity has 'value', 'tag', and 'label'.
                Return only the JSON list, with no additional text.

                Original output:
                {model_response}

                Example format:
                [
                {{'value': 'EntityName', 'tag': 'NEUTRAL', 'label': 'NEUTRAL'}},
                {{'value': 'EntityName2', 'tag': 'POSITIVE', 'label': 'POSITIVE'}}
                ]

                Please ensure the format is valid JSON with all required fields. Make sure it does not throw a JSON decoding error."""
    return prompt


def causal_classifciation_extraction_prompt(llm_response: str):
    """Generate a prompt to extract the label from the LLM response."""
    return f"""The LLM output provided below contains the predicted label. Extract the label as a single number (0, 1, or 2) without any explanation or additional text. If the label is missing, return 'error'.

    LLM Response: "{llm_response}" """
    
    
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
banking77_label_map = {category: index for index, category in enumerate(banking77_list)}

# Define the prompt for LLM response extraction
def banking_77_extraction_prompt(llm_response: str):
    prompt = f"""Based on the following list of banking intents: {banking77_list}, extract the most relevant category from the following response:
                "{llm_response}"
                Provide only the label that best matches the response, exactly as it appears in the initial list of intents, with an underscore (_) between words. Only output alphanumeric characters and underscores. Do not include any special characters or punctuation. Only output the label. Do not list an explanation or multiple labels."""
    return prompt