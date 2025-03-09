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


def finbench_extraction_prompt(llm_response: str):
    """Generate a prompt for extracting risk labels."""
    prompt = f"""Based on the following list of labels: ‘HIGH RISK’, ‘LOW RISK’, extract the most relevant label from the following response:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    return prompt


def finqa_extraction_prompt(llm_response: str):
    prompt = f"""
    You will receive a response from a language model that may include a numerical answer within its text. 
    Your task is to extract and return only the main/final answer. This could be represented as an integer, decimal, percentage, or text.
    Respond with whatever is labeled as the final answer, if that exists, even if that contains text. Otherwise, stick to numerical answers.
    Do not include any additional text or formatting. 

    Model Response: {llm_response}

    Please respond with the final answer. If a final answer was not provided, respond NA.
    """
    return prompt


# Define possible relationships
possible_relationships = [
    'subsidiary', 'owned_by', 'employer', 'product_or_material_produced', 'industry',
    'manufacturer', 'developer', 'legal_form', 'parent_organization', 'distribution_format',
    'chairperson', 'location_of_formation', 'headquarters_location', 'operator', 'creator',
    'currency', 'founded_by', 'original_broadcaster', 'owner_of', 'director_/_manager',
    'business_division', 'chief_executive_officer', 'position_held', 'platform', 'brand',
    'distributed_by', 'publisher', 'stock_exchange', 'member_of'
]

def finred_extraction_prompt(llm_response: str):
    """Generate a prompt to extract the classification label from the LLM response."""
    relationship_choices = ', '.join(possible_relationships)
    prompt = f'''Extract the classification label from the following LLM response. The label should be one of the following {relationship_choices}. 
    
                Pick the label out of the list that is the closest to the LLM response, but list ‘NO-REL’ if the LLM did not output a clear answer.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response, exactly as it is listed in the approved label list, with an underscore (_) between words. Only output alphanumeric characters, spaces, dashes, and underscores. Do not include any special characters, quotations, asterisks, or punctuation, etc. Only output the label. Do not list an explanation or multiple labels.'''
    return prompt



def fomc_extraction_prompt(llm_response: str) -> str:
    """Generate a prompt to extract the classification label from the LLM response.
    
    Args:
        llm_response: The raw response from the language model
        
    Returns:
        A formatted prompt string for label extraction
    """
    prompt = f'''Extract the classification label from the following LLM response. The label should be one of the following: 'HAWKISH', 'DOVISH', or 'NEUTRAL'.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation.'''
    return prompt

def fpb_extraction_prompt(llm_response: str):
    """Generate a prompt to extract the most relevant label from the LLM response."""
    prompt = f"""Based on the following list of labels: ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’, extract the most relevant label from the following response:
                "{llm_response}"
                Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    return prompt