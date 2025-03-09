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

def finer_extraction_prompt(llm_response: str):
    """Generate a prompt to extract numeric labels for named entity recognition."""
    prompt = f"""For each token in the following response, map the named entity labels to these numeric values:
                    - "O" (Other): 0
                    - "PER_B" (Person_B): 1
                    - "PER_I" (Person_I): 2
                    - "LOC_B" (Location_B): 3
                    - "LOC_I" (Location_I): 4
                    - "ORG_B" (Organisation_B): 5
                    - "ORG_I" (Organisation_I): 6
 
                Provide only the list of integer labels, in the format:
                [0, 1, 0, ...]
 
                Do not include any additional text, explanations, or formatting other than a plain list.
 
                LLM response:
                "{llm_response}"."""
    return prompt

def causal_classifciation_extraction_prompt(llm_response: str):
    """Generate a prompt to extract the label from the LLM response."""
    return f"""The LLM output provided below contains the predicted label. Extract the label as a single number (0, 1, or 2) without any explanation or additional text. If the label is missing, return 'error'.

    LLM Response: "{llm_response}" """

def causal_detection_extraction_prompt(llm_response: str):
    prompt = f"""Given the following output from a language model, extract the entire list of tokens. The allowed tokens are 'O', 'I-CAUSE', 'B-CAUSE', 'I-EFFECT', and 'B-EFFECT'.
                The list should only contain these tokens and should be enclosed in brackets. Each token should be a string and surrounded by quotations ('').
                Extract all tokens that were found and output them in the exact order they were originally written. Only output tokens from the input, do not add any tokens. If no tokens were found, output an empty list.
                Only output a list of tokens enclosed in brackets, do not include any additional text or formatting.
                Response: {llm_response}"""
    return prompt
    
    
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

def finqa_evaluate_answer(predicted_answer: str, correct_answer: str):
    prompt = f"""
    You will receive two answers. Your job is to evaluate if they are exactly the same, with some caveats. 
    If they are wholly different answers (eg: 8 and 9), they are considered different.
    If the first answer is a more precise version of the second answer (eg: units listed, more decimal points reported, etc), they are the same.
    If the first answer can be rounded to the second answer, with the exact level of precision that the second answer uses, they are considered the same. If they cannot, they are different.
    If the answers are numbers and the first number cannot be rounded to the second number, respond with 'different'.
    For example, if the first answer is '1.02' and the second answer is '1', they are considered the same,
    but if the second answer is '1.02' and the first answer is '1.03' or '1', they are considered different.
    If the first answer is '5%' and the second answer is '5', they are considered the same.
    If the answers are the same, respond with 'correct'. If they are different, respond with 'wrong'.
    First answer: {predicted_answer}. Second answer: {correct_answer}
    """
    return prompt

finred_possible_relationships = [
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


def headlines_extraction_prompt(llm_response: str):
    """Generate a prompt to extract the relevant information from the LLM response."""
    prompt = f"""Extract the relevant information from the following LLM response and provide a score of 0 or 1 for each attribute based on the content. Format your output as a JSON object with these keys:
    - "Price_or_Not"
    - "Direction_Up"
    - "Direction_Down"
    - "Direction_Constant"
    - "Past_Price"
    - "Future_Price"
    - "Past_News"
    Only output the keys and values in the JSON object. Do not include any additional text.
    LLM Response:
    "{llm_response}" """
    return prompt

def refind_extraction_prompt(llm_response: str):
    """Construct the extraction prompt."""
    prompt = f"""Extract the classification label from the following LLM response. The label should be one of the following: ‘PERSON-TITLE’, ‘PERSON-GOV_AGY’, ‘PERSON-ORG’, ‘PERSON-UNIV’, ‘ORG-ORG’, ‘ORG-MONEY’, ‘ORG-GPE’, or ‘ORG-DATE’. List ‘NO-REL’ if the LLM did not output a clear answer.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response, exactly as it is listed in the approved label list, with a dash (-) between words. Only output alphanumeric characters, spaces, dashes, and underscores. Do not include any special characters, quotations, or punctuation. Only output the label."""
    return prompt


# Function to create the extraction prompt
def fiqa_1_extraction_prompt(llm_response: str):
    prompt = f"""
    You are tasked with extracting the sentiment score from a response. 
    The sentiment score should be a single numeric value between -1 and 1.

    Model Response: {llm_response}

    Provide only the numerical sentiment score as the output.
    """
    return prompt