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