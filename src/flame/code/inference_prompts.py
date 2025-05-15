def headlines_prompt(sentence: str):
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


def fiqa_task1_prompt(sentence: str):
    prompt = f"""You are a financial sentiment analysis expert. Analyze the provided sentence, identify relevant target aspects (such as companies, products, or strategies), and assign a sentiment score for each target. 
                The sentiment score should be between -1 (highly negative) and 1 (highly positive), using up to three decimal places to capture nuances in sentiment.

                Financial sentence:
                {sentence}"""
    return prompt


def fiqa_task2_prompt(question: str):
    prompt = f"""
    You are a financial analysis expert tasked with answering opinion-based financial questions. Your answer should be drawn from a broad corpus of structured and unstructured financial data sources, such as microblogs, reports, and news articles. 

    Carefully analyze the given question and identify:
    - Relevant financial entities (e.g., companies, products, indexes)
    - Key aspects (e.g., market trends, corporate strategies, economic indicators)
    - Sentiment polarity (positive, neutral, or negative)
    - Opinion holders (e.g., analysts, companies, general public sentiment)

    Use this information to provide a precise and contextually relevant answer that reflects the financial opinions expressed in the data. Answer in a concise manner, focusing on the opinions and insights that directly address the question.

    Financial Question:
    {question}
    """
    return prompt


def edtsum_prompt(document: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert at summarization tasks.	
        You are given a text that consists of multiple sentences. Your task is to perform abstractive summarization 
        on this text. Use your understanding of the content to express the main ideas and crucial details in a shorter, coherent, and natural sounding text.
        \nThe text:\n{document}.\nOutput your concise summary below. Try to keep your summary to one sentence and a maximum of 50 words, preferably around 25 words."""
    return prompt


def fomc_prompt(sentence: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence clas-
                sifier. Classify the following sentence from FOMC into ‘HAWKISH’, ‘DOVISH’, or ‘NEU-
                TRAL’ class. Label ‘HAWKISH’ if it is corresponding to tightening of the monetary policy,
                ‘DOVISH’ if it is corresponding to easing of the monetary policy, or ‘NEUTRAL’ if the
                stance is neutral. Provide the label in the first line and provide a short explanation in the
                second line. This is the sentence: {sentence}"""

    return prompt


def finer_prompt(sentence: str):
    system_prompt = """Discard all the previous instructions. Behave like you are an expert named entity
                    identifier. """
    user_msg = f"""Below a sentence is tokenized and each list item contains a word token from the
                    sentence. Identify ‘Person’, ‘Location’, and ‘Organisation’ from them and label them. If the
                    entity is multi token use post-fix_B for the first label and _I for the remaining token labels
                    for that particular entity. The start of the separate entity should always use _B post-fix for
                    the label. If the token doesn’t fit in any of those three categories or is not a named entity
                    label it ‘Other’. Do not combine words yourself. Use a colon to separate token and label.
                    So the format should be token:label. \n\n + {sentence} """

    # prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""
    prompt = system_prompt + user_msg

    return prompt


def fpb_prompt(sentence: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence clas-
                    sifier. Classify the following sentence into ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’
                    class. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
                    corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral. Provide
                    the label in the first line and provide a short explanation in the second line. This is the sentence: {sentence}"""

    return prompt


def finentity_prompt(sentence: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert entity recognizer and sentiment classifier. Identify the entities which are companies or organizations from the following content and classify the sentiment of the corresponding entities into ‘Neutral’ ‘Positive’ or ‘Negative’ classes. Considering every paragraph as a String in Python, provide the entities with the start and end index to mark the boundaries of it including spaces and punctuation using zero-based indexing. In the output, 
    Tag means sentiment; value means entity name. If no entity is found in the paragraph, 
    the response should be empty. Only give the output, not python code. The output should be a list that looks like:
    [{{'end': 210,
   'label': 'Neutral',
   'start': 207,
   'tag': 'Neutral',
   'value': 'FAA'}},
  {{'end': 7, 'label': 'Neutral', 'start': 4, 'tag': 'Neutral', 'value': 'FAA'}},
  {{'end': 298,
   'label': 'Neutral',
   'start': 295,
   'tag': 'Neutral',
   'value': 'FAA'}},
  {{'end': 105,
   'label': 'Neutral',
   'start': 99,
   'tag': 'Neutral',
   'value': 'Boeing'}}]
   Do not repeat any JSON object in the list. Evey JSON object should be unique.
   The paragraph:
                {sentence}"""
    return prompt


def finbench_prompt(profile: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expect risk assessor.
                Classify the following individual as either ‘LOW RISK’ or ‘HIGH RISK’ for approving a loan for. 
                Categorize the person as ‘HIGH RISK’ if their profile indicates that they will likely default on 
                the loan and not pay it back, and ‘LOW RISK’ if it is unlikely that they will fail to pay the loan back in full.
                Provide the label in the first line and provide a short explanation in the second line. Explain how you came to your classification decision and output the label that you chose. Do not write any code, simply think and provide your decision.
                Here is the information about the person:\nProfile data: {profile}\nPredict the risk category of this person:
                """
    return prompt


def ectsum_prompt(document: str):
    prompt = f"""Discard all the previous instructions.
        Behave like you are an expert at summarization tasks.
        Below an earnings call transcript of a Russell 3000 Index company
        is provided. Perform extractive summarization followed by
        paraphrasing the transcript in bullet point format according to the
        experts-written short telegram-style bullet point summaries
        derived from corresponding Reuters articles. The target length of
        the summary should be at most 50 words. \n\n The document:
        {document}"""

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


def banking77_prompt(sentence: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert at
                fine-grained single-domain intent detection. From the following list: {banking77_list}, identify
                which category the following sentence belongs to.
                {sentence}"""

    return prompt


def finqa_prompt(document: str):
    prompt = f"""Discard all the previous instructions. Behave like you are a financial expert in question answering. 
                Your task is to answer a financial question based on the  provided context.\n\n The context:
                {document}. Repeat you final answer at the end of your response. """

    return prompt


def convfinqa_prompt(document: str):
    prompt = f"""
    Discard all previous instructions. You are a financial expert specializing in answering questions.
    The context provided includes a previous question and its answer, followed by a new question that you need to answer.
    Focus on answering only the final question based on the entire provided context:
    {document}.
    Answer the final question based on the context above. Repeat your final answer at the end of your response. 
    """
    return prompt


def tatqa_prompt(document: str):
    prompt = f"""Discard all previous instructions. Behave like an expert in table-and-text-based financial question answering.
                Your task is to answer a question by extracting relevant information from both tables and text 
                provided in the context. Ensure that you use both sources comprehensively to generate an accurate response. Repeat your final answer at the
                end of your response. 
                
                The context: {document}"""

    return prompt


def causal_classification_prompt(text: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert causal classification model.
    Below is a sentence. Classify it into one of the following categories: 
                    0 - Non-causal
                    1 - Direct causal
                    2 - Indirect causal
                    Only return the label number without any additional text. \n\n {text}"""

    return prompt


possible_relationships = [
    "product or material produced",
    "manufacturer",
    "distributed by",
    "industry",
    "position held",
    "original broadcaster",
    "owned by",
    "founded by",
    "distribution format",
    "headquarters location",
    "stock exchange",
    "currency",
    "parent organization",
    "chief executive officer",
    "director/manager",
    "owner of",
    "operator",
    "member of",
    "employer",
    "chairperson",
    "platform",
    "subsidiary",
    "legal form",
    "publisher",
    "developer",
    "brand",
    "business division",
    "location of formation",
    "creator",
]


def finred_prompt(sentence: str, entity1: str, entity2: str):
    prompt = f"""Classify what relationship {entity2} (the head) has to {entity1} (the tail) within the following sentence:
    "{sentence}"
    
    The relationship should match one of the following categories, where the relationship is what the head entity is to the tail entity:
    {", ".join(possible_relationships)}.

    You must output one, and only one, relationship out of the previous list that connects the head entity {entity2} to the tail entity {entity1}. Find what relationship best fits {entity2} 'RELATIONSHIP' {entity1} for this sentence.
    """
    return prompt


def causal_detection_prompt(tokens: list):
    prompt = f"""You are an expert in detecting cause and effect phrases in text.
    You are given the following tokenized sentence. For each token, assign one of these labels:
        - 'B-CAUSE': The first token of a cause phrase.
        - 'I-CAUSE': A token inside a cause phrase, but not the first token.
        - 'B-EFFECT': The first token of an effect phrase.
        - 'I-EFFECT': A token inside an effect phrase, but not the first token.
        - 'O': A token that is neither part of a cause nor an effect phrase.
        
    Return only the list of labels in the same order as the tokens, without additional commentary or repeating the tokens themselves. 

    Tokens: {", ".join(tokens)}"""

    return prompt


def subjectiveqa_prompt(feature, definition, question, answer):
    system_prompt = """Discard all the previous instructions. Behave like you are an expert named entity
                    identifier. """
    user_msg = f"""Given the following feature: {feature} and its corresponding definition: {definition}\n
              Give the answer a rating of:\n
              2: If the answer positively demonstrates the chosen feature, with regards to the question.\n
              1: If there is no evident/neutral correlation between the question and the answer for the feature.\n
              0: If the answer negatively correlates to the question on the chosen feature.\n
              Provide the rating only. No explanations. This is the question: {question} and this is the answer: {answer}."""

    prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""

    return prompt


def fnxl_prompt(sentence, company, doc_type):
    """
    Prompt that instructs the LLM to:
      - Extract ALL numerals from the sentence
      - Assign each numeral to the most appropriate XBRL tag or 'other'.
      - Return a JSON object mapping 'numeral_string' -> 'xbrl_tag'.

    Example JSON output:
    {
      "7.2": "us-gaap:SomeExpenseTag",
      "9.0": "us-gaap:SomeExpenseTag",
      "2.5": "other"
    }
    """
    prompt = f"""
    You are an SEC reporting expert. Given a sentence from a financial filing, do two things:
    1) Identify every numeral in the sentence.
    2) For each numeral, assign the most appropriate US-GAAP XBRL tag based on context. 
    If no tag is appropriate, label it as "other".

    Return only valid JSON in this format:
    ```json
    {{
    "12.0": "us-gaap:Revenue",
    "9.5": "us-gaap:SomeExpense",
    "100.0": "other"
    }}```
    The sentnce is: {sentence}"""

    return prompt


def refind_prompt(entities):
    relations = "PERSON/TITLE - person subject, title object, relation title\nPERSON/GOV_AGY - person subject, government agency object, relation member_of\nPERSON/UNIV - person subject, university object, relation employee_of, member_of, attended\nPERSON/ORG - person subject, organization object, relation employee_of, member_of, founder_of\nORG/DATE - organization subject, date object, relation formed_on, acquired_on\nORG/MONEY - organization subject, money object, relation revenue_of, profit_of, loss_of, cost_of\nORG/GPE - organization subject, geopolitical entity object, relation headquartered_in, operations_in, formed_in\nORG/ORG - organization subject, organization object, relation shares_of, subsidiary_of, acquired_by, agreement_with"
    prompt = f"Classify the following relationship between ENT1 (the subject) and ENT2 (the object). The entities are marked by being enclosed in [ENT1] and [/EN1] and [ENT2] and [/ENT2] respectively. The subject entity will either be a person (PER) or an organization (ORG). The possible relationships are as follows, with the subject listed first and object listed second:\n{relations}\nText about entities: {entities}"
    return prompt


def numclaim_prompt(sentence: str) -> str:
    """Prompt for sentence claim classification."""
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence senti-
            ment classifier. Classify the following sentence into ‘INCLAIM’, or ‘OUTOFCLAIM’ class.
            Label ‘INCLAIM’ if consist of a claim and not just factual past or present information, or
            ‘OUTOFCLAIM’ if it has just factual past or present information. Provide the label in the
            first line and provide a short explanation in the second line. The sentence:{sentence}"""

    return prompt


prompt_map = {
    "numclaim_prompt": numclaim_prompt,
    "fiqa_task1_prompt": fiqa_task1_prompt,
    "fiqa_task2_prompt": fiqa_task2_prompt,
    "fomc_prompt": fomc_prompt,
    "finer_prompt": finer_prompt,
    "fpb_prompt": fpb_prompt,
    "finentity_prompt": finentity_prompt,
    "ectsum_prompt": ectsum_prompt,
    "edtsum_prompt": edtsum_prompt,
    "banking77_prompt": banking77_prompt,
    "finqa_prompt": finqa_prompt,
    "convfinqa_prompt": convfinqa_prompt,
    "tatqa_prompt": tatqa_prompt,
    "finred_prompt": finred_prompt,
    "causal_detection_prompt": causal_detection_prompt,
    "finbench_prompt": finbench_prompt,
    "refind_prompt": refind_prompt,
}


def prompt_function(prompt_name):
    return prompt_map.get(prompt_name, None)
