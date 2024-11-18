
def headlines_prompt(sentence: str):

        prompt = f'''Discard all the previous instructions. Behave like you are an expert at analyzing headlines.
                    Give a score of 0 for each of the following attributes if the news headline does not contain the following information or 1 if it does.
                    Price or Not: Does the news item talk about price or not.
                    Direction Up: Does the news headline talk about price going up or not?
                    Direction Down: Does the news headline talk about price going down or not?
                    Direction Constant: Does the news headline talk about price remaining constant or not?
                    Past Price: Does the news headline talk about an event in the past?
                    Future Price: Does the news headline talk about an event in the future?
                    Past News: Does the news headline talk about a general event (apart from prices) in the past?
                    The news headline is:
                    {sentence}'''

        return prompt


def fiqa_prompt(sentence: str):
    prompt = f"""Lorem ipsum: {sentence}"""
    return prompt


def fiqa_task1_prompt(sentence: str):
    prompt = f"""Lorem ipsum: {sentence}"""
    return prompt


def fiqa_task2_prompt(sentence: str):
    prompt = f"""Lorem ipsum: {sentence}"""
    return prompt


def edtsum_prompt(document: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert at summarization tasks.	
        You are given a text that consists of multiple sentences. Your task is to perform abstractive summarization 
        on this text. Use your understanding of the content to express the main ideas and crucial details in a shorter, coherent, and natural sounding text.
        \nThe text:\n{document}.\nOutput your concise summary below. Try to keep your summary to one sentence and a maximum of 50 words, preferably around 25 words."""
    return prompt


def numclaim_prompt(sentence: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence senti-
            ment classifier. Classify the following sentence into ‘INCLAIM’, or ‘OUTOFCLAIM’ class.
            Label ‘INCLAIM’ if consist of a claim and not just factual past or present information, or
            ‘OUTOFCLAIM’ if it has just factual past or present information. Provide the label in the
            first line and provide a short explanation in the second line. The sentence:{sentence}"""

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

    prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""

    return prompt


def fpb_prompt(sentence: str, prompt_format: str):
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence clas-
                    sifier. Classify the following sentence into ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’
                    class. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
                    corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral. Provide
                    the label in the first line and provide a short explanation in the second line. This is the sentence: {sentence}"""

    return prompt

    if prompt_format == "superflue":
        system_prompt = """ Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier"""

        user_msg = f""" Classify the following sentence into ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’
                    class. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
                    corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral. 
                    Provide the label in the first line and provide a short explanation in the second line.
                    Explain how you came to your classification decision. This is the sentence: {sentence}."""

    elif prompt_format == "finben_icl":
        system_prompt = """"""
        user_msg = f""" Analyze the sentiment of this statement extracted from a financial news article.
                        Provide your answer as either NEGATIVE, POSITIVE or NEUTRAL.
                        For instance, ’The company’s stocks plummeted following the scandal.’ would be classified as negative. This is the sentence: {sentence}"""

    elif prompt_format == "finben_noicl":
        system_prompt = """"""
        user_msg = f""" Analyze the sentiment of this statement extracted from a financial news article.
                        Provide your answer as either NEGATIVE, POSITIVE or NEUTRAL.
                        This is the sentence: {sentence}"""

    elif prompt_format == "superflue_icl":
        system_prompt = """Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier """
        user_msg = f""" Classify the following sentence into ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’
                        class. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
                        corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral. Provide
                        the label in the first line and provide a short explanation in the second line.
                        For instance: 
                        "According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing" would be classified as 'NEUTRAL.
                        "When this investment is in place , Atria plans to expand into the Moscow market" would be classified as 'NEUTRAL'.
                        "With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability" would be classified as 'POSITIVE'.
                        "For the last quarter of 2010 , Componenta's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m" would be classified as 'POSITIVE'.
                        "Aspocomp has a large factory in China and a factory building project in India that was halted due to financing problems" would be classified as 'NEGATIVE'.
                        "The low capacity utilisation rate in steel production considerably increases the fixed costs per unit of steel produced" would be classified as 'NEGATIVE'.
                        This is the sentence: {sentence}"""

    elif prompt_format == "superflue_cot":  # TODO modify this prompt text
        system_prompt = """Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier """
        user_msg = f""" Classify the following sentence into ‘NEGATIVE’, ‘POSITIVE’, or ‘NEUTRAL’
                        class. Label ‘NEGATIVE’ if it is corresponding to negative sentiment, ‘POSITIVE’ if it is
                        corresponding to positive sentiment, or ‘NEUTRAL’ if the sentiment is neutral. Let's think about this sentiment classification task step by step.
                        First, generate your reasoning steps for the classification. After your reasoning, end the response with the label that fits your reasoning.
                        This is the sentence: {sentence}"""

    prompt = f"""{system_prompt}\n{user_msg}"""
    print(prompt)

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
                {document}"""

    return prompt


def convfinqa_prompt(document: str):
    prompt = f"""Discard all the previous instructions. Behave like you are a financial expert in question answering.
                You are to answer a series of interconnected financial questions where later questions may depend on the answers to previous ones.
                I'll provide the series of questions as the context and you will answer the last question.\n\n The context:
                {document}"""

    return prompt


def tatqa_prompt(question: str, context: str):
    system_prompt = """Discard all the previous instructions. Behave like an expert in table-and-text-based question answering."""

    user_msg = f"""Given the following context (which contains a mixture of tables and textual information), 
                answer the question based on the information provided. If the context includes tables, ensure 
                you extract relevant information from both the table and the text to form a comprehensive answer.
                
                The question: {question}
                
                The context: {context}"""

    prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""

    return prompt


def causal_classification_prompt(text: str):
    system_prompt = """Discard all the previous instructions. Behave like you are an expert causal classification model."""
    user_msg = f"""Below is a sentence. Classify it into one of the following categories: 
                    0 - Non-causal
                    1 - Direct causal
                    2 - Indirect causal
                    Only return the label number without any additional text. \n\n {text}"""

    prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""

    return prompt

possible_relationships = ['product or material produced', 'manufacturer', 'distributed by', 'industry', 'position held', 'original broadcaster', 'owned by', 'founded by', 'distribution format', 'headquarters location', 'stock exchange', 'currency', 'parent organization', 'chief executive officer', 'director/manager', 'owner of', 'operator', 'member of', 'employer', 'chairperson', 'platform', 'subsidiary', 'legal form', 'publisher', 'developer', 'brand', 'business division', 'location of formation', 'creator']

def finred_prompt(sentence: str, entity1: str, entity2: str):
    prompt = f"""Classify what relationship {entity2} (the head) has to {entity1} (the tail) within the following sentence:
    "{sentence}"
    
    The relationship should match one of the following categories, where the relationship is what the head entity is to the tail entity:
    {', '.join(possible_relationships)}.

    You must output one, and only one, relationship out of the previous list that connects the head entity {entity2} to the tail entity {entity1}. Find what relationship best fits {entity2} 'RELATIONSHIP' {entity1} for this sentence.
    """
    return prompt

def causal_detection_prompt(tokens: list):
    """
    Generates a prompt for Causal Detection to classify tokens in a sentence into cause, effect, or other categories,
    with an explanation of the B- and I- labeling scheme.

    Args:
        tokens (list): The list of tokens from a sentence to be classified.

    Returns:
        str: The formatted prompt for Causal Detection classification.
    """

    system_prompt = """Discard all previous instructions. Behave like an expert in cause and effect detection in text."""

    user_msg = f"""You are given the following tokenized sentence. Classify each token using the following labels:
                - 'B-CAUSE': The beginning of a cause phrase.
                - 'I-CAUSE': A token inside a cause phrase, but not the first token.
                - 'B-EFFECT': The beginning of an effect phrase.
                - 'I-EFFECT': A token inside an effect phrase, but not the first token.
                - 'O': A token that is neither part of a cause nor an effect.
                
                Provide the classification for each token in the format 'token:label'.
                
                The tokens: {', '.join(tokens)}"""

    prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""

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


def fnxl_prompt(sentence, numerals, company, doc_type):
    """
    Generates a prompt for the LLM to associate numerals with the correct XBRL tags.
    
    Parameters:
    - sentence (str): The sentence containing financial numerals.
    - numerals (list): List of numerals to be tagged.
    - company (str): The name of the company (for additional context).
    - doc_type (str): The document type (e.g., "10-K").

    Returns:
    - str: The prompt to be passed to the LLM.
    """
    # Example prompt format
    prompt = f"""
You are a financial assistant skilled in SEC reporting. Your task is to analyze sentences containing financial numerals and associate each numeral with its corresponding financial XBRL tag based on the context. 
Give the most accurate tag for that particular numeral.
Below is an example:

**Example Input**:
- Sentence: "The Operating Partnership incurred expenses pursuant to the Corporate Services Agreement for the years ended December 31, 2020, 2019 and 2018 of $3.5 million, $3.5 million and $1.9 million, respectively."
- Numerals: [3.5, 3.5, 1.9]
- Metadata:
  - Company: MGM Growth Properties LLC
  - Document Type: 10-K
- Expected Output:
  ```json
  {{"3.5": "us-gaap:RelatedPartyTransactionSellingGeneralAndAdministrativeExpensesFromTransactionsWithRelatedParty", "1.9": "us-gaap:RelatedPartyTransactionSellingGeneralAndAdministrativeExpensesFromTransactionsWithRelatedParty"}}
  
  Input:
  Sentence: {sentence}
  Numerals: {numerals}
  Company: {company}
  Document Type: {doc_type}"""
  
    return prompt

def refind_prompt(entities):
    relations = "PERSON/TITLE - person subject, title object, relation title\nPERSON/GOV_AGY - person subject, government agency object, relation member_of\nPERSON/UNIV - person subject, university object, relation employee_of, member_of, attended\nPERSON/ORG - person subject, organization object, relation employee_of, member_of, founder_of\nORG/DATE - organization subject, date object, relation formed_on, acquired_on\nORG/MONEY - organization subject, money object, relation revenue_of, profit_of, loss_of, cost_of\nORG/GPE - organization subject, geopolitical entity object, relation headquartered_in, operations_in, formed_in\nORG/ORG - organization subject, organization object, relation shares_of, subsidiary_of, acquired_by, agreement_with"
    prompt = f"Classify the following relationship between ENT1 (the subject) and ENT2 (the object). The entities are marked by being enclosed in [ENT1] and [/EN1] and [ENT2] and [/ENT2] respectively. The subject entity will either be a person (PER) or an organization (ORG). The possible relationships are as follows, with the subject listed first and object listed second:\n{relations}\nText about entities: {entities}"
    return prompt

prompt_map = {
    "numclaim_prompt": numclaim_prompt,
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
    'finbench_prompt': finbench_prompt,
    'refind_prompt': refind_prompt
}


def prompt_function(prompt_name):
    return prompt_map.get(prompt_name, None)
