"""
Zero-Shot Prompt Functions

This module contains zero-shot prompt functions for various FLaME tasks.
These prompts do not include examples and expect the model to perform the task
based solely on instructions.
"""

from .registry import register_prompt, PromptFormat
from .constants import BANKING77_CATEGORIES, FINRED_RELATIONSHIPS


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
    prompt = f"""Classify the following Federal Reserve statement as HAWKISH (indicating a restrictive monetary policy stance), DOVISH (indicating an accommodative monetary policy stance), or NEUTRAL (indicating a balanced monetary policy stance).
    Statement: {sentence}
    Provide only one word as your answer: HAWKISH, DOVISH, or NEUTRAL."""
    return prompt


@register_prompt("bizbench", PromptFormat.ZERO_SHOT)
def bizbench_zeroshot_prompt(question: str, context: str) -> str:
    """Generate a zero-shot prompt for the BizBench task."""
    prompt = f"""Discard all previous instructions. You are an expert financial data extractor. 
                Extract the answer to the following question from the provided SEC filing context.
                Provide the answer with just the number without any units or other text.

                Question: {question}
                Context: {context}"""
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
    # TODO: Decompose this prompt into different registered prompts
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
    prompt = f"""Discard all the previous instructions. Behave like you are an expert at
                fine-grained single-domain intent detection. From the following list: {BANKING77_CATEGORIES}, identify
                which category the following sentence belongs to.
                {sentence}"""

    return prompt


@register_prompt("fiqa_task1", PromptFormat.ZERO_SHOT)
def fiqa_task1_zeroshot_prompt(sentence: str) -> str:
    """Generate a zero-shot prompt for the FiQA Task 1 (sentiment analysis).

    This prompt asks the model to analyze financial sentences, identify target aspects,
    and assign sentiment scores for each target.

    Args:
        sentence: The financial sentence to analyze

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a financial sentiment analysis expert. Analyze the provided sentence, identify relevant target aspects (such as companies, products, or strategies), and assign a sentiment score for each target. 
                The sentiment score should be between -1 (highly negative) and 1 (highly positive), using up to three decimal places to capture nuances in sentiment.

                Financial sentence:
                {sentence}"""
    return prompt


@register_prompt("fiqa_task2", PromptFormat.ZERO_SHOT)
def fiqa_task2_zeroshot_prompt(question: str) -> str:
    """Generate a zero-shot prompt for the FiQA Task 2 (opinion-based QA).

    This prompt asks the model to answer opinion-based financial questions
    based on financial data sources.

    Args:
        question: The financial question to answer

    Returns:
        Formatted prompt string
    """
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


@register_prompt("finer", PromptFormat.ZERO_SHOT)
def finer_zeroshot_prompt(sentence: str) -> str:
    """Generate a zero-shot prompt for the Finer (named entity recognition) task.

    This prompt asks the model to identify Person, Location, and Organisation entities
    in tokenized text and label them with appropriate tags.

    Args:
        sentence: The tokenized sentence to analyze

    Returns:
        Formatted prompt string
    """
    system_prompt = """Discard all the previous instructions. Behave like you are an expert named entity
                    identifier. """
    user_msg = f"""Below a sentence is tokenized and each list item contains a word token from the
                    sentence. Identify 'Person', 'Location', and 'Organisation' from them and label them. If the
                    entity is multi token use post-fix_B for the first label and _I for the remaining token labels
                    for that particular entity. The start of the separate entity should always use _B post-fix for
                    the label. If the token doesn't fit in any of those three categories or is not a named entity
                    label it 'Other'. Do not combine words yourself. Use a colon to separate token and label.
                    So the format should be token:label. \n\n + {sentence} """

    prompt = system_prompt + user_msg
    return prompt


@register_prompt("econlogicqa", PromptFormat.ZERO_SHOT)
def econlogicqa_prompt(question: str, A: str, B: str, C: str, D: str) -> str:
    """Generate prompt for EconLogicQA task.

    Args:
        question: The ordering question
        A: First event
        B: Second event
        C: Third event
        D: Fourth event

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all previous instructions. You are a Financial Event Analyst.
                You are given a question and 4 events, labeled A,B,C,D.
                You need to return an order of events as requested by the question. It could be a chronological order, order of importance, or a logical sequence.
                Question: {question}
                Events:
                A: {A}
                B: {B}
                C: {C}
                D: {D}
                Please provide the order of events, followed by a short explanation of the reasoning for the ordering. Output only the four labels (e.g., "C, B, A, D") in the expected order on the first line, and then briefly explain your reasoning in the next lines"""
    return prompt


@register_prompt("finentity", PromptFormat.ZERO_SHOT)
def finentity_zeroshot_prompt(sentence: str) -> str:
    """Generate a zero-shot prompt for the FinEntity task.

    This prompt asks the model to identify company/organization entities and classify
    their sentiment in financial text, providing entity boundaries and sentiment labels.

    Args:
        sentence: The paragraph to analyze

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all the previous instructions. Behave like you are an expert entity recognizer and sentiment classifier. Identify the entities which are companies or organizations from the following content and classify the sentiment of the corresponding entities into 'Neutral' 'Positive' or 'Negative' classes. Considering every paragraph as a String in Python, provide the entities with the start and end index to mark the boundaries of it including spaces and punctuation using zero-based indexing. In the output, 
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


@register_prompt("finbench", PromptFormat.ZERO_SHOT)
def finbench_zeroshot_prompt(profile: str) -> str:
    """Generate a zero-shot prompt for the FinBench task.

    This prompt asks the model to classify a loan applicant as high or low risk
    based on their profile information.

    Args:
        profile: The applicant profile data

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all the previous instructions. Behave like you are an expect risk assessor.
                Classify the following individual as either 'LOW RISK' or 'HIGH RISK' for approving a loan for. 
                Categorize the person as 'HIGH RISK' if their profile indicates that they will likely default on 
                the loan and not pay it back, and 'LOW RISK' if it is unlikely that they will fail to pay the loan back in full.
                Provide the label in the first line and provide a short explanation in the second line. Explain how you came to your classification decision and output the label that you chose. Do not write any code, simply think and provide your decision.
                Here is the information about the person:\nProfile data: {profile}\nPredict the risk category of this person:
                """
    return prompt


@register_prompt("ectsum", PromptFormat.ZERO_SHOT)
def ectsum_zeroshot_prompt(document: str) -> str:
    """Generate a zero-shot prompt for the ECTSum task.

    This prompt asks the model to perform extractive summarization followed by
    paraphrasing on an earnings call transcript.

    Args:
        document: The earnings call transcript to summarize

    Returns:
        Formatted prompt string
    """
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


@register_prompt("finqa", PromptFormat.ZERO_SHOT)
def finqa_zeroshot_prompt(document: str) -> str:
    """Generate a zero-shot prompt for the FinQA task.

    This prompt asks the model to answer a financial question based on
    provided context.

    Args:
        document: The context and question to process

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all the previous instructions. Behave like you are a financial expert in question answering. 
                Your task is to answer a financial question based on the provided context.\n\n The context:
                {document}. Repeat you final answer at the end of your response. """

    return prompt


@register_prompt("convfinqa", PromptFormat.ZERO_SHOT)
def convfinqa_zeroshot_prompt(document: str) -> str:
    """Generate a zero-shot prompt for the ConvFinQA task.

    This prompt asks the model to answer a conversational financial question
    based on provided context including previous questions and answers.

    Args:
        document: The context containing previous Q&A and current question

    Returns:
        Formatted prompt string
    """
    prompt = f"""
    Discard all previous instructions. You are a financial expert specializing in answering questions.
    The context provided includes a previous question and its answer, followed by a new question that you need to answer.
    Focus on answering only the final question based on the entire provided context:
    {document}.
    Answer the final question based on the context above. Repeat your final answer at the end of your response. 
    """
    return prompt


@register_prompt("tatqa", PromptFormat.ZERO_SHOT)
def tatqa_zeroshot_prompt(document: str) -> str:
    """Generate a zero-shot prompt for the TATQA task.

    This prompt asks the model to answer a question by extracting information
    from both tables and text in the provided context.

    Args:
        document: The context with tables, text, and question

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all previous instructions. Behave like an expert in table-and-text-based financial question answering.
                Your task is to answer a question by extracting relevant information from both tables and text 
                provided in the context. Ensure that you use both sources comprehensively to generate an accurate response. Repeat your final answer at the
                end of your response. 
                
                The context: {document}"""

    return prompt


@register_prompt("causal_classification", PromptFormat.ZERO_SHOT)
def causal_classification_zeroshot_prompt(text: str) -> str:
    """Generate a zero-shot prompt for the Causal Classification task.

    This prompt asks the model to classify a sentence into non-causal,
    direct causal, or indirect causal categories.

    Args:
        text: The sentence to classify

    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all the previous instructions. Behave like you are an expert causal classification model.
    Below is a sentence. Classify it into one of the following categories: 
                    0 - Non-causal
                    1 - Direct causal
                    2 - Indirect causal
                    Only return the label number without any additional text. \n\n {text}"""

    return prompt


@register_prompt("finred", PromptFormat.ZERO_SHOT)
def finred_zeroshot_prompt(sentence: str, entity1: str, entity2: str) -> str:
    """Generate a zero-shot prompt for the FinRED task.

    This prompt asks the model to classify the relationship between two entities
    in a financial text into predefined relationship categories.

    Args:
        sentence: The sentence containing the entities
        entity1: The tail entity
        entity2: The head entity

    Returns:
        Formatted prompt string
    """
    prompt = f"""Classify what relationship {entity2} (the head) has to {entity1} (the tail) within the following sentence:
    "{sentence}"
    
    The relationship should match one of the following categories, where the relationship is what the head entity is to the tail entity:
    {", ".join(FINRED_RELATIONSHIPS)}.

    You must output one, and only one, relationship out of the previous list that connects the head entity {entity2} to the tail entity {entity1}. Find what relationship best fits {entity2} 'RELATIONSHIP' {entity1} for this sentence.
    """
    return prompt


@register_prompt("causal_detection", PromptFormat.ZERO_SHOT)
def causal_detection_zeroshot_prompt(tokens: list) -> str:
    """Generate a zero-shot prompt for the Causal Detection task.

    This prompt asks the model to label each token in a sentence as part of
    a cause phrase, effect phrase, or neither.

    Args:
        tokens: List of tokens from the sentence

    Returns:
        Formatted prompt string
    """
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


@register_prompt("subjectiveqa", PromptFormat.ZERO_SHOT)
def subjectiveqa_zeroshot_prompt(feature, definition, question, answer) -> str:
    """Generate a zero-shot prompt for the SubjectiveQA task.

    This prompt asks the model to rate an answer on how well it demonstrates
    a specific feature in relation to a question.

    Args:
        feature: The feature to evaluate
        definition: The definition of the feature
        question: The question being answered
        answer: The answer to evaluate

    Returns:
        Formatted prompt string
    """
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


@register_prompt("fnxl", PromptFormat.ZERO_SHOT)
def fnxl_zeroshot_prompt(sentence, company=None, doc_type=None) -> str:
    """Generate a zero-shot prompt for the FNXL task.

    This prompt asks the model to extract numerals from financial text and
    assign appropriate XBRL tags to each numeral.

    Args:
        sentence: The financial sentence to analyze
        company: Optional company context (not used in current implementation)
        doc_type: Optional document type context (not used in current implementation)

    Returns:
        Formatted prompt string
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


@register_prompt("refind", PromptFormat.ZERO_SHOT)
def refind_zeroshot_prompt(entities: str) -> str:
    """Generate a zero-shot prompt for the REFinD task.

    This prompt asks the model to classify relationships between entities
    in financial text.

    Args:
        entities: Text with marked entities to analyze

    Returns:
        Formatted prompt string
    """
    relations = "PERSON/TITLE - person subject, title object, relation title\nPERSON/GOV_AGY - person subject, government agency object, relation member_of\nPERSON/UNIV - person subject, university object, relation employee_of, member_of, attended\nPERSON/ORG - person subject, organization object, relation employee_of, member_of, founder_of\nORG/DATE - organization subject, date object, relation formed_on, acquired_on\nORG/MONEY - organization subject, money object, relation revenue_of, profit_of, loss_of, cost_of\nORG/GPE - organization subject, geopolitical entity object, relation headquartered_in, operations_in, formed_in\nORG/ORG - organization subject, organization object, relation shares_of, subsidiary_of, acquired_by, agreement_with"
    prompt = f"Classify the following relationship between ENT1 (the subject) and ENT2 (the object). The entities are marked by being enclosed in [ENT1] and [/EN1] and [ENT2] and [/ENT2] respectively. The subject entity will either be a person (PER) or an organization (ORG). The possible relationships are as follows, with the subject listed first and object listed second:\n{relations}\nText about entities: {entities}"
    return prompt
