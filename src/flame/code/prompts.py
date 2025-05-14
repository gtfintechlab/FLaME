"""Prompts for all FLaME tasks, merged and updated."""


# Original Functions from Code File 1
def fiqa_task1_prompt(sentence: str) -> str:
    """Generate prompt for FiQA Task 1 sentiment analysis."""
    prompt = f"""You are a financial sentiment analysis expert. Analyze the provided sentence, identify relevant target aspects (such as companies, products, or strategies), and assign a sentiment score for each target. 
                The sentiment score should be between -1 (highly negative) and 1 (highly positive), using up to three decimal places to capture nuances in sentiment.

                Financial sentence:
                {sentence}"""
    return prompt


def fiqa_task2_prompt(question: str) -> str:
    """Generate prompt for FiQA Task 2 opinion QA."""
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


def fomc_prompt(sentence: str) -> str:
    """Generate prompt for FOMC sentiment classification."""
    prompt = f"""Classify the following Federal Reserve statement as HAWKISH (indicating a restrictive monetary policy stance), DOVISH (indicating an accommodative monetary policy stance), or NEUTRAL (indicating a balanced monetary policy stance).

Statement: {sentence}

Provide only one word as your answer: HAWKISH, DOVISH, or NEUTRAL."""
    return prompt


def bizbench_prompt(question: str, context: str) -> str:
    """Generate prompt for BizBench QA task."""
    prompt = f"""Discard all previous instructions. You are an expert financial data extractor. 
                Extract the answer to the following question from the provided SEC filing context.
                Provide the answer with just the number without any units or other text.

                Question: {question}
                Context: {context}"""
    return prompt


# Merged Function (Minor differences reconciled)
def fiqa_prompt(sentence: str) -> str:
    """Generate a basic FIQA prompt."""
    prompt = f"""You are a financial analysis expert. Analyze the provided sentence:
                {sentence}"""
    return prompt


# New Functions from Code File 2
def fnxl_prompt(sentence: str) -> str:
    """Prompt for FNXL task."""
    system_prompt = """You are an expert in financial text processing focused on numeric data tagging for financial documents."""
    user_msg = f"""Identify the numerical figures in the following financial text, and assign each a label according to the FNXL taxonomy. 
                Use one of these categories based on the context:
                - 0 (No special label needed for this numeral)
                - 1 (Rarely used numeral label for financial extremes)
                - Higher integers as appropriate based on the prominence or financial significance. 
                Output as a list of integers with the length matching the number of numerical figures identified in the text.
                
                Sentence: {sentence}"""

    prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {user_msg} [/INST]"""
    return prompt


def headlines_prompt(sentence: str) -> str:
    """Prompt for headline attribute scoring."""
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


def edtsum_prompt(document: str) -> str:
    """Prompt for EDTSUM summarization task."""
    prompt = f"""Discard all the previous instructions. Behave like you are an expert at summarization tasks.	
        You are given a text that consists of multiple sentences. Your task is to perform abstractive summarization 
        on this text. Use your understanding of the content to express the main ideas and crucial details in a shorter, coherent, and natural sounding text.
        \nThe text:\n{document}.\nOutput your concise summary below. Try to keep your summary to one sentence and a maximum of 50 words, preferably around 25 words."""
    return prompt


def numclaim_prompt(sentence: str) -> str:
    """Prompt for sentence claim classification."""
    prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence senti-
            ment classifier. Classify the following sentence into ‘INCLAIM’, or ‘OUTOFCLAIM’ class.
            Label ‘INCLAIM’ if consist of a claim and not just factual past or present information, or
            ‘OUTOFCLAIM’ if it has just factual past or present information. Provide the label in the
            first line and provide a short explanation in the second line. The sentence:{sentence}"""

    return prompt
