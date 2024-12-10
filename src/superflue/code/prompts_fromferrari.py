"""Prompts for all Superflue tasks."""


def fiqa_task1_prompt(sentence: str) -> str:
    """Generate prompt for FiQA Task 1 sentiment analysis.
    
    Args:
        sentence: The sentence to analyze
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a financial sentiment analysis expert. Analyze the provided sentence, identify relevant target aspects (such as companies, products, or strategies), and assign a sentiment score for each target. 
                The sentiment score should be between -1 (highly negative) and 1 (highly positive), using up to three decimal places to capture nuances in sentiment.

                Financial sentence:
                {sentence}"""
    return prompt


def fiqa_task2_prompt(question: str) -> str:
    """Generate prompt for FiQA Task 2 opinion QA.
    
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


def fomc_prompt(sentence: str) -> str:
    """Generate prompt for FOMC sentiment classification.
    
    Args:
        sentence: The sentence to classify
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Classify the following Federal Reserve statement as HAWKISH (indicating a restrictive monetary policy stance), DOVISH (indicating an accommodative monetary policy stance), or NEUTRAL (indicating a balanced monetary policy stance).

Statement: {sentence}

Provide only one word as your answer: HAWKISH, DOVISH, or NEUTRAL."""
    return prompt


def bizbench_prompt(question: str, context: str) -> str:
    """Generate prompt for BizBench QA task.
    
    Args:
        question: The question to answer
        context: The SEC filing context
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Discard all previous instructions. You are an expert financial data extractor. 
                Extract the answer to the following question from the provided SEC filing context.
                Provide the answer with just the number without any units or other text.

                Question: {question}
                Context: {context}"""
    return prompt


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