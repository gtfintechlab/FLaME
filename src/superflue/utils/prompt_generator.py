"""Utility functions for generating prompts."""

def generate_prompt(task: str, text: str) -> str:
    """Generate a prompt for a given task.
    
    Args:
        task: The task name
        text: The input text
        
    Returns:
        Formatted prompt string
    """
    if task == "fomc":
        return f"""Classify the following Federal Reserve statement as HAWKISH (indicating a restrictive monetary policy stance), 
                DOVISH (indicating an accommodative monetary policy stance), or NEUTRAL (indicating a balanced monetary policy stance).
                
                Statement: {text}
                
                Provide only one word as your answer: HAWKISH, DOVISH, or NEUTRAL."""
    elif task == "fiqa":
        return f"""You are a financial sentiment analysis expert. Analyze the provided sentence, identify relevant target aspects 
                (such as companies, products, or strategies), and assign a sentiment score for each target.
                
                Financial sentence: {text}"""
    elif task == "bizbench":
        return f"""Extract the answer to the following question from the provided SEC filing context.
                Provide the answer with just the number without any units or other text.
                
                Question: {text}"""
    elif task == "econlogicqa":
        return f"""You are given a question and 4 events. Return an order of events as requested by the question.
                It could be a chronological order, order of importance, or a logical sequence.
                
                Question: {text}"""
    else:
        raise ValueError(f"Unknown task: {task}")
