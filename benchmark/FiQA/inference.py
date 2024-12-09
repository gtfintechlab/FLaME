import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import json
import pandas as pd

# Load the BERT model and tokenizer
model_name = "bert-base-uncased"  # Using BERT instead of Llama
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def load_custom_json_dataset():
    try:
        # Load both JSON files
        headline_data = load_dataset("ambakick/FiQA", name="task1_headline_ABSA_test", split="test")
        post_data = load_dataset("ambakick/FiQA", name="task1_post_ABSA_test", split="test")
        
        # Convert datasets to pandas DataFrames
        headline_df = pd.DataFrame(headline_data)
        post_df = pd.DataFrame(post_data)
        
        # Combine the DataFrames
        combined_df = pd.concat([headline_df, post_df], ignore_index=True)
        
        # Normalize the 'info' column
        combined_df['info'] = combined_df['info'].apply(lambda x: x[0] if isinstance(x, list) else x)
        combined_df = pd.concat([combined_df.drop(['info'], axis=1), combined_df['info'].apply(pd.Series)], axis=1)
        
        # Rename columns for consistency
        combined_df = combined_df.rename(columns={'snippets': 'snippets', 'target': 'target'})
        
        return combined_df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Use the custom loading function
dataset = load_custom_json_dataset()

# Ensure you're using a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Task 1: Aspect-based Financial Sentiment Analysis
def task_1_inference(text, aspects_list):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # For demonstration, we'll use the raw output as a proxy for sentiment
    sentiment_score = outputs.logits.squeeze().item()
    
    aspect_sentiment = {}
    for aspect in aspects_list:
        aspect_sentiment[aspect] = {
            "sentiment_score": sentiment_score
        }
    
    return {
        "input_text": text,
        "generated_output": f"Sentiment score: {sentiment_score}",
        "aspect_sentiment": aspect_sentiment
    }

# Task 2: Opinion-based Question Answering over Financial Data
def task_2_inference(question, corpus):
    # Tokenize input question
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the raw output as a relevance score for demonstration
    relevance_score = outputs.logits.squeeze().item()
    
    # Simulate ranking answers from the corpus based on relevance
    ranked_answers = {}
    for idx, answer in enumerate(corpus):
        ranked_answers[idx] = {
            "answer_text": answer,
            "relevance_score": relevance_score + torch.rand(1).item() - 0.5  # Add some randomness
        }
    
    # Sort answers by relevance score (descending order)
    ranked_answers = sorted(ranked_answers.items(), key=lambda x: x[1]["relevance_score"], reverse=True)
    
    return {
        "question": question,
        "generated_output": f"Relevance score: {relevance_score}",
        "ranked_answers": ranked_answers[:10]  # Returning top 10 answers
    }

# Example usage
if __name__ == "__main__":
    if dataset is not None:
        # Task 1: Aspect-based Financial Sentiment Analysis
        aspects_list = ["Corporate/Strategy", "Product", "Stock", "AI"]
        
        for i, row in dataset.iterrows():
            input_text = row['sentence']
            result = task_1_inference(input_text, aspects_list)
            print(f"Task 1 - Input: {result['input_text']}")
            print(f"Task 1 - Generated Output: {result['generated_output']}")
            print(f"Task 1 - Aspect Sentiment: {result['aspect_sentiment']}")
            print("-" * 50)
            
            if i >= 5:  # Limit to first 5 samples for demonstration
                break
        
        # Task 2: Opinion-based Question Answering over Financial Data
        corpus = dataset['sentence'].tolist()[:100]  # Use first 100 sentences as corpus
        
        for i, row in dataset.iterrows():
            question = row['sentence']  # Using sentence as question for demonstration
            result = task_2_inference(question, corpus)
            print(f"Task 2 - Question: {result['question']}")
            print(f"Task 2 - Generated Output: {result['generated_output']}")
            print(f"Task 2 - Top 3 Ranked Answers: {result['ranked_answers'][:3]}")
            print("-" * 50)
            
            if i >= 5:  # Limit to first 5 samples for demonstration
                break
