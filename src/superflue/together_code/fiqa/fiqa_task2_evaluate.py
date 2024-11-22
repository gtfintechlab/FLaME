import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_metrics(df, llm_col, actual_col, k=10):
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform responses and answers to TF-IDF representations
    llm_responses_tfidf = vectorizer.fit_transform(df[llm_col])
    actual_answers_tfidf = vectorizer.transform(df[actual_col])
    
    # Calculate cosine similarities for each pair of LLM response and actual answer
    cosine_similarities = cosine_similarity(llm_responses_tfidf, actual_answers_tfidf)
    
    # Function to calculate DCG for a given relevance score list
    def dcg_at_k(relevance_scores, k):
        return np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k])])
    
    ndcg_scores = []
    reciprocal_ranks = []
    binary_relevance = []
    
    for idx in range(len(df)):
        # Sort relevances to get top-k relevance scores for NDCG and MRR
        sorted_relevances = np.sort(cosine_similarities[idx])[::-1]
        
        # Calculate DCG and IDCG for NDCG
        dcg = dcg_at_k(sorted_relevances, k)
        idcg = dcg_at_k(np.ones(k), k)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # Mean Reciprocal Rank (MRR)
        # Find the rank of the first relevant answer (thresholded at 0.5 for relevance)
        relevant_ranks = np.where(cosine_similarities[idx] >= 0.5)[0]
        if relevant_ranks.size > 0:
            reciprocal_ranks.append(1 / (relevant_ranks[0] + 1))
        else:
            reciprocal_ranks.append(0)
        
        # F-score Calculation (using binary relevance based on a threshold)
        # We threshold cosine similarities to get binary relevance scores
        binary_prediction = (cosine_similarities[idx] >= 0.5).astype(int)
        binary_truth = np.ones_like(binary_prediction)
        binary_relevance.append(f1_score(binary_truth[:k], binary_prediction[:k]))

    # Calculate average metrics
    avg_ndcg = np.mean(ndcg_scores)
    avg_mrr = np.mean(reciprocal_ranks)
    avg_fscore = np.mean(binary_relevance)

    return avg_fscore, avg_ndcg, avg_mrr

def fiqa_task2_evaluate(file_name, args):
    # Load data
    df = pd.read_csv(file_name)
    # Calculate and print metrics
    avg_fscore, avg_ndcg, avg_mrr = calculate_metrics(df, 'llm_responses', 'actual_answers')
    print(f"Average F-score: {avg_fscore:.4f}")
    print(f"Average NDCG: {avg_ndcg:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")
    metrics_df = pd.DataFrame(
        {
            "Average F-score": [avg_fscore],
            "Average NDCG": [avg_ndcg],
            "Average MRR": [avg_mrr],
        }
    )
    return df, metrics_df