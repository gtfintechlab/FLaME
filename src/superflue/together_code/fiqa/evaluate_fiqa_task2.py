import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


file_path = '/Users/yangyang/Desktop/SuperFLUE/src/superflue/results/evaluation_results/fiqa2/evaluation_fiqa2_meta-llama/Llama-2-7b-chat-hf_06_10_2024.csv'
df = pd.read_csv(file_path)

def calculate_ndcg(df, llm_col, actual_col, k=10):
  
  
    vectorizer = TfidfVectorizer()
    llm_responses_tfidf = vectorizer.fit_transform(df[llm_col])
    actual_answers_tfidf = vectorizer.transform(df[actual_col])
    
    
    cosine_similarities = cosine_similarity(llm_responses_tfidf, actual_answers_tfidf)
    
  
    def dcg_at_k(relevance_scores, k):
        return np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k])])
    

    ndcg_scores = []
    for idx in range(len(df)):
       
        sorted_relevances = np.sort(cosine_similarities[idx])[::-1]
        
 
        dcg = dcg_at_k(sorted_relevances, k)
        
       
        ideal_relevances = np.sort(cosine_similarities[idx])[::-1]
        idcg = dcg_at_k(ideal_relevances, k)
        
      
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    
    return np.mean(ndcg_scores)


average_ndcg = calculate_ndcg(df, 'llm_responses', 'actual_answers')
print(average_ndcg)
