from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import date
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet
from rouge_score import rouge_scorer
from ferrari.utils.logging_utils import setup_logger
from ferrari.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL

# Set up logger
logger = setup_logger(
    name="fiqa_task2_evaluation",
    log_file=LOG_DIR / "fiqa_task2_evaluation.log",
    level=LOG_LEVEL,
)

def validate_input_data(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required columns.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ["llm_responses", "actual_answers"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def calculate_qa_metrics(response: str, reference: str) -> Dict[str, float]:
    """Calculate various QA metrics between response and reference.
    
    Args:
        response: Model's response text
        reference: Reference answer text
        
    Returns:
        Dictionary containing calculated metrics
    """
    try:
        # Tokenize texts
        response_tokens = word_tokenize(response.lower())
        reference_tokens = word_tokenize(reference.lower())
        
        # Calculate BLEU score
        bleu = sentence_bleu([reference_tokens], response_tokens)
        
        # Calculate METEOR score
        meteor = meteor_score([reference_tokens], response_tokens)
        
        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, response)
        
        return {
            "BLEU": bleu,
            "METEOR": meteor,
            "ROUGE-1": rouge_scores['rouge1'].fmeasure,
            "ROUGE-2": rouge_scores['rouge2'].fmeasure,
            "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        }
    except Exception as e:
        logger.error(f"Error calculating QA metrics: {e}")
        return {
            "BLEU": 0.0,
            "METEOR": 0.0,
            "ROUGE-1": 0.0,
            "ROUGE-2": 0.0,
            "ROUGE-L": 0.0,
        }

def calculate_ranking_metrics(df: pd.DataFrame, llm_col: str, actual_col: str, k: int = 10) -> Dict[str, float]:
    """Calculate ranking-based metrics using TF-IDF and cosine similarity.
    
    Args:
        df: DataFrame containing responses and references
        llm_col: Column name for model responses
        actual_col: Column name for reference answers
        k: Top-k value for ranking metrics
        
    Returns:
        Dictionary containing calculated metrics
    """
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform responses and answers to TF-IDF representations
    llm_responses_tfidf = vectorizer.fit_transform(df[llm_col].fillna(''))
    actual_answers_tfidf = vectorizer.transform(df[actual_col].fillna(''))
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(llm_responses_tfidf, actual_answers_tfidf)
    
    # Calculate DCG@k
    def dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
        return float(np.sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k])]))
    
    ndcg_scores = []
    reciprocal_ranks = []
    binary_relevance = []
    
    for idx in range(len(df)):
        # Calculate NDCG
        sorted_relevances = np.sort(cosine_similarities[idx])[::-1]
        dcg = dcg_at_k(sorted_relevances, k)
        idcg = dcg_at_k(np.ones(k), k)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # Calculate MRR
        relevant_ranks = np.where(cosine_similarities[idx] >= 0.5)[0]
        reciprocal_ranks.append(1 / (relevant_ranks[0] + 1) if relevant_ranks.size > 0 else 0)
        
        # Calculate binary F-score
        binary_prediction = (cosine_similarities[idx] >= 0.5).astype(int)
        binary_truth = np.ones_like(binary_prediction)
        binary_relevance.append(f1_score(binary_truth[:k], binary_prediction[:k]))

    return {
        "NDCG": float(np.mean(ndcg_scores)),
        "MRR": float(np.mean(reciprocal_ranks)),
        "F-score": float(np.mean(binary_relevance))
    }

def fiqa_task2_evaluate(file_name: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate FiQA Task 2 question answering results.
    
    Args:
        file_name: Path to the CSV file containing LLM responses
        args: Arguments containing model configuration
        
    Returns:
        Tuple containing (results DataFrame, metrics DataFrame)
        
    Raises:
        ValueError: If input validation fails
    """
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}...")

    # Load and validate input data
    try:
        df = pd.read_csv(file_name)
        logger.info(f"Loaded {len(df)} rows from {file_name}.")
        validate_input_data(df)
    except Exception as e:
        logger.error(f"Error loading or validating input file: {e}")
        raise

    # Calculate metrics for each response
    qa_metrics_list = []
    valid_responses = 0
    
    for i, (response, reference) in enumerate(zip(df['llm_responses'], df['actual_answers'])):
        if pd.isna(response) or pd.isna(reference):
            qa_metrics_list.append({
                "BLEU": 0.0,
                "METEOR": 0.0,
                "ROUGE-1": 0.0,
                "ROUGE-2": 0.0,
                "ROUGE-L": 0.0,
            })
            continue
            
        metrics = calculate_qa_metrics(response, reference)
        qa_metrics_list.append(metrics)
        valid_responses += 1

    # Add QA metrics to DataFrame
    for metric in ["BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L"]:
        df[f"metric_{metric}"] = [m[metric] for m in qa_metrics_list]

    # Calculate ranking metrics
    ranking_metrics = calculate_ranking_metrics(df, 'llm_responses', 'actual_answers')
    
    # Combine all metrics
    all_metrics = {}
    # Average QA metrics
    for metric in ["BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L"]:
        all_metrics[metric] = float(np.mean([m[metric] for m in qa_metrics_list]))
    # Add ranking metrics
    all_metrics.update(ranking_metrics)
    # Add response stats
    all_metrics["Valid_Responses"] = valid_responses
    all_metrics["Response_Rate"] = valid_responses / len(df)
    
    # Log metrics
    for metric, value in all_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Metric": list(all_metrics.keys()),
        "Value": list(all_metrics.values()),
    })

    # Save results
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Results saved to {evaluation_results_path}")

    # Save metrics
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df 