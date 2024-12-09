from typing import Dict, Tuple, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from ferrari.code.tokens import tokens
from litellm import completion
import re
from ferrari.config import EVALUATION_DIR, LOG_DIR, LOG_LEVEL
from ferrari.utils.logging_utils import setup_logger
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Setup logger
logger = setup_logger(
    name="fiqa_task1_evaluation",
    log_file=LOG_DIR / "fiqa_task1_evaluation.log",
    level=LOG_LEVEL,
)

def extraction_prompt(llm_response: str) -> str:
    """Create a prompt for extracting sentiment scores.
    
    Args:
        llm_response: Raw response from the language model
        
    Returns:
        Formatted prompt for sentiment score extraction
    """
    prompt = f"""
    You are tasked with extracting the sentiment score from a response. 
    The sentiment score should be a single numeric value between -1 and 1.
    -1 represents extremely negative sentiment
    0 represents neutral sentiment
    1 represents extremely positive sentiment

    Model Response: {llm_response}

    Provide only the numerical sentiment score as the output.
    """
    return prompt

def extract_numerical_value(text: str) -> Optional[float]:
    """Extract a numerical sentiment score from text.
    
    Args:
        text: Text containing a sentiment score
        
    Returns:
        Extracted sentiment score or None if invalid
    """
    try:
        match = re.search(r"(-?\d+\.?\d*)", text)
        if match:
            value = float(match.group(0))
            # Validate sentiment score range
            if -1 <= value <= 1:
                return value
            else:
                logger.warning(f"Sentiment score {value} outside valid range [-1, 1]")
        return None
    except Exception as e:
        logger.error(f"Error extracting numerical value: {e}")
        return None

def validate_input_data(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required columns.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ["llm_responses", "actual_sentiment"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def calculate_sentiment_metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    """Calculate metrics for sentiment analysis task.
    
    Args:
        actual: List of actual sentiment scores
        predicted: List of predicted sentiment scores
        
    Returns:
        Dictionary containing calculated metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(actual, predicted)[0, 1]
    
    # Calculate directional accuracy (sign agreement)
    sign_actual = np.sign(actual)
    sign_pred = np.sign(predicted)
    directional_accuracy = np.mean(sign_actual == sign_pred)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": correlation,
        "Directional_Accuracy": directional_accuracy
    }

def fiqa_task1_evaluate(file_name: str, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate FiQA Task 1 sentiment analysis results.
    
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

    # Output path for evaluation results
    evaluation_results_path = (
        EVALUATION_DIR
        / task
        / f"evaluation_{task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    extraction_response: List[Optional[str]] = []
    extraction_model_response: List[Any] = []
    extracted_scores: List[Optional[float]] = []
    
    # Process each response
    for i, entry in enumerate(df["llm_responses"]):
        if pd.isna(entry):
            extraction_response.append(None)
            extraction_model_response.append(None)
            extracted_scores.append(None)
            continue
            
        try:
            model_response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(entry)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
            )
            extraction_model_response.append(model_response)
            response_text = model_response.choices[0].message.content  # type: ignore
            extraction_response.append(response_text)
            score = extract_numerical_value(response_text)
            extracted_scores.append(score)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extraction_response.append(None)
            extracted_scores.append(None)
            extraction_model_response.append(str(e))
            time.sleep(10.0)

    # Add results to DataFrame
    df['extraction_model_response'] = extraction_model_response
    df['extraction_response'] = extraction_response
    df['extracted_score'] = extracted_scores

    # Filter valid responses for metric calculation
    valid_indices = [i for i, score in enumerate(extracted_scores) if score is not None]
    if not valid_indices:
        logger.error("No valid sentiment scores extracted for evaluation")
        raise ValueError("No valid sentiment scores for evaluation")

    valid_extracted = [extracted_scores[i] for i in valid_indices]  # type: ignore
    valid_actual = [df['actual_sentiment'][i] for i in valid_indices]

    # Calculate metrics
    metrics = calculate_sentiment_metrics(valid_actual, valid_extracted)
    
    # Log metrics
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info(f"Valid responses: {len(valid_indices)}/{len(df)} ({len(valid_indices)/len(df)*100:.1f}%)")

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        "Metric": list(metrics.keys()) + ["Valid_Responses"],
        "Value": list(metrics.values()) + [len(valid_indices)],
    })

    # Save results
    df.to_csv(evaluation_results_path, index=False)
    logger.info(f"Results saved to {evaluation_results_path}")

    # Save metrics
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df