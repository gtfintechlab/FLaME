import pandas as pd
import logging
from datetime import date
from pathlib import Path
from litellm import completion
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.utils.path_utils import get_evaluation_save_path
from superflue.config import LOG_DIR, LOG_LEVEL

# Configure logging
logger = setup_logger(
    name="finred_evaluation",
    log_file=LOG_DIR / "finred_evaluation.log",
    level=LOG_LEVEL,
)

# Define possible relationships
possible_relationships = [
    'subsidiary', 'owned_by', 'employer', 'product_or_material_produced', 'industry',
    'manufacturer', 'developer', 'legal_form', 'parent_organization', 'distribution_format',
    'chairperson', 'location_of_formation', 'headquarters_location', 'operator', 'creator',
    'currency', 'founded_by', 'original_broadcaster', 'owner_of', 'director_/_manager',
    'business_division', 'chief_executive_officer', 'position_held', 'platform', 'brand',
    'distributed_by', 'publisher', 'stock_exchange', 'member_of'
]

def extraction_prompt(llm_response: str):
    """Generate a prompt to extract the classification label from the LLM response."""
    relationship_choices = ', '.join(possible_relationships)
    prompt = f'''Extract the classification label from the following LLM response. The label should be one of the following {relationship_choices}. 
    
                Pick the label out of the list that is the closest to the LLM response, but list 'NO-REL' if the LLM did not output a clear answer.
                
                Here is the LLM response to analyze:
                "{llm_response}"
                Provide only the label that best matches the response, exactly as it is listed above. Only output alphanumeric characters, spaces, dashes, and underscores. Do not include any special characters, quotations, or punctuation.'''
    return prompt

def finred_evaluate(file_name, args):
    """Evaluate FinRED dataset and return results and metrics DataFrames."""
    task = args.dataset.strip('"""')
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    # Load CSV
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    # Define paths using consistent utility
    evaluation_results_path = get_evaluation_save_path(args.dataset, args.model)
    evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    correct_labels = df["actual_label"].tolist()
    extracted_labels = []

    for i, llm_response in tqdm(enumerate(df["llm_responses"]), total=len(df["llm_responses"])):
        if pd.notna(df.at[i, "extracted_labels"]):
            continue

        try:
            response = completion(
                model=args.model,
                messages=[{"role": "user", "content": extraction_prompt(llm_response)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            extracted_label = response.choices[0].message.content.strip() # type: ignore

            # Normalize and validate extracted label
            extracted_label = extracted_label.replace(' ', '')
            if extracted_label not in possible_relationships:
                extracted_label = 'NO-REL'

            extracted_labels.append(extracted_label)
            df.at[i, "extracted_labels"] = extracted_label
            df.to_csv(evaluation_results_path, index=False)

        except Exception as e:
            logger.error(f"Error processing response {i}: {e}")
            extracted_labels.append('NO-REL')

    # Calculate metrics
    accuracy = accuracy_score(correct_labels, extracted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        correct_labels, extracted_labels, average="weighted"
    )

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # Create metrics DataFrame with consistent format
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1],
    })

    # Save metrics using consistent naming
    metrics_path = evaluation_results_path.with_name(f"{evaluation_results_path.stem}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    return df, metrics_df
