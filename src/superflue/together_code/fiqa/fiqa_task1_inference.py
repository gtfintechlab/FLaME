import time
import pandas as pd
from datetime import date
from datasets import load_dataset
from litellm import completion 
from superflue.together_code.prompts import fiqa_task1_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Set up logger
logger = setup_logger(
    name="fiqa_task1_inference",
    log_file=LOG_DIR / "fiqa_task1_inference.log",
    level=LOG_LEVEL,
)

def fiqa_task1_inference(args):
    # Load dataset and initialize storage for results
    dataset = load_dataset("gtfintechlab/FiQA_Task1", split="test", trust_remote_code=True)
    context = []
    llm_responses = []
    actual_targets = []
    actual_sentiments = []
    complete_responses = []
    client = Together()

    for entry in dataset:
        # Extract relevant fields
        sentence = entry["sentence"]  # type: ignore
        snippets = entry["snippets"]  # type: ignore
        target = entry["target"]  # type: ignore
        sentiment_score = entry["sentiment_score"]  # type: ignore

        # Combine context and target information
        combined_text = f"Sentence: {sentence}. Snippets: {snippets}. Target aspect: {target}"
        context.append(combined_text)
        actual_targets.append(target)
        actual_sentiments.append(sentiment_score)

        try:
            model_response = completion(
                messages=[{"role": "user", "content": fiqa_task1_prompt(combined_text)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            # Process and log response
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content # type: ignore
            llm_responses.append(response_label)
            print(response_label)

            # Create DataFrame with results
            df = pd.DataFrame(
                {
                    "context": context,
                    "llm_responses": llm_responses,
                    "actual_target": actual_targets,
                    "actual_sentiment": actual_sentiments,
                    "complete_responses": complete_responses,
                }
            )

            # Save results intermittently
            time.sleep(10)
            results_path = (
                RESULTS_DIR
                / 'fiqa1/fiqa1_meta-llama/'
                / f"{'fiqa_task1'}_{'llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(results_path, index=False)

        except Exception as e:
            logger.error(f"Error encountered: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)

    logger.info(f"Inference completed. Results saved to {results_path}")
    return df
