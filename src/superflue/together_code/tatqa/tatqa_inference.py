import logging
import time
from datetime import date
from pathlib import Path
import together

import nltk
import pandas as pd
from datasets import load_dataset

# Mock imports for custom TATQA prompt and tokens
from superflue.together_code.prompts import tatqa_prompt  # To be implemented for TAT-QA prompt
from superflue.together_code.tokens import tokens  # Token logic for TAT-QA

nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def tatqa_inference(args):
    today = date.today()
    logger.info(f"Starting TAT-QA inference on {today}")

    logger.info("Loading dataset...")
    # Replace with the appropriate Hugging Face dataset for TAT-QA
    dataset = load_dataset("gtfintechlab/TATQA", token=args.hf_token)

    # Initialize lists to store the question, context, actual answers, and model responses
    questions = []
    contexts = []
    actual_answers = []
    model_responses = []

    logger.info(f"Starting inference on {args.task}...")
    start_t = time.time()
    for i in range(len(dataset["test"])):
        question = dataset["test"][i]["query"]
        context = dataset["test"][i]["text"]
        actual_answer = dataset["test"][i]["answer"]

        questions.append(question)
        contexts.append(context)
        actual_answers.append(actual_answer)
        
        try:
            logger.info(f"Processing question {i+1}/{len(dataset['test'])}")
            # TAT-QA-specific prompt logic, create the prompt for table and text-based QA
            model_response = together.Complete.create(
                prompt=tatqa_prompt(question, context),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            model_responses.append(model_response["output"]["choices"][0]["text"])

            df = pd.DataFrame(
                {
                    "questions": questions,
                    "contexts": contexts,
                    "actual_answers": actual_answers,
                    "model_responses": model_responses,
                }
            )

        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
            time.sleep(20.0)
            continue

    results_path = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df
