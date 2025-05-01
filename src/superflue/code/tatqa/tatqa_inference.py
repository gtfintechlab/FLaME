import time
# from pathlib import Path
from litellm import completion 
import nltk
import pandas as pd
from datasets import load_dataset
from datetime import date
from superflue.code.prompts_zeroshot import tatqa_prompt
from superflue.code.prompts_fewshot import tatqa_fewshot_prompt
from superflue.code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="tatqa_inference", log_file=LOG_DIR / "tatqa_inference.log", level=LOG_LEVEL
)

def tatqa_inference(args):
    today = date.today()
    dataset = load_dataset("gtfintechlab/TATQA", trust_remote_code=True)
    
    # Initialize lists to store context, model responses, actual answers, and complete responses
    context = []
    llm_responses = []
    actual_answers = []
    complete_responses = []
    
    for entry in dataset["test"]:  # type: ignore
        question = entry["query"]  # type: ignore
        context_text = entry["text"]  # type: ignore
        combined_text = f"{context_text} {question}"  # Combine context and question
        context.append(combined_text)
        
        actual_answer = entry["answer"]  # type: ignore
        actual_answers.append(actual_answer)

        try:
            logger.info(f"Processing question {i+1}/{len(dataset['test'])}") # type: ignore
            # TAT-QA-specific prompt logic, create the prompt for table and text-based QA
            model_response = completion(
                messages=[{"role": "user", "content": tatqa_prompt(combined_text)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content # type: ignore
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error processing entry {len(context)}: {e}")
            llm_responses.append(None)
            complete_responses.append(None)
            time.sleep(20.0)
    
    df = pd.DataFrame(
        {
            "context": context,
            "response": llm_responses,
            "actual_answer": actual_answers,
            "complete_responses": complete_responses,
        }
    )
    
    time.sleep(10)
    results_path = (
        RESULTS_DIR
        / "tatqa"
        / f"{args.dataset}_{'llama-3.1-8b'}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    logger.info(f"Inference completed. Results saved to {results_path}")
    return df
