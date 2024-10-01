import time
import pandas as pd
from datetime import date
from datasets import load_dataset
from superflue.together_code.tokens import tokens
import together
from superflue.together_code.prompts import fiqa_task2_prompt
from superflue.utils.logging_utils import setup_logger

from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="fiqa_task2_inference",
    log_file=LOG_DIR / "fiqa_task2_inference.log",
    level=LOG_LEVEL,
)

# TODO: (Glenn) Is FiQA saving results to a file properly?


def fiqa_task2_inference(args):
    dataset = load_dataset(
        "gtfintechlab/FiQA_Task2", split="test", trust_remote_code=True
    )

    context = []
    llm_responses = []
    actual_answers = []
    complete_responses = []

    

    for entry in dataset:
        question = entry["question"] # type: ignore
        context.append(question)
        actual_answer = entry["answer"] # type: ignore

    
        actual_answers.append(actual_answer)
        
        try:
            model_response = together.Complete.create(
                prompt=fiqa_task2_prompt(question),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            complete_responses.append(model_response)
            response_label = model_response["choices"][0]["text"]
            llm_responses.append(response_label)
           
            df = pd.DataFrame(
                {
                    "question": context,
                    "llm_responses": llm_responses,
                    "actual_answers": actual_answers,
                    "complete_responses": complete_responses,
                }
            )

            results_path = (
                RESULTS_DIR
                / 'fiqa2/fiqa2_meta-llama/'
                / f"{'fiqa_task2'}_{'llama-3.1-8b'}_{date.today().strftime('%d_%m_%Y')}.csv"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(results_path, index=False)

        except Exception as e:
            print(f"Error encountered: {e}")
            complete_responses.append(None)
            llm_responses.append(None)

            time.sleep(10.0)

    return df