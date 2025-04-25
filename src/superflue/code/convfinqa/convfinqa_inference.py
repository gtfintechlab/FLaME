import time
from datetime import date
import pandas as pd
from datasets import load_dataset
from superflue.code.prompts_zeroshot import convfinqa_prompt
from superflue.code.tokens import tokens
from litellm import completion 
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Set up logger
logger = setup_logger(
    name="convfinqa_inference", log_file=LOG_DIR / "convfinqa_inference.log", level=LOG_LEVEL
)

def convfinqa_inference(args):
    today = date.today()
    dataset = load_dataset("gtfintechlab/convfinqa", trust_remote_code=True)
    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    for entry in dataset["train"]:  # type: ignore
        pre_text = " ".join(entry["pre_text"])  # type: ignore
        post_text = " ".join(entry["post_text"])  # type: ignore
        table_text = " ".join([" ".join(map(str, row)) for row in entry["table_ori"]])  # type: ignore
        question_0 = str(entry["question_0"]) if entry["question_0"] is not None else ""  # type: ignore
        question_1 = str(entry["question_1"]) if entry["question_1"] is not None else ""  # type: ignore
        answer_0 = str(entry["answer_0"]) if entry["answer_0"] is not None else ""  # type: ignore
        answer_1 = str(entry["answer_1"]) if entry["answer_1"] is not None else ""  # type: ignore

        combined_text = f"{pre_text} {post_text} {table_text} Question 0: {question_0} Answer: {answer_0}. Now answer the following question: {question_1}"
        context.append(combined_text)
        actual_label = entry["answer_1"]  # type: ignore
        actual_labels.append(actual_label)

        try:
            model_response = completion(
                messages=[{"role": "user", "content": convfinqa_prompt(combined_text)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
            )
            
            # Log and process the response
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content # type: ignore
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(e)
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(20.0)
    
    df = pd.DataFrame(
        {
            "context": context,
            "response": llm_responses,
            "actual_label": actual_labels,
            "complete_responses": complete_responses,
        }
    )
    results_path = (
        RESULTS_DIR
        / "convfinqa"
        / f"{args.dataset}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    return df
