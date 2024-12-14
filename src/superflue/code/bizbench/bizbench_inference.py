import time
from datetime import date

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from superflue.code.prompts import bizbench_prompt

# from superflue.code.tokens import tokens
from superflue.utils.logging_utils import get_logger
from superflue.utils.path_utils import get_inference_path

from together import Together

# Get logger for this module
logger = get_logger(__name__)


def bizbench_inference(args):
    today = date.today()
    logger.info(f"Starting BizBench inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("glennmatlin/bizbench", trust_remote_code=True)

    # Initialize lists to store actual labels and model responses
    X_question = []
    X_context = []
    y_answer = []
    llm_responses = []
    complete_responses = []

    logger.info("Starting inference on dataset...")
    # start_t = time.time()
    client = Together()

    # Iterating through the test split of the dataset
    for i in tqdm(range(len(dataset["test"])), desc="Processing sentences"):  # type: ignore
        instance = dataset["test"][i]  # type: ignore
        """
        instance = {
            'question': __,
            'answer': __,
            'task': __,
            'context': __,
            'context_type': __,
            'options': __, (all rows are null)
            'program': __
        }
        """
        question = instance["question"]
        answer = instance["answer"]
        context = instance["context"]

        # ignore all instances where context is None
        if not context:
            continue

        try:
            logger.info(f"Processing instance {i+1}/{len(dataset['test'])}")  # type: ignore
            X_question.append(question)
            X_context.append(context)
            y_answer.append(answer)

            model_response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": bizbench_prompt(question, context)}
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model)
            )
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content  # type: ignore
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error processing instance {i+1}: {e}")
            time.sleep(20.0)
            continue

    df = pd.DataFrame(
        {
            "X_question": X_question,
            "X_context": X_context,
            "y_answer": y_answer,
            "llm_responses": llm_responses,
            "complete_responses": complete_responses,
        }
    )

    results_path = get_inference_path(args.dataset, args.model)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    return df
