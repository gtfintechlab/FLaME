import time
import pandas as pd
from datasets import load_dataset
from litellm import completion
from superflue.code.prompts import fiqa_task2_prompt
from superflue.utils.logging_utils import get_logger
from superflue.utils.path_utils import get_inference_path

# Get logger for this module
logger = get_logger(__name__)


def fiqa_task2_inference(args):
    # Load dataset and initialize lists for results
    dataset = load_dataset(
        "gtfintechlab/FiQA_Task2", split="test", trust_remote_code=True
    )
    context = []
    llm_responses = []
    actual_answers = []
    complete_responses = []

    for entry in dataset:
        # Extract question and actual answer
        question = entry["question"]  # type: ignore
        actual_answer = entry["answer"]  # type: ignore
        context.append(question)
        actual_answers.append(actual_answer)

        try:
            model_response = completion(
                messages=[{"role": "user", "content": fiqa_task2_prompt(question)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=tokens(args.model)
            )

            # Process and store the response
            logger.debug(f"Model response: {model_response}")
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content  # type: ignore
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error encountered: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(10.0)

    # Save results intermittently
    df = pd.DataFrame(
        {
            "context": context,
            "llm_responses": llm_responses,
            "actual_answer": actual_answers,
            "complete_responses": complete_responses,
        }
    )

    results_path = get_inference_path(args.dataset, args.model)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)

    return df
