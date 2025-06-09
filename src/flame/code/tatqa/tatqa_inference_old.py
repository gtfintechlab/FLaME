import pandas as pd
from flame.utils.dataset_utils import safe_load_dataset
from litellm import completion
from tqdm import tqdm

from flame.code.prompts import get_prompt, PromptFormat
from flame.utils.logging_utils import get_component_logger

# Use component-based logger that follows the logging configuration
logger = get_component_logger("inference", "tatqa")


def tatqa_inference(args):
    dataset = safe_load_dataset("gtfintechlab/TATQA", trust_remote_code=True)

    # Initialize lists to store context, model responses, actual answers, and complete responses
    context = []
    llm_responses = []
    actual_answers = []
    complete_responses = []

    tatqa_prompt = get_prompt("tatqa", PromptFormat.ZERO_SHOT)
    if tatqa_prompt is None:
        raise RuntimeError("TATQA prompt not found in registry")

    for i, entry in enumerate(tqdm(dataset["test"], desc="Processing TATQA examples")):  # type: ignore
        question = entry["query"]  # type: ignore
        context_text = entry["text"]  # type: ignore
        combined_text = f"{context_text} {question}"  # Combine context and question
        context.append(combined_text)

        actual_answer = entry["answer"]  # type: ignore
        actual_answers.append(actual_answer)

        try:
            logger.debug(f"Processing question {i + 1}/{len(dataset['test'])}")  # type: ignore
            # TAT-QA-specific prompt logic, create the prompt for table and text-based QA
            model_response = completion(
                messages=[{"role": "user", "content": tatqa_prompt(combined_text)}],
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content  # type: ignore
            llm_responses.append(response_label)

        except Exception as e:
            logger.error(f"Error processing entry {len(context)}: {e}")
            llm_responses.append(None)
            complete_responses.append(None)
            # time.sleep(20.0)  # Removed sleep for better performance
            pass

    df = pd.DataFrame(
        {
            "context": context,
            "response": llm_responses,
            "actual_answer": actual_answers,
            "complete_responses": complete_responses,
        }
    )

    return df
