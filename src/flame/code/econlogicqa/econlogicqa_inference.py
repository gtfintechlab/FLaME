from datetime import date

import pandas as pd
from datasets import load_dataset
from together import Together
from tqdm import tqdm

from flame.config import LOG_DIR, LOG_LEVEL
from flame.code.prompts import get_prompt, PromptFormat
from flame.code.tokens import tokens
from flame.utils.logging_utils import setup_logger

logger = setup_logger(
    name="econlogicqa_inference",
    log_file=LOG_DIR / "econlogicqa_inference.log",
    level=LOG_LEVEL,
)


def econlogicqa_inference(args):
    today = date.today()
    logger.info(f"Starting EconLogicQA inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("glennmatlin/econlogicqa", trust_remote_code=True)["test"]

    # Apply sample size limit if specified
    if hasattr(args, "sample_size") and args.sample_size is not None:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
        logger.info(f"Limited dataset to {len(dataset)} samples")

    # Initialize Together API client
    client = Together()

    # Retrieve the zero-shot prompt from the registry
    econlogicqa_prompt = get_prompt("econlogicqa", PromptFormat.ZERO_SHOT)

    if econlogicqa_prompt is None:
        raise RuntimeError("EconLogicQA prompt not found in registry")

    responses_ordered_importance = []
    for i in tqdm(range(len(dataset)), desc="Accessing EconLogicQA"):
        row = dataset[i]
        question = row["Question"]
        event_a = row["A"]
        event_b = row["B"]
        event_c = row["C"]
        event_d = row["D"]

        prompt = econlogicqa_prompt(question, event_a, event_b, event_c, event_d)
        if i == 10:
            print(prompt)
        try:
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )
            llm_response = model_response.choices[0].message.content
            ordered_response = llm_response.splitlines()[0]
        except Exception as e:
            if i == 10:
                print("Error " + str(e))
            ordered_response = "Error"
            llm_response = f"Error: {str(e)}"

        row["llm_responses"] = ordered_response
        row["llm_complete_responses"] = llm_response
        responses_ordered_importance.append(row)

    output_df = pd.DataFrame(responses_ordered_importance)

    return output_df
