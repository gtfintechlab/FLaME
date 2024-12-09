import time
import pandas as pd
from datasets import load_dataset
import together
from dotenv import dotenv_values
from ferrari.code.prompts import fiqa_prompt
from ferrari.code.tokens import tokens
from ferrari.utils.logging_utils import setup_logger
from ferrari.config import LOG_DIR, LOG_LEVEL
from huggingface_hub import login
from dotenv import dotenv_values
from tqdm import tqdm

logger = setup_logger(
    name="fiqa_task1_inference",
    log_file=LOG_DIR / "fiqa_task1_inference.log",
    level=LOG_LEVEL,
)

HF_ORGANIZATION = "glennmatlin"
DATASET = "FiQA_Task1"

def hg_fiqa_inference(args):
    config = dotenv_values(".env")
    token = config.get("HUGGINGFACEHUB_API_TOKEN")
    together_api_key = config.get("TOGETHER_API_KEY")
    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")
    if not together_api_key:
        raise ValueError("TOGETHER_API_KEY not found in .env file")
    
    login(token)
    together.api_key = together_api_key

    dataset = load_dataset(
        "glennmatlin/FiQA_Task1", split="test", trust_remote_code=True
    )

    context = []
    llm_responses = []
    actual_targets = []
    actual_sentiments = []
    complete_responses = []

    # start_time = time.time()

    for entry in tqdm(dataset, desc="Processing sentences"):
        sentence = entry["sentence"]
        snippets = entry["snippets"]
        target = entry["target"]
        sentiment_score = entry["sentiment_score"]

        combined_text = f"Sentence: {sentence}. Snippets: {snippets}. Target aspect: {target}. What is the sentiment?"
        context.append(combined_text)

        actual_targets.append(target)
        actual_sentiments.append(sentiment_score)

        try:
            model_response = together.Complete.create(
                prompt=fiqa_prompt(combined_text),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            complete_responses.append(model_response)

            # Adjusted logic to access choices directly
            if "choices" in model_response and len(model_response["choices"]) > 0:
                response_label = model_response["choices"][0]["text"]
            else:
                print("Unexpected model response format. Model Response:", model_response)
                response_label = None

            llm_responses.append(response_label)

        except Exception as e:
            print(f"Error encountered: {e}")
            complete_responses.append(None)
            llm_responses.append(None)
            time.sleep(20.0)

    # Construct the DataFrame outside the loop
    df = pd.DataFrame({
        "context": context,
        "llm_responses": llm_responses,
        "actual_target": actual_targets,
        "actual_sentiment": actual_sentiments,
        "complete_responses": complete_responses,
    })

    return df
