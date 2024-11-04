import time
from datetime import date
import pandas as pd
from datasets import load_dataset

import together
from superflue.together_code.prompts import headlines_prompt
from superflue.together_code.tokens import tokens
from superflue.utils.logging_utils import setup_logger
from superflue.config import RESULTS_DIR, LOG_DIR, LOG_LEVEL

# Setup logger for Headlines inference
logger = setup_logger(
    name="headlines_inference",
    log_file=LOG_DIR / "headlines_inference.log",
    level=LOG_LEVEL,
)

# Initialize the Together client
client = together.Together()

def headlines_inference(args):
    today = date.today()
    logger.info(f"Starting Headlines inference on {today}")

    # Load the Headlines dataset (test split with specific config)
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Headlines", '5768', trust_remote_code=True)

    results_path = (
        RESULTS_DIR
        / "headlines"
        / f"headlines_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize lists to store news, model responses, and labels
    news = []
    llm_responses = []
    complete_responses = []
    price_or_not_list = []
    direction_up_list = []
    direction_down_list = []
    direction_constant_list = []
    past_price_list = []
    future_price_list = []
    past_news_list = []

    logger.info(f"Starting inference on Headlines with model {args.model}...")

    # Iterate through the test split of the dataset
    for i in range(len(dataset["test"])):  # type: ignore
        sentence = dataset["test"][i]["News"]  # Extract news (sentence) # type: ignore
        price_or_not = dataset["test"][i]["PriceOrNot"]  # Extract price or not # type: ignore
        direction_up = dataset["test"][i]["DirectionUp"]  # Extract direction up # type: ignore
        direction_down = dataset["test"][i]["DirectionDown"]  # Extract direction down # type: ignore
        direction_constant = dataset["test"][i]["DirectionConstant"]  # Extract direction constant # type: ignore
        past_price = dataset["test"][i]["PastPrice"]  # Extract past price # type: ignore
        future_price = dataset["test"][i]["FuturePrice"]  # Extract future price # type: ignore
        past_news = dataset["test"][i]["PastNews"]  # Extract past news # type: ignore

        # Append to respective lists
        news.append(sentence)
        price_or_not_list.append(price_or_not)
        direction_up_list.append(direction_up)
        direction_down_list.append(direction_down)
        direction_constant_list.append(direction_constant)
        past_price_list.append(past_price)
        future_price_list.append(future_price)
        past_news_list.append(past_news)

        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}")  # type: ignore
            # Generate the model's response using Together API
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": headlines_prompt(sentence)}],
                tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            # Append the model response and complete response for the sentence
            complete_responses.append(model_response)
            response_text = model_response.choices[0].message.content.strip()  # type: ignore
            llm_responses.append(response_text)

            logger.info(f"Model response for sentence {i+1}: {response_text}")

        except Exception as e:
            # Log the error and retry the same sentence after a delay
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(10.0)
            complete_responses.append(None)
            llm_responses.append(None)
            continue  # Proceed to the next sentence after sleeping

    # Create the final DataFrame after the loop
    df = pd.DataFrame(
        {
            "news": news,
            "llm_responses": llm_responses,
            "price_or_not": price_or_not_list,
            "direction_up": direction_up_list,
            "direction_down": direction_down_list,
            "direction_constant": direction_constant_list,
            "past_price": past_price_list,
            "future_price": future_price_list,
            "past_news": past_news_list,
            "complete_responses": complete_responses,
        }
    )

    # Save the results to a CSV file
    df.to_csv(results_path, index=False)
    logger.info(f"Inference completed. Results saved to {results_path}")

    return df