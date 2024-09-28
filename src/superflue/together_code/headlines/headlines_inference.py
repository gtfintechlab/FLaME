import together
import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from superflue.together_code.tokens import tokens
from superflue.together_code.prompts import headlines_prompt
from superflue.config import PACKAGE_DIR
# TODO: use logging helper function here
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def headlines_inference(args):
    together.api_key = args.api_key
    today = date.today()
    logger.info(f"Starting Numclaim inference on {today}")

    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Headlines")
    
    # Initialize lists to store actual labels and model responses
    news = []
    complete_responses = []
    llm_responses = []
    price_or_not_list = []
    direction_up_list = []
    direction_down_list = []
    direction_constant_list = []
    past_price_list = []
    future_price_list = []
    past_news_list = []

    logger.info(f"Starting inference on {args.task}...")
    
    for i in range(len(dataset['test'])):
        time.sleep(5.0)
        sentence = dataset['test'][i]['News']
        price_or_not = dataset['test'][i]['PriceOrNot']
        direction_up = dataset['test'][i]['DirectionUp']
        direction_down = dataset['test'][i]['DirectionDown']
        direction_constant = dataset['test'][i]['DirectionConstant']
        past_price = dataset['test'][i]['PastPrice']
        future_price = dataset['test'][i]['FuturePrice']
        past_news = dataset['test'][i]['PastNews']
        
        news.append(sentence)
        price_or_not_list.append(price_or_not)
        direction_up_list.append(direction_up)
        direction_down_list.append(direction_down)
        direction_constant_list.append(direction_constant)
        past_price_list.append(past_price)
        future_price_list.append(future_price)
        past_news_list.append(past_news)
        
        try:
            logger.info(f"Processing sentence {i+1}/{len(dataset['test'])}")
            model_response = together.Complete.create(
                prompt=headlines_prompt(sentence),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model)
            )
            complete_responses.append(model_response)
            response_label = model_response["output"]["choices"][0]["text"]
            llm_responses.append(response_label)
            logger.info(f"Model response: {response_label}")
            time.sleep(10)

        except Exception as e:
            logger.error(f"Error processing sentence {i+1}: {e}")
            time.sleep(10.0)
            complete_responses.append(None)
            llm_responses.append(None)

        df = pd.DataFrame({
            'news': news,
            'complete_responses': complete_responses,
            'llm_responses': llm_responses,
            'price_or_not': price_or_not_list,
            'direction_up': direction_up_list,
            'direction_down': direction_down_list,
            'direction_constant': direction_constant_list,
            'past_price': past_price_list,
            'future_price': future_price_list,
            'past_news': past_news_list
        })
        results_path = PACKAGE_DIR / 'results' / args.task / f"{args.task}_{args.model}_{date.today().strftime('%d_%m_%Y')}.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        logger.info(f"Intermediate results saved to {results_path}")

    logger.info(f"Inference completed. Final results saved to {results_path}")
    return df