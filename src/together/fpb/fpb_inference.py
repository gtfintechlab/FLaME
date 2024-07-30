import pandas as pd
import time
from datasets import load_dataset
from datetime import date
from src.together.prompts import fpb_prompt
from pathlib import Path
from src.together.models import get_model_name
from tqdm import tqdm
from src.utils.logging_utils import setup_logger
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT_DIR / "logs"
logger = setup_logger("fpb_inference", LOG_DIR / "fpb_inference.log")

BATCH_SIZE = 10  # Adjust this value based on API limitations and performance

def prepare_batch(data_points: List[Dict[str, Any]], args) -> List[str]:
    prompts = []
    for dp in data_points:
        try:
            prompt = fpb_prompt(sentence=dp["sentence"], prompt_format=args.prompt_format)
            prompts.append(prompt)
        except Exception as e:
            logger.error(f"Error preparing prompt for sentence: {dp['sentence']}. Error: {str(e)}")
            prompts.append(None)
    return prompts

def process_batch_response(batch_response: Dict[str, Any], data_points: List[Dict[str, Any]], task: str, model: str) -> List[Dict[str, Any]]:
    results = []
    for i, choice in enumerate(batch_response["output"]["choices"]):
        try:
            result = {
                "sentence": data_points[i]["sentence"],
                "actual_label": data_points[i]["label"],
                "llm_response": choice["text"],
                "complete_response": {
                    "task": task,
                    "model": model,
                    "response": choice,
                    "metadata": {
                        "timestamp": batch_response["output"]["created"]
                    }
                }
            }
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing response for sentence: {data_points[i]['sentence']}. Error: {str(e)}")
            results.append({
                "sentence": data_points[i]["sentence"],
                "actual_label": data_points[i]["label"],
                "llm_response": "error",
                "complete_response": str(e),
            })
    return results

def fpb_inference(args, make_api_call, process_api_response):
    import time
    total_time = 0
    total_batches = 0
    logger.info(f"Starting FPB inference on {date.today()}")
    configs = ["sentences_allagree"]
    
    all_results = []
    
    for config in configs:
        logger.info(f"Loading dataset for config: {config}")
        try:
            dataset = load_dataset("financial_phrasebank", config, token=args.hf_token)
        except Exception as e:
            logger.error(f"Error loading dataset for config {config}: {str(e)}")
            continue
        
        for i in tqdm(range(0, len(dataset["train"]), BATCH_SIZE), desc="Processing batches"):
            batch_start_time = time.time()
            batch = dataset["train"][i:i+BATCH_SIZE]
            prompts = prepare_batch(batch, args)
            
            if not any(prompts):
                logger.warning(f"All prompts in batch {i//BATCH_SIZE + 1} failed to prepare. Skipping batch.")
                continue
            
            try:
                model_response = make_api_call(
                    prompts=[p for p in prompts if p is not None],
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    stop=None,
                )
                
                batch_results = process_batch_response(model_response, batch, "fpb", args.model)
                all_results.extend(batch_results)
                
                process_api_response(batch_results, "fpb", args.model)
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                total_time += batch_time
                total_batches += 1
                logger.info(f"Processed batch {i//BATCH_SIZE + 1}, sentences {i+1}-{min(i+BATCH_SIZE, len(dataset['train']))}")
                logger.info(f"Batch processing time: {batch_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//BATCH_SIZE + 1}: {str(e)}")
                for dp in batch:
                    all_results.append({
                        "sentence": dp["sentence"],
                        "actual_label": dp["label"],
                        "llm_response": "error",
                        "complete_response": str(e),
                    })
            
            time.sleep(1)  # Rate limiting
    
    df = pd.DataFrame(all_results)
    logger.info(f"FPB inference completed. Total processed sentences: {len(df)}")
    if total_batches > 0:
        avg_time_per_batch = total_time / total_batches
        logger.info(f"Average time per batch: {avg_time_per_batch:.2f} seconds")
    
    return df
