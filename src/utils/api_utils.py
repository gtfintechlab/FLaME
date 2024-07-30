import backoff
import together
import json
from pathlib import Path
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def make_api_call(prompts: List[str], model: str, max_tokens: int, temperature: float, top_k: int, top_p: float, repetition_penalty: float, stop: List[str] = None) -> Dict[str, Any]:
    return together.Complete.create(
        prompt=prompts,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop=stop,
    )

def save_raw_output(results: List[Dict[str, Any]], task: str, model: str, output_dir: Path) -> None:
    raw_output_path = output_dir / task / model / f"raw_output_{results[0]['metadata']['timestamp']}.jsonl"
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(raw_output_path, 'a') as f:
        for result in results:
            json.dump(result, f)
            f.write('
')
    
    logger.info(f"Raw output saved to {raw_output_path}")
