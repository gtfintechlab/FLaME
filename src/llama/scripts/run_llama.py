import sys
from pathlib import Path

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
SRC_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from time import time
from llama.instructions import llama2_prompt_generator
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from llama.instructions import TASK_MAP
from llama.pipeline import LlamaTextGenerationPipeline

from transformers.pipelines import TextGenerationPipeline

from utils.args import parse_args
from utils.config import SEEDS, TODAY
from utils.hf_model import get_hf_model
from utils.logging import setup_logger

logger = setup_logger(__name__)


def main(args):
    TASK_INSTRUCTION, TASK_DATA = (
        TASK_MAP[args.task_name]["instruction"],
        TASK_MAP[args.task_name]["data"],
    )

    # get model and tokenizer
    model, tokenizer = get_hf_model(args)

    # get pipeline ready for instruction text generation
    generation_pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        # NOTE: Set `do_sample = True` when `temperature > 0.0`
        # https://github.com/huggingface/transformers/issues/25326
        temperature=0.0,  # [0.0, 1.0]; 0.0 means greedy sampling
        do_sample=False,
        max_new_tokens=512,
        top_k=10,
        top_p=0.92,
        repetition_penalty=1.0,  # 1.0 means no penalty
        num_return_sequences=1,  # Only generate one response
    )

    for seed in tqdm(SEEDS):
        logger.info(f"Running inference for seed {seed}")

        # Assign seed to NumPy and PyTorch
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Setup directories and filepaths
        # TODO: I shouldn't need to make the data or test directory -- they should exist -- if they dont, throw an error!
        DATA_DIRECTORY = ROOT_DIRECTORY / "data"
        DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        TASK_DIRECTORY = DATA_DIRECTORY / args.task_name
        TASK_DIRECTORY.mkdir(parents=True, exist_ok=True)
        TEST_DIRECTORY = TASK_DIRECTORY / "test"
        TEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        PROMPT_OUTPUTS = TASK_DIRECTORY / "llm_prompt_outputs" / args.quantization
        PROMPT_OUTPUTS.mkdir(parents=True, exist_ok=True)

        # TODO: ask Agam if we should count the time it takes to load the model to GPU in our results
        # To me it makes more sense to measure loading time separately and then focus on inference time per sequence
        # Model loading time is a one-time cost, inference time is per sequence and we ammortize the loading over time
        start_t = time()
        test_data_fp = TEST_DIRECTORY / f"{TASK_DATA}-test-{seed}.xlsx"
        logger.info(f"Loading test data from {test_data_fp}")
        data_df = pd.read_excel(test_data_fp)
        sentences = data_df["sentence"].to_list()
        logger.debug(f"Number of sentences: {len(sentences)}")
        labels = data_df["label"].to_numpy()
        logger.debug(f"Number of labels: {len(labels)}")

        inputs_list = llama2_prompt_generator(TASK_INSTRUCTION, sentences)
        # for SENTENCE in tqdm(sentences, desc="Generating prompts"):
        #     # inputs_list.append({"instruction": TASK_INSTRUCTION, "sentence": SENTENCE})
        #     inputs_list.append(

        logger.info(f"Prompts created -- Running inference on {args.model_id}...")
        generation_result = generation_pipeline(inputs_list)

        logger.info(f"Model {args.model_id} inference completed. Processing outputs...")
        output_list = []
        for i in range(len(generation_result)):
            output_list.append(
                [labels[i], sentences[i], generation_result[i][0]["generated_text"]]
            )
        logger.debug(f"Number of outputs: {len(output_list)}")
        time_taken = int((time() - start_t) / 60.0)

        results = pd.DataFrame(
            output_list, columns=["true_label", "original_sent", "text_output"]
        )
        model_name = args.model_id.split("/")[-1]
        results_fp = (
            f"{model_name}_{seed}_{TODAY.strftime('%d_%m_%Y')}_{time_taken}.csv"
        )
        logger.info(f"Time taken: {time_taken} minutes")
        results.to_csv(
            PROMPT_OUTPUTS / results_fp,
            index=False,
        )
        logger.info(f"Results saved to {PROMPT_OUTPUTS / results_fp}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
