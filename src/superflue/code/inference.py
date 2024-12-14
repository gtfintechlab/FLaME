from time import time
from datetime import datetime
from superflue.code.fpb.fpb_inference import fpb_inference
from superflue.code.numclaim.numclaim_inference import numclaim_inference
from superflue.code.fnxl.fnxl_inference import fnxl_inference
from superflue.code.fomc.fomc_inference import fomc_inference
from superflue.code.finbench.finbench_inference import finbench_inference
from superflue.code.finer.finer_inference import finer_inference
from superflue.code.finentity.finentity_inference import finentity_inference
from superflue.code.headlines.headlines_inference import headlines_inference
from superflue.code.fiqa.fiqa_task1_inference import fiqa_task1_inference
from superflue.code.fiqa.fiqa_task2_inference import fiqa_task2_inference
from superflue.code.edtsum.edtsum_inference import edtsum_inference
from superflue.code.banking77.banking77_inference import banking77_inference
from superflue.code.finred.finred_inference import finred_inference
from superflue.code.causal_classification.causal_classification_inference import (
    causal_classification_inference,
)
from superflue.code.subjectiveqa.subjectiveqa_inference import subjectiveqa_inference
from superflue.code.ectsum.ectsum_inference import ectsum_inference
from superflue.code.refind.refind_inference import refind_inference
from superflue.code.finqa.finqa_inference import finqa_inference
from superflue.code.tatqa.tatqa_inference import tatqa_inference
from superflue.code.convfinqa.convfinqa_inference import convfinqa_inference
from superflue.code.causal_detection.casual_detection_inference import (
    casual_detection_inference,
)
from superflue.code.mmlu.mmlu_inference import mmlu_inference
from superflue.code.bizbench.bizbench_inference import bizbench_inference
from superflue.code.econlogicqa.econlogicqa_inference import econlogicqa_inference
from superflue.utils.save_utils import save_inference_results
from superflue.utils.logging_utils import get_logger
from superflue.utils.sampling_utils import sample_dataset

# Get logger for this module
logger = get_logger(__name__)


def wrap_inference_with_sampling(inference_fn):
    """Wrapper to handle dataset sampling before task-specific inference.

    This decorator-like function wraps task inference functions to handle dataset sampling
    based on the provided arguments before passing the data to the task-specific inference.
    """

    def wrapped(args):
        # Store original sample_size and method
        original_sample_size = getattr(args, "sample_size", None)
        original_method = getattr(
            args, "method", "head"
        )  # Default to 'head' if not specified

        # If sampling is requested, modify the dataset loading behavior
        if original_sample_size is not None:

            def dataset_wrapper(dataset, split="test"):
                if original_sample_size > 0:
                    logger.info(
                        f"Sampling {original_sample_size} examples using method: {original_method}"
                    )
                    return sample_dataset(
                        dataset=dataset,
                        sample_size=original_sample_size,
                        method=original_method,
                        split=split,
                    )
                return dataset[split]

            # Temporarily modify the dataset loading behavior
            # This assumes each task's inference function uses load_dataset internally
            from datasets import load_dataset

            original_getitem = load_dataset.__getitem__
            load_dataset.__getitem__ = dataset_wrapper

            try:
                result = inference_fn(args)
            finally:
                # Restore original behavior
                load_dataset.__getitem__ = original_getitem
        else:
            result = inference_fn(args)

        return result

    return wrapped


def main(args):
    """Run inference for the specified task.

    Args:
        args: Command line arguments containing:
            - dataset: Name of the task/dataset
            - dataset_org: Organization holding the dataset
            - inference_model: Model to use
            - sample_size: Optional number of samples to use
            - method: Sampling method ('random', 'head', or 'tail')
            - Other task-specific parameters
    """
    task = args.dataset.strip('"""')

    # Log dataset organization info
    logger.info(f"Using dataset organization: {args.dataset_org}")

    task_inference_map = {
        "numclaim": numclaim_inference,
        "fpb": fpb_inference,
        "fomc": fomc_inference,
        "finbench": finbench_inference,
        "finqa": finqa_inference,
        "finer": finer_inference,
        "convfinqa": convfinqa_inference,
        "finentity": finentity_inference,
        "headlines": headlines_inference,
        "fiqa_task1": fiqa_task1_inference,
        "fiqa_task2": fiqa_task2_inference,
        "edtsum": edtsum_inference,
        "fnxl": fnxl_inference,
        "tatqa": tatqa_inference,
        "causal_detection": casual_detection_inference,
        "finred": finred_inference,
        "causal_classification": causal_classification_inference,
        "subjectiveqa": subjectiveqa_inference,
        "ectsum": ectsum_inference,
        "refind": refind_inference,
        "banking77": banking77_inference,
        "mmlu": mmlu_inference,
        "bizbench": bizbench_inference,
        "econlogicqa": econlogicqa_inference,
    }

    if task in task_inference_map:
        start_t = time()
        # Wrap the inference function with sampling support
        inference_function = wrap_inference_with_sampling(task_inference_map[task])
        df = inference_function(args)
        time_taken = time() - start_t
        logger.info(f"Time taken for inference: {time_taken}")

        # Use standardized save utility with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "task": task,
            "model": args.inference_model,
            "dataset_org": args.dataset_org,
            "runtime_seconds": time_taken,
            "timestamp": timestamp,
            "parameters": {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "top_k": getattr(args, "top_k", None),
                "repetition_penalty": args.repetition_penalty,
                "batch_size": args.batch_size,
                "sample_size": getattr(args, "sample_size", None),
                "sampling_method": getattr(args, "method", None),
            },
        }

        results_path = save_inference_results(
            df=df,
            task=task,
            model=args.inference_model,
            metadata=metadata,
            timestamp=timestamp,
        )

        logger.info(f"Inference completed for {task}. Results saved to {results_path}")
    else:
        logger.error(f"Task '{task}' not found in the task generation map.")
