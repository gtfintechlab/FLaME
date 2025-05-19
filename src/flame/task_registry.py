"""Central registry mapping task names to their inference / evaluation functions.
Keeping a single source-of-truth avoids divergence.
Any time you add a new task, register its callable here in *both* maps
(if it has both inference and evaluation),
and update the skip lists in the offline test suite if necessary.
"""

from __future__ import annotations

# ------------------------------
# Inference functions
# ------------------------------
from flame.code.numclaim.numclaim_inference import numclaim_inference
from flame.code.fomc.fomc_inference import fomc_inference
from flame.code.finer.finer_inference import finer_inference
from flame.code.finentity.finentity_inference import finentity_inference
from flame.code.causal_classification.causal_classification_inference import (
    causal_classification_inference,
)
from flame.code.subjectiveqa.subjectiveqa_inference import subjectiveqa_inference
from flame.code.ectsum.ectsum_inference import ectsum_inference
from flame.code.fnxl.fnxl_inference import fnxl_inference
from flame.code.fpb.fpb_inference import fpb_inference
from flame.code.banking77.banking77_inference import banking77_inference
from flame.code.bizbench.bizbench_inference import bizbench_inference
from flame.code.causal_detection.causal_detection_inference import (
    causal_detection_inference,
)
from flame.code.convfinqa.convfinqa_inference import convfinqa_inference
from flame.code.econlogicqa.econlogicqa_inference import econlogicqa_inference
from flame.code.edtsum.edtsum_inference import edtsum_inference
from flame.code.finbench.finbench_inference import finbench_inference
from flame.code.finqa.finqa_inference import finqa_inference
from flame.code.finred.finred_inference import finred_inference
from flame.code.fiqa.fiqa_task1_inference import fiqa_task1_inference
from flame.code.fiqa.fiqa_task2_inference import fiqa_task2_inference
from flame.code.headlines.headlines_inference import headlines_inference
from flame.code.mmlu.mmlu_inference import mmlu_inference
from flame.code.refind.refind_inference import refind_inference
from flame.code.tatqa.tatqa_inference import tatqa_inference

# ------------------------------
# Evaluation functions
# ------------------------------
from flame.code.numclaim.numclaim_evaluate import numclaim_evaluate
from flame.code.finer.finer_evaluate import finer_evaluate
from flame.code.finentity.finentity_evaluate import finentity_evaluate
from flame.code.fnxl.fnxl_evaluate import fnxl_evaluate
from flame.code.causal_classification.causal_classification_evaluate import (
    causal_classification_evaluate,
)
from flame.code.subjectiveqa.subjectiveqa_evaluate import subjectiveqa_evaluate
from flame.code.ectsum.ectsum_evaluate import ectsum_evaluate
from flame.code.refind.refind_evaluate import refind_evaluate
from flame.code.fpb.fpb_evaluate import fpb_evaluate
from flame.code.banking77.banking77_evaluate import banking77_evaluate
from flame.code.convfinqa.convfinqa_evaluate import convfinqa_evaluate
from flame.code.finqa.finqa_evaluate import finqa_evaluate
from flame.code.tatqa.tatqa_evaluate import tatqa_evaluate
from flame.code.causal_detection.casual_detection_evaluate_llm import (
    causal_detection_evaluate,
)

# Note: not every task has both an inference and an evaluation implementation.
# When missing, simply omit it from the corresponding map.

INFERENCE_MAP: dict[str, callable] = {
    "numclaim": numclaim_inference,
    "fomc": fomc_inference,
    "finer": finer_inference,
    "finentity": finentity_inference,
    "causal_classification": causal_classification_inference,
    "subjectiveqa": subjectiveqa_inference,
    "ectsum": ectsum_inference,
    "fnxl": fnxl_inference,
    "fpb": fpb_inference,
    "banking77": banking77_inference,
    "bizbench": bizbench_inference,
    "causal_detection": causal_detection_inference,
    "convfinqa": convfinqa_inference,
    "econlogicqa": econlogicqa_inference,
    "edtsum": edtsum_inference,
    "finbench": finbench_inference,
    "finqa": finqa_inference,
    "finred": finred_inference,
    "fiqa_task1": fiqa_task1_inference,
    "fiqa_task2": fiqa_task2_inference,
    "headlines": headlines_inference,
    "mmlu": mmlu_inference,
    "refind": refind_inference,
    "tatqa": tatqa_inference,
}

EVALUATE_MAP: dict[str, callable] = {
    "numclaim": numclaim_evaluate,
    "finer": finer_evaluate,
    "finentity": finentity_evaluate,
    "fnxl": fnxl_evaluate,
    "causal_classification": causal_classification_evaluate,
    "subjectiveqa": subjectiveqa_evaluate,
    "ectsum": ectsum_evaluate,
    "refind": refind_evaluate,
    "fpb": fpb_evaluate,
    "banking77": banking77_evaluate,
    "convfinqa": convfinqa_evaluate,
    "finqa": finqa_evaluate,
    "tatqa": tatqa_evaluate,
    "causal_detection": causal_detection_evaluate,
}


def supported(mode: str) -> set[str]:
    """Return the set of supported task names for the given *mode*.

    Parameters
    ----------
    mode : str
        Either ``"inference"`` or ``"evaluate"``.
    """
    _mode = mode.lower()
    if _mode not in {"inference", "evaluate"}:
        raise ValueError("mode must be 'inference' or 'evaluate'")
    return set(INFERENCE_MAP if _mode == "inference" else EVALUATE_MAP)
