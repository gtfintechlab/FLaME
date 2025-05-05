"""Central registry mapping task names to their inference / evaluation functions.

Keeping a single source-of-truth avoids the two divergent dictionaries that used to
live in `inference.py` and `evaluate.py`.

Any time you add a new task, register its callable here in *both* maps (if it has
both inference and evaluation), and update the skip lists in the offline test
suite if necessary.
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
