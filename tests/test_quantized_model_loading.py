import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from superflue.utils.logging_utils import setup_logger
from superflue.utils.hf_model import get_hf_model

ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

logger = setup_logger(__name__)


@pytest.mark.parametrize(
    "quantization, expected_dtype",
    [
        ("fp16", torch.bfloat16),
        ("int8", torch.int8),  # Assuming int8 quantization sets the dtype to torch.int8
        ("none", torch.float32),  # Default dtype for models without quantization
    ],
)
def test_get_hf_model(quantization, expected_dtype):
    logger.info(f"Testing get_dolly with quantization '{quantization}'")
    model, tokenizer = get_hf_model(quantization)

    # Check if the returned objects are of the correct type
    assert isinstance(model, AutoModelForCausalLM)
    assert isinstance(tokenizer, AutoTokenizer)

    # Check if the model's dtype matches the expected dtype based on quantization
    for param in model.parameters():
        assert param.dtype == expected_dtype
        break  # Checking the dtype of the first parameter should suffice
