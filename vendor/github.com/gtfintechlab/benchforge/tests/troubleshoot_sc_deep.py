#!/usr/bin/env python3
"""Deep troubleshooting for Causal Classification (SC) task.

This script performs comprehensive analysis to identify why both implementations
are getting poor accuracy and ensure proper API integration.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

# Add paths for both implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# Configure logging with detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import utilities
from flame.utils.dataset_utils import safe_load_dataset
from litellm import completion
import litellm
import pandas as pd

# Default model
DEFAULT_MODEL = "together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct"

# CRITICAL: Disable ALL caching
litellm.cache = None
print("🚫 CACHING DISABLED - All API calls will be independent")


def test_raw_api_call(text: str, prompt_version: str = "flame", model: str = DEFAULT_MODEL):
    """Test raw API call to understand model behavior."""
    
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
    if not api_key:
        return None
    
    # Different prompt versions to test
    prompts = {
        "flame": f"""Discard all the previous instructions. Behave like you are an expert causal classification model.
    Below is a sentence. Classify it into one of the following categories:
                    0 - Non-causal
                    1 - Direct causal
                    2 - Indirect causal
                    Only return the label number without any additional text. \n\n {text}""",
        
        "simple": f"""Classify this sentence as:
0 - Non-causal (no cause-effect relationship)
1 - Direct causal (clear cause-effect)
2 - Indirect causal (conditional/complex causality)

Sentence: {text}

Answer with only the number (0, 1, or 2):""",
        
        "explicit": f"""You must classify the following sentence into exactly one category.

Categories:
- If the sentence has NO causal relationship, respond with: 0
- If the sentence has a DIRECT cause-effect relationship, respond with: 1
- If the sentence has an INDIRECT or conditional causal relationship, respond with: 2

Sentence: "{text}"

Your response must be a single digit (0, 1, or 2):""",

        "examples": f"""Examples of causal classification:

"The stock price rose 10%" → 0 (just a fact, no causality)
"Rising costs led to lower profits" → 1 (direct cause-effect)
"If inflation continues, we may see reduced spending" → 2 (conditional/indirect)

Now classify this sentence:
"{text}"

Answer (0, 1, or 2):"""
    }
    
    prompt = prompts.get(prompt_version, prompts["flame"])
    
    print(f"\n{'='*60}")
    print(f"Testing prompt version: {prompt_version}")
    print(f"Text: {text[:100]}...")
    print(f"Prompt length: {len(prompt)} chars")
    
    try:
        start_time = time.time()
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,  # Very short - we only need a digit
            top_p=0.9,
            api_key=api_key
        )
        api_time = time.time() - start_time
        
        response_text = response.choices[0].message.content.strip()
        
        print(f"API Response time: {api_time:.2f}s")
        print(f"Raw response: '{response_text}'")
        
        # Try to extract the digit
        extracted = None
        for char in response_text:
            if char in ['0', '1', '2']:
                extracted = char
                break
        
        print(f"Extracted label: {extracted}")
        
        return {
            "prompt_version": prompt_version,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response": response_text,
            "extracted": extracted,
            "api_time": api_time
        }
        
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None


def diagnose_dataset_distribution():
    """Check the dataset label distribution."""
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    # Load dataset
    dataset = safe_load_dataset("gtfintechlab/CausalClassification", trust_remote_code=True)
    
    # Analyze distribution
    train_labels = [str(sample.get("label", "")) for sample in dataset["train"]]
    test_labels = [str(sample.get("label", "")) for sample in dataset["test"]]
    
    print("\n📊 Label Distribution:")
    print("-" * 40)
    
    for split_name, labels in [("Train", train_labels), ("Test", test_labels)]:
        print(f"\n{split_name} Set ({len(labels)} samples):")
        for label in ['0', '1', '2']:
            count = labels.count(label)
            pct = count / len(labels) * 100 if labels else 0
            print(f"  Label {label}: {count:4d} ({pct:5.1f}%)")
    
    # Get diverse samples
    test_samples = []
    for label in ['0', '1', '2']:
        label_samples = [s for s in dataset["test"] if str(s.get("label", "")) == label]
        if label_samples:
            test_samples.extend(label_samples[:2])  # Get 2 of each type
    
    return dataset, test_samples


def test_both_implementations(dataset, test_samples):
    """Test both BenchForge and FLAME implementations side by side."""
    print("\n" + "="*80)
    print("IMPLEMENTATION COMPARISON")
    print("="*80)
    
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
    if not api_key:
        return None, None
    
    # Test BenchForge
    print("\n📦 Testing BenchForge Implementation:")
    print("-" * 40)
    
    from bench_forge.flame.tasks.causal_classification import CausalClassificationTask, CausalClassificationConfig
    from bench_forge.tasks.config import PromptFormat
    
    config = CausalClassificationConfig(
        name="causal_classification",
        model=DEFAULT_MODEL,
        prompt_format=PromptFormat.ZERO_SHOT,
        max_tokens=10,
        temperature=0.0,
        top_p=0.9
    )
    bf_task = CausalClassificationTask(config)
    
    bf_results = []
    for sample in test_samples:
        text = sample.get("text", "")
        label = str(sample.get("label", ""))
        
        # Create prompt
        prompt = bf_task.create_prompt({"text": text})
        print(f"\nSample (Label {label}):")
        print(f"  Text: {text[:80]}...")
        print(f"  Prompt preview: {prompt[:100]}...")
        
        # Get response
        response = completion(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            top_p=0.9,
            api_key=api_key
        )
        
        response_text = response.choices[0].message.content.strip()
        extracted = bf_task.extract_label_from_response(response_text)
        
        print(f"  Response: '{response_text}'")
        print(f"  Extracted: {extracted}")
        print(f"  Correct: {extracted == label}")
        
        bf_results.append({
            "text": text[:100],
            "actual": label,
            "response": response_text,
            "extracted": extracted,
            "correct": extracted == label
        })
        
        time.sleep(0.5)  # Rate limiting
    
    # Test FLAME
    print("\n🔥 Testing FLAME Implementation:")
    print("-" * 40)
    
    from flame.code.prompts import PromptFormat as FLAMEPromptFormat, get_prompt
    import flame.code.prompts.zeroshot
    
    sc_prompt = get_prompt("causal_classification", FLAMEPromptFormat.ZERO_SHOT)
    
    fl_results = []
    for sample in test_samples:
        text = sample.get("text", "")
        label = str(sample.get("label", ""))
        
        # Create prompt
        prompt = sc_prompt(text)
        print(f"\nSample (Label {label}):")
        print(f"  Text: {text[:80]}...")
        print(f"  Prompt preview: {prompt[:100]}...")
        
        # Get response
        response = completion(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            top_p=0.9,
            api_key=api_key
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract using simple logic
        extracted = None
        for char in response_text:
            if char in ['0', '1', '2']:
                extracted = char
                break
        
        print(f"  Response: '{response_text}'")
        print(f"  Extracted: {extracted}")
        print(f"  Correct: {extracted == label}")
        
        fl_results.append({
            "text": text[:100],
            "actual": label,
            "response": response_text,
            "extracted": extracted,
            "correct": extracted == label
        })
        
        time.sleep(0.5)  # Rate limiting
    
    return bf_results, fl_results


def test_prompt_variations(test_samples):
    """Test different prompt variations to find what works."""
    print("\n" + "="*80)
    print("PROMPT VARIATION TESTING")
    print("="*80)
    
    results = []
    
    # Test each sample with different prompts
    for sample in test_samples[:3]:  # Test first 3 samples
        text = sample.get("text", "")
        label = str(sample.get("label", ""))
        
        print(f"\n📝 Testing sample (Label: {label}):")
        print(f"   {text[:100]}...")
        
        for prompt_version in ["flame", "simple", "explicit", "examples"]:
            result = test_raw_api_call(text, prompt_version)
            if result:
                result["actual_label"] = label
                result["correct"] = result["extracted"] == label
                results.append(result)
            
            time.sleep(0.5)  # Rate limiting
    
    # Analyze results
    print("\n" + "="*80)
    print("PROMPT VARIATION ANALYSIS")
    print("="*80)
    
    for version in ["flame", "simple", "explicit", "examples"]:
        version_results = [r for r in results if r["prompt_version"] == version]
        if version_results:
            correct = sum(1 for r in version_results if r["correct"])
            total = len(version_results)
            print(f"\n{version.upper()} prompt:")
            print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
            print(f"  Responses: {[r['response'] for r in version_results]}")


def main():
    """Main troubleshooting execution."""
    print("\n" + "="*80)
    print(" DEEP CAUSAL CLASSIFICATION TROUBLESHOOTING")
    print("="*80)
    
    # Check API key
    api_key = os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY")
    if not api_key:
        print("❌ Error: TOGETHER_API_KEY not set")
        return 1
    
    print(f"✅ API key configured")
    print(f"🔧 Model: {DEFAULT_MODEL}")
    print(f"🚫 Caching: DISABLED")
    print(f"🌡️ Temperature: 0.0")
    
    # Step 1: Analyze dataset
    dataset, test_samples = diagnose_dataset_distribution()
    
    # Step 2: Test prompt variations
    test_prompt_variations(test_samples)
    
    # Step 3: Test both implementations
    bf_results, fl_results = test_both_implementations(dataset, test_samples)
    
    # Step 4: Final analysis
    print("\n" + "="*80)
    print("TROUBLESHOOTING SUMMARY")
    print("="*80)
    
    if bf_results and fl_results:
        bf_correct = sum(1 for r in bf_results if r["correct"])
        fl_correct = sum(1 for r in fl_results if r["correct"])
        
        print(f"\n📊 Results:")
        print(f"  BenchForge: {bf_correct}/{len(bf_results)} correct")
        print(f"  FLAME: {fl_correct}/{len(fl_results)} correct")
        
        # Check if responses are identical
        identical = sum(1 for b, f in zip(bf_results, fl_results) if b["response"] == f["response"])
        print(f"  Identical responses: {identical}/{len(bf_results)}")
        
        # Check unique responses
        all_bf_responses = set(r["response"] for r in bf_results)
        all_fl_responses = set(r["response"] for r in fl_results)
        
        print(f"\n🔍 Response diversity:")
        print(f"  BenchForge unique responses: {all_bf_responses}")
        print(f"  FLAME unique responses: {all_fl_responses}")
    
    print("\n✅ Troubleshooting complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())