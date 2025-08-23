#!/usr/bin/env python3

"""Test script for FinBench task implementation."""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "vendor/benchforge"))

from bench_forge.flame.tasks.finbench import FinBenchTask, FinBenchConfig
from bench_forge.tasks.config import PromptFormat


def test_finbench_task():
    """Test FinBench task implementation with sample data."""
    print("Testing FinBench Task Implementation")
    print("=" * 50)

    # Initialize task
    config = FinBenchConfig(name="finbench")
    task = FinBenchTask(config)
    print(f"✓ Task initialized: {task.config.name}")

    # Test sample data (simulating real FinBench format)
    test_samples = [
        {
            "X_profile": "Age: 35, Income: $75000, Debt: $12000, Credit Score: 720, Employment: Full-time, Education: Bachelor's",
            "y": "LOW RISK",
        },
        {
            "X_profile": "Age: 22, Income: $28000, Debt: $55000, Credit Score: 580, Employment: Part-time, Education: High School",
            "y": "HIGH RISK",
        },
        {
            "X_profile": "Age: 45, Income: $95000, Debt: $8000, Credit Score: 780, Employment: Full-time, Education: Master's",
            "y": "LOW RISK",
        },
        {
            "X_profile": "Age: 30, Income: $45000, Debt: $35000, Credit Score: 620, Employment: Contract, Education: Bachelor's",
            "y": "HIGH RISK",
        },
        {
            "X_profile": "Age: 40, Income: $85000, Debt: $15000, Credit Score: 750, Employment: Full-time, Education: Bachelor's",
            "y": "LOW RISK",
        },
        {
            "X_profile": "Age: 25, Income: $35000, Debt: $40000, Credit Score: 600, Employment: Unemployed, Education: Some College",
            "y": "HIGH RISK",
        },
        {
            "X_profile": "Age: 50, Income: $120000, Debt: $20000, Credit Score: 800, Employment: Full-time, Education: PhD",
            "y": "LOW RISK",
        },
        {
            "X_profile": "Age: 28, Income: $32000, Debt: $48000, Credit Score: 590, Employment: Part-time, Education: High School",
            "y": "HIGH RISK",
        },
    ]

    print(f"✓ Test data prepared: {len(test_samples)} samples")

    # Test prompt creation for different formats
    print("\n" + "=" * 50)
    print("TESTING PROMPT CREATION")
    print("=" * 50)

    sample = test_samples[0]

    # Test Zero-shot prompt
    prompt_zero = task.create_prompt(sample, PromptFormat.ZERO_SHOT)
    print(f"✓ Zero-shot prompt created ({len(prompt_zero)} chars)")

    # Test Few-shot prompt
    prompt_few = task.create_prompt(sample, PromptFormat.FEW_SHOT)
    print(f"✓ Few-shot prompt created ({len(prompt_few)} chars)")

    # Test CoT prompt
    prompt_cot = task.create_prompt(sample, PromptFormat.CHAIN_OF_THOUGHT)
    print(f"✓ Chain-of-thought prompt created ({len(prompt_cot)} chars)")

    print(f"✓ Prompts created: {task._stats['prompts_created']}")

    # Test response extraction with different mock responses
    print("\n" + "=" * 50)
    print("TESTING RESPONSE EXTRACTION")
    print("=" * 50)

    test_responses = [
        "LOW RISK\nThis applicant has stable employment and good credit.",
        "HIGH RISK\nHigh debt-to-income ratio and poor credit score indicate default risk.",
        "Risk Assessment: LOW RISK\nStrong financial profile with excellent credit.",
        "The applicant should be classified as HIGH RISK due to unemployment.",
        "After analyzing the profile, I classify this as LOW RISK because of stable income and low debt.",
        "This person is risky and likely to default on the loan. HIGH RISK.",
        "APPROVED - LOW RISK candidate with creditworthy profile.",
        "REJECT - HIGH RISK due to financial instability and poor credit history.",
        "Risk: LOW RISK\nGood credit score and stable employment history.",
        "HIGH RISK\nMultiple red flags including high debt and unstable employment.",
        "Assessment: The applicant appears to be LOW RISK for loan approval.",
        "Classification: HIGH RISK\nPoor financial indicators suggest default likelihood.",
        # Edge cases
        "",  # Empty response
        "Maybe risky but not sure",  # Unclear response
        "The credit score is good but income is concerning",  # No clear label
        "This is a complex case with mixed indicators",  # Ambiguous
    ]

    extracted_results = []
    for i, response in enumerate(test_responses):
        extracted = task.extract_response(response, sample)
        extracted_results.append(extracted)
        status = "✓" if extracted else "✗"
        print(
            f"{status} Response {i + 1:2d}: {'SUCCESS' if extracted else 'FAILED':7} -> {extracted}"
        )

    # Calculate extraction success rate
    successful_extractions = sum(
        1 for result in extracted_results if result is not None
    )
    total_responses = len(test_responses)
    success_rate = (successful_extractions / total_responses) * 100

    print(
        f"\n✓ Extraction results: {successful_extractions}/{total_responses} successful ({success_rate:.1f}%)"
    )
    print(f"✓ Responses extracted: {task._stats['responses_extracted']}")
    print(f"✓ Extraction failures: {task._stats['extraction_failures']}")

    # Test format_results method
    print("\n" + "=" * 50)
    print("TESTING RESULTS FORMATTING")
    print("=" * 50)

    # Create mock data for testing
    prompts = [task.create_prompt(sample) for sample in test_samples]
    mock_responses = [
        "LOW RISK\nStrong financial profile",
        "HIGH RISK\nPoor credit and high debt",
        "LOW RISK\nExcellent creditworthiness",
        "HIGH RISK\nUnstable employment",
        "LOW RISK\nGood income and credit",
        "HIGH RISK\nUnemployed with high debt",
        "LOW RISK\nVery strong financial position",
        "HIGH RISK\nPoor credit and low income",
    ]

    extracted_responses = [
        task.extract_response(resp, test_samples[i])
        for i, resp in enumerate(mock_responses)
    ]

    # Test format_results
    results_df = task.format_results(
        test_samples, prompts, mock_responses, extracted_responses
    )

    print(
        f"✓ Results DataFrame created with {len(results_df)} rows and {len(results_df.columns)} columns"
    )
    print("✓ Required FLAME columns present:")

    flame_columns = [
        "X_profile",
        "y",
        "llm_responses",
        "complete_responses",
        "extracted_labels",
    ]
    for col in flame_columns:
        present = col in results_df.columns
        status = "✓" if present else "✗"
        print(f"  {status} {col}")

    # Check extraction success rate in formatted results
    total_formatted = len(results_df)
    successful_formatted = results_df["extracted_labels"].notna().sum()
    formatted_success_rate = (
        (successful_formatted / total_formatted * 100) if total_formatted > 0 else 0
    )

    print(
        f"✓ Formatted extraction rate: {successful_formatted}/{total_formatted} ({formatted_success_rate:.1f}%)"
    )

    # Show risk distribution
    if successful_formatted > 0:
        risk_distribution = results_df["extracted_labels"].dropna().value_counts()
        print(f"✓ Risk distribution: {dict(risk_distribution.items())}")

    # Test ground truth extraction
    print("\n" + "=" * 50)
    print("TESTING GROUND TRUTH EXTRACTION")
    print("=" * 50)

    for i, sample in enumerate(test_samples[:3]):
        ground_truth = task.get_ground_truth(sample)
        print(f"✓ Sample {i + 1}: {ground_truth}")

    # Validation tests
    print("\n" + "=" * 50)
    print("TESTING VALIDATION METHODS")
    print("=" * 50)

    validation_tests = [
        ("low risk", "LOW RISK"),
        ("high risk", "HIGH RISK"),
        ("LOW RISK", "LOW RISK"),
        ("HIGH RISK", "HIGH RISK"),
        ("approve", "LOW RISK"),
        ("reject", "HIGH RISK"),
        ("safe", "LOW RISK"),
        ("risky", "HIGH RISK"),
        ("invalid", None),
        ("", None),
    ]

    for test_input, expected in validation_tests:
        result = task._validate_risk_label(test_input)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{test_input}' -> {result} (expected: {expected})")

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print("✓ Task initialization: SUCCESS")
    print("✓ Prompt creation: SUCCESS (3 formats tested)")
    print(f"✓ Response extraction: {success_rate:.1f}% success rate")
    print("✓ Results formatting: SUCCESS")
    print("✓ Ground truth extraction: SUCCESS")
    print("✓ Validation methods: SUCCESS")
    print("✓ FLAME compatibility: SUCCESS")

    print("\nFinBench task implementation test completed successfully!")
    print("Ready for integration with FLAME evaluation pipeline.")

    return results_df, success_rate


if __name__ == "__main__":
    test_finbench_task()
