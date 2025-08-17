#!/usr/bin/env python3
"""Test the exact extraction workflow as used in FLAME-BenchForge."""

import pandas as pd
from flame.benchforge import FLAMEConfig, ResponseExtractor, ExtractionStrategy
from flame.tasks import create_task, register_all_flame_tasks

# Register tasks first
register_all_flame_tasks()


def test_extraction_workflow():
    """Test extraction using the exact FLAME workflow."""

    # Create FOMC task
    config = FLAMEConfig(
        name="fomc",
        dataset="fomc",
        huggingface_dataset="gtfintechlab/fomc_communication",
        valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
        extraction_strategy=ExtractionStrategy.KEYWORD,
    )

    task = create_task("fomc", config)

    # Test responses
    test_responses = [
        "The correct classification is:\n\nNEUTRAL\n\nThe",
        "Classification: HAWKISH",
        "DOVISH",
        "Based on the analysis, this is HAWKISH",
    ]

    print("Testing extraction with FLAME task:")
    print("=" * 60)

    for i, response in enumerate(test_responses, 1):
        print(f"\nTest {i}:")
        print(f"Response: {response[:50]}...")

        # Extract using task's method
        extracted = task.extract_response(response)
        print(f"Extracted: {extracted}")

        # Also test with extractor directly
        extractor = ResponseExtractor()
        direct_result = extractor.extract_label(
            response, config.valid_labels, strategy=config.extraction_strategy
        )
        print(f"Direct extractor: {direct_result}")


def test_actual_results():
    """Test on actual results from the live test."""
    print("\n" + "=" * 60)
    print("Testing on actual results:")
    print("=" * 60)

    # Load actual results
    df = pd.read_csv(
        "scratch/live_test_results/test_1_basic/fomc_20250816_134452_b531740c.csv"
    )

    # Create task for extraction
    config = FLAMEConfig(
        name="fomc",
        dataset="fomc",
        huggingface_dataset="gtfintechlab/fomc_communication",
        valid_labels=["HAWKISH", "DOVISH", "NEUTRAL"],
        extraction_strategy=ExtractionStrategy.KEYWORD,
    )

    task = create_task("fomc", config)

    for idx, row in df.head(3).iterrows():
        print(f"\nRow {idx}:")
        raw_response = row["raw_response"]
        print(f"Raw response: {raw_response[:50]}...")

        # Try extraction
        extracted = task.extract_response(raw_response)
        print(f"Task extraction: {extracted}")

        # Check type
        print(f"Response type: {type(raw_response)}")
        print(f"Response repr: {repr(raw_response)}")


if __name__ == "__main__":
    test_extraction_workflow()
    test_actual_results()
