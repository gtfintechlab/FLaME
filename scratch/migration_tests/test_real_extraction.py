#!/usr/bin/env python3
"""Test extraction on real saved FOMC responses."""

import pandas as pd
import sys

# Add paths for imports
sys.path.insert(0, "/home/gmatlin/Codespace/FLAME")
sys.path.insert(0, "/home/gmatlin/Codespace/FLAME/benchforge")

from benchforge.bench_forge.flame.tasks.fomc import FOMCTask, FOMCConfig


def test_extraction_on_real_data():
    """Test our extraction methods on real saved responses."""

    # Load a pre-existing result file with real responses
    result_file = "results/fomc/fomc_together_ai_meta-llama_Llama-4-Scout-17B-16E-Instruct_16_08_2025_2051.csv"

    print(f"Loading saved responses from: {result_file}")
    df = pd.read_csv(result_file)

    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print()

    # Check what response data we have
    if "raw_response" in df.columns:
        non_null = df["raw_response"].notna().sum()
        print(f"Raw responses available: {non_null}/{len(df)}")
    else:
        print("ERROR: No raw_response column found!")
        return

    # Initialize our FOMC task with fixed extraction
    config = FOMCConfig(
        name="fomc",
        huggingface_dataset="gtfintechlab/fomc_communication",
        text_field="sentence",
        label_field="label",
    )
    task = FOMCTask(config)

    # Test extraction on all available responses
    successful_extractions = 0
    failed_extractions = 0
    extraction_results = []

    print("\nTesting extraction on real responses...")
    print("-" * 50)

    for idx, row in df.iterrows():
        if pd.isna(row.get("raw_response")):
            continue

        raw_response = row["raw_response"]

        # Apply our extraction method
        extracted = task.extract_label_from_response(raw_response)

        # Check if extraction succeeded
        if extracted is not None:
            successful_extractions += 1
        else:
            failed_extractions += 1
            # Show failures for debugging
            if failed_extractions <= 5:  # Show first 5 failures
                print(f"\nFAILED extraction at index {idx}:")
                print(f"Response: {raw_response[:200]}...")

        extraction_results.append(
            {
                "index": idx,
                "original_extraction": row.get("extracted_response"),
                "new_extraction": extracted,
                "raw_response": raw_response[:100],
                "matches": row.get("extracted_response") == extracted
                if pd.notna(row.get("extracted_response"))
                else None,
            }
        )

    # Analyze results
    total_tested = successful_extractions + failed_extractions
    success_rate = (
        (successful_extractions / total_tested * 100) if total_tested > 0 else 0
    )

    print("\n" + "=" * 50)
    print("EXTRACTION TEST RESULTS")
    print("=" * 50)
    print(f"Total samples tested: {total_tested}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Success rate: {success_rate:.2f}%")

    # Compare with original extractions if available
    if "extracted_response" in df.columns:
        results_df = pd.DataFrame(extraction_results)

        # Filter to non-null comparisons
        comparable = results_df[results_df["matches"].notna()]
        if len(comparable) > 0:
            matches = comparable["matches"].sum()
            match_rate = matches / len(comparable) * 100
            print("\nComparison with original extractions:")
            print(
                f"Matching extractions: {matches}/{len(comparable)} ({match_rate:.2f}%)"
            )

        # Show some examples where extractions differ
        differs = results_df[
            (not results_df["matches"]) & results_df["new_extraction"].notna()
        ]
        if len(differs) > 0:
            print(f"\nFound {len(differs)} cases where extractions differ")
            print("\nFirst 3 differences:")
            for i, row in differs.head(3).iterrows():
                print(
                    f"  Original: {row['original_extraction']}, New: {row['new_extraction']}"
                )

    # Check label distribution
    results_df = pd.DataFrame(extraction_results)
    valid_extractions = results_df[results_df["new_extraction"].notna()]
    if len(valid_extractions) > 0:
        print("\nLabel distribution in extracted responses:")
        distribution = valid_extractions["new_extraction"].value_counts()
        for label, count in distribution.items():
            pct = count / len(valid_extractions) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    # Return success metrics
    return {
        "success_rate": success_rate,
        "total_tested": total_tested,
        "successful": successful_extractions,
        "failed": failed_extractions,
    }


if __name__ == "__main__":
    results = test_extraction_on_real_data()

    # Determine if we meet the >95% threshold
    if results["success_rate"] >= 95:
        print("\n✅ SUCCESS: Extraction rate exceeds 95% threshold!")
    else:
        print(
            f"\n⚠️ WARNING: Extraction rate {results['success_rate']:.2f}% is below 95% threshold"
        )
        print("Further investigation needed...")
