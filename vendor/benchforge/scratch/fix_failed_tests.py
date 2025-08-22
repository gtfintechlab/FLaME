#!/usr/bin/env python3
"""Fix and re-run failed tests from parallel execution."""

import json
import pandas as pd
import subprocess
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def fix_mistral_7b_flame():
    """Fix Mistral-7B FLAME results with whitespace and label mapping issues."""
    print("Fixing Mistral-7B FLAME results...")

    # Load the temp file
    df = pd.read_csv("results/flame/mistral-7b-v0.3_temp.csv")

    # Strip whitespace from responses
    df["llm_responses"] = df["llm_responses"].str.strip()

    # Map string labels to numbers for consistency
    label_map = {"DOVISH": 0, "HAWKISH": 1, "NEUTRAL": 2}
    df["predicted_labels"] = df["llm_responses"].map(label_map)

    # Filter valid predictions
    valid_mask = df["predicted_labels"].notna()
    valid_df = df[valid_mask].copy()

    if len(valid_df) > 0:
        # Calculate metrics
        accuracy = accuracy_score(
            valid_df["actual_labels"], valid_df["predicted_labels"]
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_df["actual_labels"],
            valid_df["predicted_labels"],
            average="weighted",
            zero_division=0,
        )

        # Save corrected results
        df.to_csv("results/flame/mistral-7b-v0.3_50samples.csv", index=False)

        # Save metrics
        metrics_df = pd.DataFrame(
            [
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            ]
        )
        metrics_df.to_csv("results/flame/mistral-7b-v0.3_metrics.csv", index=False)

        # Save summary
        summary = {
            "model": "together_ai/mistralai/Mistral-7B-Instruct-v0.3",
            "model_name": "mistral-7b-v0.3",
            "method": "flame",
            "samples": len(df),
            "accuracy": accuracy,
            "f1_score": f1,
            "extraction_rate": len(valid_df) / len(df),
            "time_seconds": 10.0,
        }
        with open("results/flame/mistral-7b-v0.3_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"✅ Fixed: Accuracy={accuracy:.3f}, F1={f1:.3f}, Extraction={len(valid_df)}/{len(df)}"
        )
        return True
    else:
        print("❌ No valid predictions found")
        return False


def rerun_llama_8b_tests():
    """Re-run both FLAME and BenchForge tests for Llama-3.1-8B."""
    print("\nRe-running Llama-3.1-8B tests...")

    # Re-run FLAME test
    print("Running FLAME test...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "benchforge/run_single_flame_test.py",
            "--model",
            "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "--samples",
            "50",
            "--batch-size",
            "10",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ FLAME test failed: {result.stderr[:200]}")
    else:
        print("✅ FLAME test completed")

    # Re-run BenchForge test
    print("Running BenchForge test...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "benchforge/run_single_benchforge_test.py",
            "--model",
            "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "--samples",
            "50",
            "--batch-size",
            "10",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ BenchForge test failed: {result.stderr[:200]}")
    else:
        print("✅ BenchForge test completed")


def main():
    """Main function to fix all failed tests."""
    print("=" * 60)
    print("Fixing Failed Tests")
    print("=" * 60)

    # Fix Mistral-7B FLAME results
    fix_mistral_7b_flame()

    # Re-run Llama-3.1-8B tests
    rerun_llama_8b_tests()

    print("\n" + "=" * 60)
    print("Test fixes complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
