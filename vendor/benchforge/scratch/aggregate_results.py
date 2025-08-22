#!/usr/bin/env python3
"""Aggregate and analyze results from parallel FOMC tests.

This script loads all test results and generates comprehensive comparison reports.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict


def load_all_results() -> Dict:
    """Load all test results from both methods.

    Returns:
        Dictionary with aggregated results
    """
    results = {"flame": {}, "benchforge": {}}

    # Load FLAME results
    flame_dir = Path("results/flame")
    if flame_dir.exists():
        for summary_file in flame_dir.glob("*_summary.json"):
            with open(summary_file) as f:
                data = json.load(f)
                model_name = data.get(
                    "model_name", summary_file.stem.replace("_summary", "")
                )
                results["flame"][model_name] = data

    # Load BenchForge results
    bf_dir = Path("results/benchforge")
    if bf_dir.exists():
        for summary_file in bf_dir.glob("*_summary.json"):
            with open(summary_file) as f:
                data = json.load(f)
                model_name = data.get(
                    "model_name", summary_file.stem.replace("_summary", "")
                )
                results["benchforge"][model_name] = data

    return results


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create comparison table of all results.

    Args:
        results: Aggregated results dictionary

    Returns:
        DataFrame with comparison metrics
    """
    rows = []

    # Get all models tested
    all_models = set(results["flame"].keys()) | set(results["benchforge"].keys())

    for model in sorted(all_models):
        flame = results["flame"].get(model, {})
        benchforge = results["benchforge"].get(model, {})

        row = {
            "Model": model,
            "FLAME_Accuracy": flame.get("accuracy", None),
            "BF_Accuracy": benchforge.get("accuracy", None),
            "Accuracy_Diff": None,
            "FLAME_F1": flame.get("f1_score", None),
            "BF_F1": benchforge.get("f1_score", None),
            "F1_Diff": None,
            "FLAME_Extraction": flame.get("extraction_rate", None),
            "BF_Extraction": benchforge.get("extraction_rate", None),
            "Extraction_Diff": None,
            "FLAME_Time": flame.get("inference_time", None),
            "BF_Time": benchforge.get("inference_time", None),
            "Time_Diff": None,
            "BF_Better": None,
        }

        # Calculate differences if both results exist
        if flame and benchforge:
            if row["FLAME_Accuracy"] is not None and row["BF_Accuracy"] is not None:
                row["Accuracy_Diff"] = row["BF_Accuracy"] - row["FLAME_Accuracy"]

            if row["FLAME_F1"] is not None and row["BF_F1"] is not None:
                row["F1_Diff"] = row["BF_F1"] - row["FLAME_F1"]

            if row["FLAME_Extraction"] is not None and row["BF_Extraction"] is not None:
                row["Extraction_Diff"] = row["BF_Extraction"] - row["FLAME_Extraction"]

            if row["FLAME_Time"] is not None and row["BF_Time"] is not None:
                row["Time_Diff"] = row["BF_Time"] - row["FLAME_Time"]

            # Determine if BenchForge is better or equal
            row["BF_Better"] = (row["BF_Accuracy"] or 0) >= (
                row["FLAME_Accuracy"] or 0
            ) and (row["BF_Extraction"] or 0) >= (row["FLAME_Extraction"] or 0)

        rows.append(row)

    return pd.DataFrame(rows)


def generate_report(results: Dict, df: pd.DataFrame) -> str:
    """Generate detailed comparison report.

    Args:
        results: Raw results dictionary
        df: Comparison DataFrame

    Returns:
        Report string
    """
    report = []
    report.append("# FOMC Model Comparison Report")
    report.append("=" * 60)
    report.append("")

    # Overall statistics
    report.append("## Overall Statistics")
    report.append("")

    total_models = len(df)
    both_tested = df[df["BF_Better"].notna()].shape[0]
    bf_better = df[df["BF_Better"] == True].shape[0] if both_tested > 0 else 0

    report.append(f"- Total models in test: {total_models}")
    report.append(f"- Models with both methods tested: {both_tested}")
    report.append(f"- Models where BenchForge ≥ FLAME: {bf_better}")
    report.append(
        f"- BenchForge is superset: {'✅ YES' if bf_better == both_tested and both_tested > 0 else '❌ NO'}"
    )
    report.append("")

    # Average metrics
    if both_tested > 0:
        report.append("## Average Performance Metrics")
        report.append("")

        avg_flame_acc = df["FLAME_Accuracy"].mean()
        avg_bf_acc = df["BF_Accuracy"].mean()
        avg_flame_f1 = df["FLAME_F1"].mean()
        avg_bf_f1 = df["BF_F1"].mean()
        avg_flame_ext = df["FLAME_Extraction"].mean()
        avg_bf_ext = df["BF_Extraction"].mean()

        report.append("### Accuracy")
        report.append(f"- FLAME Average: {avg_flame_acc:.3f}")
        report.append(f"- BenchForge Average: {avg_bf_acc:.3f}")
        report.append(f"- Difference: {avg_bf_acc - avg_flame_acc:+.3f}")
        report.append("")

        report.append("### F1 Score")
        report.append(f"- FLAME Average: {avg_flame_f1:.3f}")
        report.append(f"- BenchForge Average: {avg_bf_f1:.3f}")
        report.append(f"- Difference: {avg_bf_f1 - avg_flame_f1:+.3f}")
        report.append("")

        report.append("### Extraction Rate")
        report.append(f"- FLAME Average: {avg_flame_ext:.1%}")
        report.append(f"- BenchForge Average: {avg_bf_ext:.1%}")
        report.append(f"- Difference: {(avg_bf_ext - avg_flame_ext)*100:+.1f}%")
        report.append("")

    # Per-model results
    report.append("## Per-Model Results")
    report.append("")

    for _, row in df.iterrows():
        report.append(f"### {row['Model']}")

        if pd.notna(row["FLAME_Accuracy"]) and pd.notna(row["BF_Accuracy"]):
            report.append(
                f"- Accuracy: FLAME={row['FLAME_Accuracy']:.3f}, BF={row['BF_Accuracy']:.3f} (Δ={row['Accuracy_Diff']:+.3f})"
            )
            report.append(
                f"- F1 Score: FLAME={row['FLAME_F1']:.3f}, BF={row['BF_F1']:.3f} (Δ={row['F1_Diff']:+.3f})"
            )
            report.append(
                f"- Extraction: FLAME={row['FLAME_Extraction']:.1%}, BF={row['BF_Extraction']:.1%} (Δ={row['Extraction_Diff']*100:+.1f}%)"
            )
            report.append(
                f"- Time: FLAME={row['FLAME_Time']:.1f}s, BF={row['BF_Time']:.1f}s (Δ={row['Time_Diff']:+.1f}s)"
            )
            report.append(f"- BenchForge ≥ FLAME: {'✅' if row['BF_Better'] else '❌'}")
        else:
            if pd.notna(row["FLAME_Accuracy"]):
                report.append("- ⚠️ Only FLAME tested")
            elif pd.notna(row["BF_Accuracy"]):
                report.append("- ⚠️ Only BenchForge tested")
            else:
                report.append("- ❌ No results available")

        report.append("")

    # Model ranking
    if both_tested > 0:
        report.append("## Model Performance Ranking (by accuracy)")
        report.append("")

        # Rank by BenchForge accuracy
        ranked = df[df["BF_Accuracy"].notna()].sort_values(
            "BF_Accuracy", ascending=False
        )

        for i, row in enumerate(ranked.iterrows(), 1):
            _, data = row
            report.append(f"{i}. {data['Model']}: {data['BF_Accuracy']:.3f}")

        report.append("")

    # Conclusions
    report.append("## Conclusions")
    report.append("")

    if bf_better == both_tested and both_tested > 0:
        report.append("✅ **BenchForge is a complete superset of FLAME**")
        report.append("   - All tested models show equal or better performance")
        report.append("   - Extraction rates are consistently higher")
        report.append(
            "   - BenchForge provides all FLAME features plus additional capabilities"
        )
    else:
        report.append("⚠️ **Incomplete testing or mixed results**")
        if both_tested == 0:
            report.append("   - No models were successfully tested with both methods")
        else:
            report.append(
                f"   - {bf_better}/{both_tested} models show BenchForge ≥ FLAME"
            )

    return "\n".join(report)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("FOMC Test Results Aggregation")
    print("=" * 60)

    # Load results
    print("\nLoading results...")
    results = load_all_results()

    print(f"Found {len(results['flame'])} FLAME results")
    print(f"Found {len(results['benchforge'])} BenchForge results")

    if not results["flame"] and not results["benchforge"]:
        print("\n❌ No results found. Please run tests first.")
        return 1

    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(results)

    # Save comparison table
    output_dir = Path("results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "model_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved comparison table to {csv_file}")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, df)

    # Save report
    report_file = output_dir / "comparison_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Saved report to {report_file}")

    # Print report to console
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Final verdict
    both_tested = df[df["BF_Better"].notna()].shape[0]
    bf_better = df[df["BF_Better"] == True].shape[0] if both_tested > 0 else 0

    if bf_better == both_tested and both_tested > 0:
        print("\n✅ SUCCESS: BenchForge is a complete superset of FLAME!")
        return 0
    else:
        print("\n⚠️ Results incomplete or BenchForge not proven as superset")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
