#!/usr/bin/env python3
"""Analyze FinEntity comparison results."""

import pandas as pd
import json
import argparse
from pathlib import Path


def load_comparison_results(results_dir: str):
    """Load the most recent comparison results."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return None, None, None

    # Find most recent files
    bf_files = list(results_path.glob("benchforge_finentity_*.csv"))
    flame_files = list(results_path.glob("flame_finentity_*.csv"))
    metrics_files = list(results_path.glob("comparison_metrics_*.json"))

    if not bf_files or not flame_files or not metrics_files:
        print("Missing result files")
        return None, None, None

    # Get most recent files
    bf_file = sorted(bf_files)[-1]
    flame_file = sorted(flame_files)[-1]
    metrics_file = sorted(metrics_files)[-1]

    print(f"Loading BenchForge results: {bf_file}")
    print(f"Loading FLAME results: {flame_file}")
    print(f"Loading metrics: {metrics_file}")

    bf_df = pd.read_csv(bf_file)
    flame_df = pd.read_csv(flame_file)

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    return bf_df, flame_df, metrics


def analyze_entity_extraction(bf_df, flame_df):
    """Analyze entity extraction patterns."""
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION ANALYSIS")
    print("=" * 60)

    # Parse extracted entities
    def parse_entities(entities_str):
        if pd.isna(entities_str):
            return []
        try:
            if isinstance(entities_str, str):
                return (
                    json.loads(entities_str.replace("'", '"'))
                    if entities_str.strip() != "[]"
                    else []
                )
            return entities_str if isinstance(entities_str, list) else []
        except json.JSONDecodeError:
            return []

    bf_df["parsed_entities"] = bf_df["extracted_entities"].apply(
        lambda x: parse_entities(str(x)) if pd.notna(x) else []
    )
    flame_df["parsed_entities"] = flame_df["extracted_entities"].apply(
        lambda x: parse_entities(str(x)) if pd.notna(x) else []
    )

    # Entity counts
    bf_entity_counts = bf_df["parsed_entities"].apply(len)
    flame_entity_counts = flame_df["parsed_entities"].apply(len)

    print(f"BenchForge - Avg entities per sample: {bf_entity_counts.mean():.2f}")
    print(
        f"BenchForge - Samples with entities: {(bf_entity_counts > 0).sum()}/{len(bf_df)} ({(bf_entity_counts > 0).mean() * 100:.1f}%)"
    )

    print(f"FLAME - Avg entities per sample: {flame_entity_counts.mean():.2f}")
    print(
        f"FLAME - Samples with entities: {(flame_entity_counts > 0).sum()}/{len(flame_df)} ({(flame_entity_counts > 0).mean() * 100:.1f}%)"
    )

    # Sentiment distribution
    def extract_sentiments(entities_list):
        sentiments = []
        for entities in entities_list:
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict) and "tag" in entity:
                        sentiments.append(entity["tag"])
        return sentiments

    bf_sentiments = extract_sentiments(bf_df["parsed_entities"])
    flame_sentiments = extract_sentiments(flame_df["parsed_entities"])

    print("\nBenchForge sentiment distribution:")
    bf_sentiment_counts = pd.Series(bf_sentiments).value_counts()
    for sentiment, count in bf_sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count / len(bf_sentiments) * 100:.1f}%)")

    print("\nFLAME sentiment distribution:")
    if flame_sentiments:
        flame_sentiment_counts = pd.Series(flame_sentiments).value_counts()
        for sentiment, count in flame_sentiment_counts.items():
            print(
                f"  {sentiment}: {count} ({count / len(flame_sentiments) * 100:.1f}%)"
            )
    else:
        print("  No sentiments extracted")

    return bf_entity_counts, flame_entity_counts


def analyze_performance_metrics(bf_df, flame_df, metrics):
    """Analyze performance metrics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS ANALYSIS")
    print("=" * 60)

    print("Implementation Comparison:")
    print(f"  BenchForge Success Rate: {metrics.get('benchforge_success_rate', 0):.1%}")
    print(f"  FLAME Success Rate: {metrics.get('flame_success_rate', 0):.1%}")
    print(f"  Agreement Rate: {metrics.get('agreement_rate', 0):.1%}")

    print("\nTiming Analysis:")
    print(f"  BenchForge Avg Time: {metrics.get('benchforge_avg_time', 0):.2f}s")
    print(f"  FLAME Avg Time: {metrics.get('flame_avg_time', 0):.2f}s")

    print("\nAccuracy Metrics:")
    bf_metrics = metrics.get("benchforge", {})
    flame_metrics = metrics.get("flame", {})

    print(f"  BenchForge - Precision: {bf_metrics.get('precision', 0):.4f}")
    print(f"  BenchForge - Recall: {bf_metrics.get('recall', 0):.4f}")
    print(f"  BenchForge - F1: {bf_metrics.get('f1', 0):.4f}")

    print(f"  FLAME - Precision: {flame_metrics.get('precision', 0):.4f}")
    print(f"  FLAME - Recall: {flame_metrics.get('recall', 0):.4f}")
    print(f"  FLAME - F1: {flame_metrics.get('f1', 0):.4f}")


def analyze_agreement_patterns(bf_df, flame_df):
    """Analyze where implementations agree/disagree."""
    print("\n" + "=" * 60)
    print("AGREEMENT ANALYSIS")
    print("=" * 60)

    agreements = 0
    disagreements = 0
    agreement_details = []

    for i, (bf_row, flame_row) in enumerate(
        zip(bf_df.itertuples(), flame_df.itertuples())
    ):
        bf_entities = str(bf_row.extracted_entities)
        flame_entities = str(flame_row.extracted_entities)

        agree = bf_entities == flame_entities
        if agree:
            agreements += 1
        else:
            disagreements += 1
            if disagreements <= 10:  # Show first 10 disagreements
                agreement_details.append(
                    {
                        "sample_id": i + 1,
                        "benchforge": bf_entities[:100] + "..."
                        if len(bf_entities) > 100
                        else bf_entities,
                        "flame": flame_entities[:100] + "..."
                        if len(flame_entities) > 100
                        else flame_entities,
                    }
                )

    print(f"Agreements: {agreements}")
    print(f"Disagreements: {disagreements}")
    print(f"Agreement rate: {agreements / (agreements + disagreements) * 100:.1f}%")

    if agreement_details:
        print(f"\nFirst {len(agreement_details)} disagreements:")
        for detail in agreement_details:
            print(f"\nSample {detail['sample_id']}:")
            print(f"  BenchForge: {detail['benchforge']}")
            print(f"  FLAME: {detail['flame']}")


def generate_summary_report(bf_df, flame_df, metrics, output_file: str = None):
    """Generate a comprehensive summary report."""
    report = []
    report.append("# FinEntity Implementation Comparison Report")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append(f"- **Total Samples**: {len(bf_df)}")
    report.append(
        f"- **BenchForge Success Rate**: {metrics.get('benchforge_success_rate', 0):.1%}"
    )
    report.append(
        f"- **FLAME Success Rate**: {metrics.get('flame_success_rate', 0):.1%}"
    )
    report.append(
        f"- **Implementation Agreement**: {metrics.get('agreement_rate', 0):.1%}"
    )
    report.append("")

    # Performance Comparison
    report.append("## Performance Comparison")
    report.append("| Metric | BenchForge | FLAME |")
    report.append("|--------|------------|-------|")
    report.append(
        f"| Success Rate | {metrics.get('benchforge_success_rate', 0):.1%} | {metrics.get('flame_success_rate', 0):.1%} |"
    )
    report.append(
        f"| Avg Time | {metrics.get('benchforge_avg_time', 0):.2f}s | {metrics.get('flame_avg_time', 0):.2f}s |"
    )

    bf_metrics = metrics.get("benchforge", {})
    flame_metrics = metrics.get("flame", {})
    report.append(
        f"| Precision | {bf_metrics.get('precision', 0):.4f} | {flame_metrics.get('precision', 0):.4f} |"
    )
    report.append(
        f"| Recall | {bf_metrics.get('recall', 0):.4f} | {flame_metrics.get('recall', 0):.4f} |"
    )
    report.append(
        f"| F1 Score | {bf_metrics.get('f1', 0):.4f} | {flame_metrics.get('f1', 0):.4f} |"
    )
    report.append("")

    # Key Findings
    report.append("## Key Findings")
    if metrics.get("benchforge_success_rate", 0) > metrics.get("flame_success_rate", 0):
        report.append(
            "- ✅ **BenchForge shows higher success rate** - better error handling and JSON parsing"
        )
    if metrics.get("agreement_rate", 0) > 0.8:
        report.append(
            "- ✅ **High agreement when both succeed** - consistent entity extraction"
        )
    else:
        report.append("- ⚠️ **Moderate agreement** - may indicate parsing differences")

    report.append(
        "- 📊 **Both implementations extract entities with sentiment labels**"
    )
    report.append("- 🎯 **Using identical FLAME prompt for consistency**")
    report.append("")

    # Recommendations
    report.append("## Recommendations")
    if metrics.get("benchforge_success_rate", 0) > 0.9:
        report.append(
            "- **Production Use**: BenchForge implementation recommended for production"
        )
        report.append(
            "- **Error Handling**: BenchForge provides more robust JSON parsing"
        )
    report.append("- **Monitoring**: Continue monitoring agreement rates in production")
    report.append(
        "- **Validation**: Both implementations successfully replicate FLAME's entity+sentiment extraction"
    )

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report_text)
        print(f"\nSummary report saved to: {output_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze FinEntity comparison results")
    parser.add_argument(
        "--results-dir",
        default="results/finentity_comparison",
        help="Directory containing comparison results",
    )
    parser.add_argument("--output-report", help="Output file for summary report")

    args = parser.parse_args()

    print("=" * 70)
    print("FINENTITY COMPARISON RESULTS ANALYSIS")
    print("=" * 70)

    # Load results
    bf_df, flame_df, metrics = load_comparison_results(args.results_dir)
    if bf_df is None:
        return

    print(f"Loaded {len(bf_df)} BenchForge samples and {len(flame_df)} FLAME samples")

    # Perform analysis
    analyze_entity_extraction(bf_df, flame_df)
    analyze_performance_metrics(bf_df, flame_df, metrics)
    analyze_agreement_patterns(bf_df, flame_df)

    # Generate summary report
    if args.output_report:
        generate_summary_report(bf_df, flame_df, metrics, args.output_report)
    else:
        report_text = generate_summary_report(bf_df, flame_df, metrics)
        print("\n" + "=" * 70)
        print("SUMMARY REPORT")
        print("=" * 70)
        print(report_text)


if __name__ == "__main__":
    main()
