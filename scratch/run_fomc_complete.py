#!/usr/bin/env python3
"""Complete FOMC dataset run with comprehensive visualizations.

This script runs the entire FOMC test dataset (496 samples) through BenchForge
with real LLM API calls and generates detailed visualizations.
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add BenchForge to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from bench_forge.flame.tasks.fomc import FOMCConfig, FOMCTask  # noqa: E402
from bench_forge.output.manager import OutputManager  # noqa: E402
from bench_forge.llm.client import LLMClient  # noqa: E402
from bench_forge.llm.config import LLMConfig  # noqa: E402
from bench_forge.tasks.config import PromptFormat  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("fomc_complete_run.log")],
)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def process_batch_with_llm(
    llm_client: LLMClient, prompts: List[str], batch_idx: int, total_batches: int
) -> Tuple[List[str], float]:
    """Process a batch of prompts through the LLM.

    Returns:
        Tuple of (responses, elapsed_time)
    """
    logger.info(
        f"Processing batch {batch_idx + 1}/{total_batches} ({len(prompts)} samples)"
    )

    start_time = time.time()
    try:
        # Call LLM with batch
        responses = llm_client.complete_batch(prompts)

        # Add small delay to avoid rate limiting
        time.sleep(0.5)

        elapsed = time.time() - start_time
        return responses, elapsed

    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {e}")
        elapsed = time.time() - start_time
        return ["ERROR: " + str(e)] * len(prompts), elapsed


def create_confusion_matrix_plot(
    y_true: List[int], y_pred: List[int], labels: List[str], output_dir: Path
) -> None:
    """Create and save confusion matrix visualization."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    # Create plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title="Confusion Matrix - FOMC Sentiment Classification",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=600,
        height=500,
    )

    fig.write_html(output_dir / "confusion_matrix.html")
    fig.write_image(output_dir / "confusion_matrix.png")
    logger.info(f"Saved confusion matrix to {output_dir}")


def create_performance_dashboard(
    metrics_df: pd.DataFrame, batch_times: List[float], output_dir: Path
) -> None:
    """Create comprehensive performance dashboard."""

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Overall Metrics",
            "Per-Class F1 Scores",
            "Batch Processing Times",
            "Precision vs Recall",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # Overall metrics bar chart
    overall_metrics = metrics_df[
        metrics_df["Metric"].isin(["Accuracy", "Precision", "Recall", "F1 Score"])
    ]
    fig.add_trace(
        go.Bar(
            x=overall_metrics["Metric"],
            y=overall_metrics["Value"],
            marker_color="lightblue",
            name="Overall",
        ),
        row=1,
        col=1,
    )

    # Per-class F1 scores
    f1_metrics = metrics_df[metrics_df["Metric"].str.contains("F1 ")]
    classes = [m.replace("F1 ", "") for m in f1_metrics["Metric"]]
    fig.add_trace(
        go.Bar(
            x=classes,
            y=f1_metrics["Value"].values,
            marker_color=["green", "orange", "purple"],
            name="F1 by Class",
        ),
        row=1,
        col=2,
    )

    # Batch processing times
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(batch_times) + 1)),
            y=batch_times,
            mode="lines+markers",
            name="Batch Time",
            marker=dict(size=8),
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    # Precision vs Recall scatter
    precision_metrics = metrics_df[metrics_df["Metric"].str.contains("Precision ")]
    recall_metrics = metrics_df[metrics_df["Metric"].str.contains("Recall ")]

    if len(precision_metrics) > 0 and len(recall_metrics) > 0:
        classes = [m.replace("Precision ", "") for m in precision_metrics["Metric"]]
        fig.add_trace(
            go.Scatter(
                x=recall_metrics["Value"].values,
                y=precision_metrics["Value"].values,
                mode="markers+text",
                text=classes,
                textposition="top center",
                marker=dict(size=15, color=["green", "orange", "purple"]),
                name="Classes",
            ),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title_text="FOMC Classification Performance Dashboard",
        showlegend=False,
        height=800,
        width=1200,
    )

    fig.update_xaxes(title_text="Metric", row=1, col=1)
    fig.update_xaxes(title_text="Class", row=1, col=2)
    fig.update_xaxes(title_text="Batch Number", row=2, col=1)
    fig.update_xaxes(title_text="Recall", row=2, col=2)

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="F1 Score", row=1, col=2)
    fig.update_yaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Precision", row=2, col=2)

    fig.write_html(output_dir / "performance_dashboard.html")
    logger.info(f"Saved performance dashboard to {output_dir}")


def create_distribution_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create distribution plots for predictions."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # True label distribution
    true_counts = results_df["ground_truth"].value_counts()
    axes[0, 0].bar(true_counts.index, true_counts.values, color="skyblue")
    axes[0, 0].set_title("True Label Distribution")
    axes[0, 0].set_xlabel("Label")
    axes[0, 0].set_ylabel("Count")

    # Predicted label distribution
    pred_counts = results_df["extracted_labels_text"].value_counts()
    axes[0, 1].bar(pred_counts.index, pred_counts.values, color="lightcoral")
    axes[0, 1].set_title("Predicted Label Distribution")
    axes[0, 1].set_xlabel("Label")
    axes[0, 1].set_ylabel("Count")

    # Correct vs Incorrect predictions
    results_df["correct"] = (
        results_df["extracted_labels_text"] == results_df["ground_truth"]
    )
    correct_counts = results_df["correct"].value_counts()
    axes[1, 0].pie(
        correct_counts.values,
        labels=["Incorrect", "Correct"],
        autopct="%1.1f%%",
        colors=["salmon", "lightgreen"],
    )
    axes[1, 0].set_title("Prediction Accuracy")

    # Confidence distribution (if available)
    # For now, we'll show response length distribution as proxy
    results_df["response_length"] = results_df["raw_response"].str.len()
    axes[1, 1].hist(
        results_df["response_length"], bins=30, color="gold", edgecolor="black"
    )
    axes[1, 1].set_title("Response Length Distribution")
    axes[1, 1].set_xlabel("Response Length (characters)")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution plots to {output_dir}")


def create_error_analysis_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create detailed error analysis report."""

    # Identify errors
    errors_df = results_df[
        results_df["extracted_labels_text"] != results_df["ground_truth"]
    ].copy()

    if len(errors_df) == 0:
        logger.info("No errors to analyze!")
        return

    # Error patterns
    error_patterns = []
    for _, row in errors_df.iterrows():
        error_patterns.append(
            {
                "true_label": row["ground_truth"],
                "predicted_label": row["extracted_labels_text"],
                "input_preview": row["input"][:100] + "..."
                if len(row["input"]) > 100
                else row["input"],
            }
        )

    error_df = pd.DataFrame(error_patterns)

    # Create error transition matrix
    error_matrix = pd.crosstab(error_df["true_label"], error_df["predicted_label"])

    # Create visualization
    fig = px.imshow(
        error_matrix.values,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=error_matrix.columns,
        y=error_matrix.index,
        title="Error Pattern Matrix",
        color_continuous_scale="Reds",
    )

    fig.write_html(output_dir / "error_analysis.html")

    # Save error samples to CSV
    error_df.to_csv(output_dir / "error_samples.csv", index=False)
    logger.info(f"Saved error analysis to {output_dir}")


def run_complete_fomc_dataset():
    """Run complete FOMC dataset with visualizations."""

    # Check for API key
    if not os.getenv("TOGETHERAI_API_KEY"):
        logger.error("TOGETHERAI_API_KEY not found in environment")
        return False

    logger.info("=" * 60)
    logger.info("🚀 Starting Complete FOMC Dataset Run with Visualizations")
    logger.info("=" * 60)

    # Create configuration with batch size of 50
    config = FOMCConfig(
        name="fomc",
        dataset="fomc_communication",
        huggingface_dataset="gtfintechlab/fomc_communication",
        dataset_split="test",
        prompt_format=PromptFormat.ZERO_SHOT,
        batch_size=50,  # As requested
        max_tokens=128,
        temperature=0.0,
        top_p=0.9,
        seed=42,
        metrics=["accuracy", "f1", "precision", "recall"],
        model="together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )

    logger.info("Configuration:")
    logger.info(f"  - Model: {config.model}")
    logger.info(f"  - Dataset: {config.huggingface_dataset}")
    logger.info(f"  - Split: {config.dataset_split}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Max tokens: {config.max_tokens}")

    # Initialize components
    task = FOMCTask(config)
    output_manager = OutputManager(base_dir=Path("benchforge_results"))

    # Initialize LLM client
    llm_config = LLMConfig(
        provider="litellm",
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        api_key=os.getenv("TOGETHERAI_API_KEY"),
    )
    llm_client = LLMClient(llm_config)

    # Load dataset
    logger.info("\n📊 Loading dataset...")
    dataset = task.load_dataset("test")
    total_samples = len(dataset)
    logger.info(f"Loaded {total_samples} test samples")

    # Process ALL samples
    logger.info(f"\n🔄 Processing ALL {total_samples} samples")
    samples = [dataset[i] for i in range(total_samples)]

    # Generate prompts
    logger.info("Generating prompts...")
    prompts = task.process_batch(samples, config.prompt_format)

    # Process in batches of 50
    batch_size = config.batch_size
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    all_responses = []
    batch_times = []
    start_time = time.time()

    logger.info(
        f"\n🚀 Starting inference with {num_batches} batches of {batch_size} samples..."
    )

    # Process batches with progress bar
    with tqdm(total=len(prompts), desc="Processing samples") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]

            # Process batch through LLM
            batch_responses, batch_time = process_batch_with_llm(
                llm_client, batch_prompts, batch_idx, num_batches
            )

            all_responses.extend(batch_responses)
            batch_times.append(batch_time)
            pbar.update(len(batch_prompts))

            # Log progress every 2 batches
            if (batch_idx + 1) % 2 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / ((batch_idx + 1) * batch_size)
                logger.info(
                    f"Progress: {(batch_idx + 1) * batch_size}/{total_samples} samples, "
                    f"Avg time: {avg_time:.2f}s/sample"
                )

    elapsed_time = time.time() - start_time
    logger.info(f"\n⏱️  Inference completed in {elapsed_time:.2f} seconds")
    logger.info(f"Average time per sample: {elapsed_time / len(prompts):.2f} seconds")
    logger.info(f"Average time per batch: {np.mean(batch_times):.2f} seconds")

    # Extract labels from responses
    logger.info("\n🏷️  Extracting labels from responses...")
    extracted_labels = []
    for response in all_responses:
        label = task.extract_response(response)
        extracted_labels.append(label)

    valid_count = sum(1 for label in extracted_labels if label is not None)
    logger.info(
        f"Extracted {valid_count}/{len(extracted_labels)} valid labels ({valid_count / len(extracted_labels) * 100:.1f}%)"
    )

    # Format results with evaluation
    logger.info("\n📊 Calculating metrics...")
    results_dict = task.format_results_with_evaluation(
        samples, prompts, all_responses, extracted_labels
    )

    results_df = results_dict["results"]
    metrics_df = results_dict["metrics"]

    # Display key metrics
    logger.info("\n📈 Key Metrics:")
    for _, row in metrics_df.head(7).iterrows():
        if isinstance(row["Value"], float):
            logger.info(f"  - {row['Metric']}: {row['Value']:.3f}")
        else:
            logger.info(f"  - {row['Metric']}: {row['Value']}")

    # Create visualizations directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = output_manager.base_dir / "fomc" / f"visualizations_{timestamp}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    logger.info("\n📊 Generating visualizations...")

    # Prepare data for confusion matrix
    y_true = []
    y_pred = []
    label_names = ["DOVISH", "HAWKISH", "NEUTRAL"]

    for _, row in results_df.iterrows():
        if row["extracted_labels"] != -1:  # Valid prediction
            y_true.append(row["ground_truth"])
            y_pred.append(row["extracted_labels"])

    # Create all visualizations
    create_confusion_matrix_plot(y_true, y_pred, label_names, viz_dir)
    create_performance_dashboard(metrics_df, batch_times, viz_dir)
    create_distribution_plots(results_df, viz_dir)
    create_error_analysis_report(results_df, viz_dir)

    # Save results
    logger.info("\n💾 Saving results...")

    # Save main results
    results_path = output_manager.base_dir / "fomc" / f"fomc_complete_{timestamp}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    # Save metrics
    metrics_path = (
        output_manager.base_dir / "fomc" / f"metrics_complete_{timestamp}.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)

    # Save detailed metadata
    metadata = {
        "task": "fomc",
        "model": config.model,
        "dataset": config.huggingface_dataset,
        "prompt_format": config.prompt_format.value,
        "num_samples": total_samples,
        "batch_size": config.batch_size,
        "num_batches": num_batches,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_seconds": elapsed_time,
        "avg_time_per_sample": elapsed_time / len(prompts),
        "avg_time_per_batch": np.mean(batch_times),
        "min_batch_time": min(batch_times),
        "max_batch_time": max(batch_times),
        "valid_predictions": valid_count,
        "invalid_predictions": len(extracted_labels) - valid_count,
        "extraction_success_rate": valid_count / len(extracted_labels),
    }

    # Add final metrics to metadata
    for _, row in metrics_df.iterrows():
        if isinstance(row["Value"], (int, float)):
            metadata[f"metric_{row['Metric'].lower().replace(' ', '_')}"] = row["Value"]

    metadata_path = (
        output_manager.base_dir / "fomc" / f"metadata_complete_{timestamp}.json"
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Results saved to {results_path.parent}")
    logger.info(f"  - Results: {results_path.name}")
    logger.info(f"  - Metrics: {metrics_path.name}")
    logger.info(f"  - Metadata: {metadata_path.name}")
    logger.info(f"  - Visualizations: {viz_dir.name}/")

    # Summary report
    logger.info("\n" + "=" * 60)
    logger.info("🎉 FOMC Complete Dataset Run Finished!")
    logger.info("=" * 60)
    logger.info(f"Total samples: {total_samples}")
    logger.info(
        f"Valid extractions: {valid_count}/{len(extracted_labels)} ({valid_count / len(extracted_labels) * 100:.1f}%)"
    )
    logger.info(
        f"Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.1f} minutes)"
    )
    logger.info(f"Average per sample: {elapsed_time / len(prompts):.2f} seconds")
    logger.info(f"Average per batch: {np.mean(batch_times):.2f} seconds")
    logger.info(f"\nFinal Accuracy: {metadata.get('metric_accuracy', 0):.3f}")
    logger.info(f"Final F1 Score: {metadata.get('metric_f1_score', 0):.3f}")
    logger.info("\nVisualization outputs:")
    logger.info(f"  - Confusion Matrix: {viz_dir}/confusion_matrix.html")
    logger.info(f"  - Performance Dashboard: {viz_dir}/performance_dashboard.html")
    logger.info(f"  - Distribution Plots: {viz_dir}/distribution_plots.png")
    logger.info(f"  - Error Analysis: {viz_dir}/error_analysis.html")

    return True


if __name__ == "__main__":
    success = run_complete_fomc_dataset()
    sys.exit(0 if success else 1)
