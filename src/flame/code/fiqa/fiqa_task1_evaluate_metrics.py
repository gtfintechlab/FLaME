import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import glob
import os


def evaluate_regression_metrics(file_path):
    """Evaluate regression metrics for FiQA Task 1 results.

    Args:
        file_path: Path to the CSV file with results

    Returns:
        tuple: (DataFrame with results, metrics DataFrame)
    """
    df = pd.read_csv(file_path)

    actual = df["actual_sentiment"].tolist()
    predicted = df["regex_extraction"].tolist()
    count_missing = 0

    for i in range(len(actual)):
        if np.isnan(predicted[i]):
            count_missing += 1
            if actual[i] >= 0:
                predicted[i] = actual[i] - 2
            else:
                predicted[i] = actual[i] + 2

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    answer_coverage = (len(actual) - count_missing) / len(actual)

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Squared Error",
                "Mean Absolute Error",
                "R2 Score",
                "Answer Coverage",
            ],
            "Value": [mse, mae, r2, answer_coverage],
        }
    )

    return df, metrics_df


def fiqa_task1_evaluate_metrics(file_name, args):
    """Evaluate FiQA Task 1 results using the standardized test interface.

    Args:
        file_name (str): Path to the results CSV file
        args: Configuration parameters

    Returns:
        tuple: (DataFrame of results, DataFrame of metrics)
    """
    # Run regression metrics evaluation
    df, metrics_df = evaluate_regression_metrics(file_name)

    # Get output path from args if provided, otherwise derive from input path
    output_path = getattr(args, "output_path", None)
    if output_path is None:
        output_path = Path(f"{str(file_name)[:-4]}_statistics.csv")
    else:
        output_path = Path(output_path)

    # Create parent directory for output if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics to CSV
    metrics_df.to_csv(output_path, index=False)

    return df, metrics_df


if __name__ == "__main__":
    # Base evaluation results directory
    base_results_dir = Path("../../../../evaluation_results/fiqa_task1")

    # Dynamically collect CSV files that don't already have statistics files
    csv_files = []
    for csv_path in glob.glob(f"{base_results_dir}/**/*.csv", recursive=True):
        # Skip files that end with _statistics.csv or already have a corresponding statistics file
        if csv_path.endswith("_statistics.csv"):
            continue

        stats_path = f"{csv_path[:-4]}_statistics.csv"
        if not os.path.exists(stats_path):
            csv_files.append(csv_path)

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each file
    for file_path in csv_files:
        try:
            df, metrics_df = evaluate_regression_metrics(file_path)
            metrics_results_path = Path(f"{str(file_path)[:-4]}_statistics.csv")
            metrics_df.to_csv(metrics_results_path, index=False)
            print(f"Processed {file_path}")
            print(f"  - MSE: {metrics_df.iloc[0, 1]:.4f}")
            print(f"  - MAE: {metrics_df.iloc[1, 1]:.4f}")
            print(f"  - R2: {metrics_df.iloc[2, 1]:.4f}")
            print(f"  - Coverage: {metrics_df.iloc[3, 1]:.2%}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
