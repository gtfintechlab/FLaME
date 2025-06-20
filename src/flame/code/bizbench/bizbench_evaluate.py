import numpy as np
import pandas as pd


def calculate_accuracy_and_stats(file_path, tolerance=0.01):
    """Calculate accuracy and other metrics for BizBench results.

    Args:
        file_path: Path to the CSV file with results
        tolerance: Tolerance threshold for approximate matching (default: 0.01)

    Returns:
        Dictionary with accuracy metrics
    """
    df = pd.read_csv(file_path)

    # Function to clean and convert values to floats where possible
    def to_float(value):
        try:
            # Remove commas and convert to float
            return float(str(value).replace(",", ""))
        except ValueError:
            # Return NaN for non-numeric values
            return np.nan

    # Apply the conversion function to y_answer and llm_responses columns
    df["y_answer"] = df["y_answer"].apply(to_float)
    df["llm_responses"] = df["llm_responses"].apply(to_float)

    # Drop rows with NaN values in either y_answer or llm_responses
    df = df.dropna(subset=["y_answer", "llm_responses"])

    # Calculate exact match accuracy
    total = len(df)
    correct = sum(df["y_answer"] == df["llm_responses"])
    accuracy = correct / total if total > 0 else 0

    # Calculate differences for numerical statistics
    y_true = df["y_answer"]
    y_pred = df["llm_responses"]
    differences = np.abs(y_true - y_pred)

    # Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
    mae = differences.mean()
    mse = (differences**2).mean()

    # Percentage of predictions within the specified tolerance
    within_tolerance = (differences <= tolerance).sum()
    tolerance_accuracy = within_tolerance / total if total > 0 else 0

    return {
        "Exact Match Accuracy": accuracy,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Tolerance Accuracy": tolerance_accuracy,
    }


def bizbench_evaluate(file_name, args):
    """Evaluate BizBench results using the standardized test interface.

    Args:
        file_name (str): Path to the results CSV file
        args: Configuration parameters which may include tolerance

    Returns:
        tuple: (DataFrame of results, DataFrame of metrics)
    """
    # Extract tolerance parameter if available, otherwise use default
    tolerance = getattr(args, "tolerance", 0.01)

    # Call the original calculation function
    metrics_dict = calculate_accuracy_and_stats(file_name, tolerance)

    # Read the original results file
    df = pd.read_csv(file_name)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(
        {"Metric": list(metrics_dict.keys()), "Value": list(metrics_dict.values())}
    )

    return df, metrics_df


# Only execute this when running the script directly
if __name__ == "__main__":
    # Example usage
    stats = calculate_accuracy_and_stats("path_to_your_file.csv")
    for metric, value in stats.items():
        print(f"{metric}: {value:.4f}")
