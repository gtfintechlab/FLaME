import pandas as pd
import numpy as np

def calculate_accuracy_and_stats(file_path, tolerance=0.01):
    df = pd.read_csv(file_path)

    # Function to clean and convert values to floats where possible
    def to_float(value):
        try:
            # Remove commas and convert to float
            return float(str(value).replace(',', ''))
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
    mse = (differences ** 2).mean()

    # Percentage of predictions within the specified tolerance
    within_tolerance = (differences <= tolerance).sum()
    tolerance_accuracy = within_tolerance / total if total > 0 else 0

    return {
        "Exact Match Accuracy": accuracy,
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Tolerance Accuracy": tolerance_accuracy,
    }

# file_path = "/home/thans/Documents/Study/fall24/cs7643/superflue_cs7643/output/results/bizbench/bizbench_meta-llama/Meta-Llama-3-70B-Instruct-Turbo_02_12_2024.csv"
stats = calculate_accuracy_and_stats(file_path)
for metric, value in stats.items():
    print(f"{metric}: {value:.4f}")
