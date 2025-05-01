import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def evaluate_regression_metrics(file_path):
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
    print(answer_coverage)

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

    metrics_results_path = Path(f"{str(file_path)[:-4]}_statistics.csv")
    metrics_df.to_csv(metrics_results_path, index=False)


if __name__ == "__main__":
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_ai21/jamba-1.5-large_29_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_ai21/jamba-1.5-mini_29_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_anthropic/claude-3-haiku-20240307_29_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_anthropic/claude-3-5-sonnet-20240620_29_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_cohere_chat/command-r-plus-08-2024_29_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_cohere_chat/command-r7b-12-2024_29_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_gemini/gemini-1.5-pro-latest_09_02_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_openai/gpt-4o-2024-08-06_29_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_openai/o1-mini_09_02_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/databricks/dbrx-instruct_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/deepseek-ai/deepseek-llm-67b-chat_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025_no_think.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/deepseek-ai/DeepSeek-V3_30_01_2025.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/google/gemma-2-9b-it_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/google/gemma-2-27b-it_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/meta-llama/Llama-2-13b-chat-hf_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/meta-llama/Llama-3-8b-chat-hf_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/meta-llama/Llama-3-70b-chat-hf_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/microsoft/WizardLM-2-8x22B_12_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/mistralai/Mistral-7B-Instruct-v0.3_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/Qwen/Qwen2-72B-Instruct_10_12_2024.csv"
    )
    evaluate_regression_metrics(
        "../../../../evaluation_results/fiqa_task1/fiqa_task1_together_ai/Qwen/QwQ-32B-Preview_30_01_2025.csv"
    )
