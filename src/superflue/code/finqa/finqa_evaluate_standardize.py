import pandas as pd
from pathlib import Path


def standardize_eval(file_path):
    df = pd.read_csv(file_path)
    judge = df["evaluation_response"].tolist()
    judge = [i.lower() for i in judge]
    answers = []
    for i in judge:
        if i.find("correct") != -1 and (
            i.find("wrong") == -1 or i.find("correct") < i.find("wrong")
        ):
            answers.append(True)
        else:
            answers.append(False)

    accuracy = len([answer for answer in answers if answer]) / len(answers)
    print(f"Accuracy: {accuracy:.4f}")

    metrics_df = pd.DataFrame(
        {
            "metric": ["accuracy"],
            "value": [accuracy],
        }
    )

    metrics_results_path = Path(f"{str(file_path)[:-4]}_statistics.csv")
    metrics_df.to_csv(metrics_results_path, index=False)


if __name__ == "__main__":
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_openai/o1-mini_09_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_gemini/gemini-1.5-pro-latest_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/deepseek-ai/DeepSeek-r1_09_02_2025_no_think.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_ai21/jamba-1.5-large_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_ai21/jamba-1.5-mini_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_anthropic/claude-3-5-sonnet-20240620_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_anthropic/claude-3-haiku-20240307_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_cohere_chat/command-r-plus-08-2024_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_cohere_chat/command-r7b-12-2024_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_openai/gpt-4o-2024-08-06_07_02_2025.csv"
    )
    # standardize_eval("../../../../evaluation_results/finqa/finqa_openai/o1-mini_07_02_2025.csv")
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/deepseek-ai/DeepSeek-V3_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/Qwen/QwQ-32B-Preview_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/meta-llama/Llama-3-70b-chat-hf_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/meta-llama/Llama-3-8b-chat-hf_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/meta-llama/Llama-2-13b-chat-hf_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/databricks/dbrx-instruct_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/deepseek-ai/deepseek-llm-67b-chat_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/google/gemma-2-27b-it_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/google/gemma-2-9b-it_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/mistralai/Mistral-7B-Instruct-v0.3_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/Qwen/Qwen2-72B-Instruct_07_02_2025.csv"
    )
    standardize_eval(
        "../../../../evaluation_results/finqa/finqa_together_ai/microsoft/WizardLM-2-8x22B_07_02_2025.csv"
    )
