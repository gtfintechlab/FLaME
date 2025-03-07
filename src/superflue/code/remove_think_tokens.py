import pandas as pd

def remove_think_tokens(file_path):
    df = pd.read_csv(file_path)
    df['llm_responses'] = df['llm_responses'].apply(lambda x : x[(x.find('</think>') + 8):])
    df.to_csv(file_path[:-4] + "_no_think.csv", index=False)

if __name__ == "__main__":
    remove_think_tokens('results/finer/finer_together_ai/deepseek-ai/DeepSeek-r1_09_02_2025.csv')
    # remove_think_tokens('results/banking77/banking77_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025.csv')
    # remove_think_tokens('results/causal_detection/causal_detection_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025.csv')
    # remove_think_tokens('results/edtsum/edtsum_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025.csv')
    # remove_think_tokens('results/finbench/finbench_together_ai/deepseek-ai/DeepSeek-r1_09_02_2025.csv')
    # remove_think_tokens('results/finred/finred_together_ai/deepseek-ai/DeepSeek-r1_09_02_2025.csv')

    # remove_think_tokens('results/fiqa_task1/fiqa_task1_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025.csv')
    # remove_think_tokens('results/fiqa_task2/fiqa_task2_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025.csv')
    # remove_think_tokens('results/fpb/fpb_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025.csv')
    # remove_think_tokens('results/refind/refind_together_ai/deepseek-ai/DeepSeek-r1_09_02_2025.csv')
    # remove_think_tokens('results/headlines/headlines_together_ai/deepseek-ai/DeepSeek-r1_08_02_2025.csv')
    # remove_think_tokens('results/finqa/finqa_together_ai/deepseek-ai/DeepSeek-r1_09_02_2025.csv')