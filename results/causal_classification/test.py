import pandas as pd

df = pd.read_csv('/Users/hp/Desktop/SuperFLUE/results/causal_classification/causal_classification_meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo_02_10_2024.csv')

print(df['llm_responses'][0])
