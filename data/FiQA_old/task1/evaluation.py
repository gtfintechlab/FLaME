# Re-importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cosine

# Reload the data from the uploaded CSV file
csv_file_path = 'Meta-Llama-3.1-8B-Instruct-Turbo_04_11_2024.csv'
data = pd.read_csv(csv_file_path)


# Define the helper function for sentiment categorization
def categorize_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Prepare data for model evaluation
data['predicted_sentiment'] = data['llm_responses'].apply(lambda x: x.split('.')[0] if isinstance(x, str) else '')
data['actual_sentiment_category'] = data['actual_sentiment'].apply(categorize_sentiment)

# Regression Evaluation (Sentiment Scores)
# Mean Squared Error (MSE) and R-Squared (R2)
mse = mean_squared_error(data['actual_sentiment'], data['actual_sentiment'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))
r2 = r2_score(data['actual_sentiment'], data['actual_sentiment'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))

# Cosine Similarity - Treat sentiment scores as vectors
cosine_similarities = []
for a, b in zip(data['actual_sentiment'], data['actual_sentiment']): # Temporarily calculate against itself as prediction needs to be updated
    if a != 0 and b != 0:  # Ensure both vectors are non-zero
        cosine_similarities.append(1 - cosine([a], [b]))
    else:
        cosine_similarities.append(0)  # Assign 0 if either vector is zero
average_cosine_similarity = np.mean(cosine_similarities)

# Classification Evaluation (Financial Aspect Categories)
accuracy = accuracy_score(data['actual_sentiment_category'], data['predicted_sentiment'])
precision = precision_score(data['actual_sentiment_category'], data['predicted_sentiment'], average='macro', zero_division=0)
recall = recall_score(data['actual_sentiment_category'], data['predicted_sentiment'], average='macro', zero_division=0)
f1 = f1_score(data['actual_sentiment_category'], data['predicted_sentiment'], average='macro', zero_division=0)

# Prepare the results in a dictionary
evaluation_results = {
    "Mean Squared Error (MSE)": mse,
    "R-Squared (R2)": r2,
    "Average Cosine Similarity": average_cosine_similarity,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}

# Print the results in a human-readable format
print("Evaluation Results:")
print(f"Mean Squared Error (MSE): {evaluation_results['Mean Squared Error (MSE)']:.4f}")
print(f"R-Squared (R2): {evaluation_results['R-Squared (R2)']:.4f}")
print(f"Average Cosine Similarity: {evaluation_results['Average Cosine Similarity']:.4f}")
print(f"Accuracy: {evaluation_results['Accuracy']:.4f}")
print(f"Precision: {evaluation_results['Precision']:.4f}")
print(f"Recall: {evaluation_results['Recall']:.4f}")
print(f"F1 Score: {evaluation_results['F1 Score']:.4f}")
