import pandas as pd
from datasets import load_dataset

# TODO: (Glenn) If possible, avoid using `nltk`, unless we need something specific not found elsewhere
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

# TODO: (Glenn) `word_count.py` is written very strictly and should be either a utility we call on a dataset we provide, or something done at runtime, etc.

# Load the dataset
dataset = load_dataset("gtfintechlab/fomc_communication", trust_remote_code=True)

# Initialize containers for the data
train_context, train_actual_labels = [], []

# Define stop words
stop_words = set(stopwords.words("english"))

# Process each observation in the dataset
context_stop_words_counts = []
context_non_stop_words_counts = []

for sentence in dataset["train"]:
    # Append the context and actual label
    context = sentence["sentence"]
    train_context.append(context)
    train_actual_label = sentence["label"]
    train_actual_labels.append(train_actual_label)

    # Tokenize context
    context_words = word_tokenize(context)

    # Count stop and non-stop words in context
    context_stop_words_count = sum(
        1 for word in context_words if word.lower() in stop_words
    )
    context_non_stop_words_count = len(context_words) - context_stop_words_count

    context_stop_words_counts.append(context_stop_words_count)
    context_non_stop_words_counts.append(context_non_stop_words_count)

# Create the DataFrame
train_df = pd.DataFrame(
    {
        "context": train_context,
        "context_stop_words_count": context_stop_words_counts,
        "context_non_stop_words_count": context_non_stop_words_counts,
        "actual_label": train_actual_labels,
    }
)
