# TODO: (Glenn) Evaluate code will need to be moved into its own folder not utils.
# pip install bert_score
# pip install evaluate
# pip install rouge-score

import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer


class Evaluate:

    def __init__(self) -> None:
        pass

    # Formatter to remove new line characters for easier evaluation
    def formatter(self, input):
        df = input
        for i, row in df.iterrows():
            # Ensure "output" is a list and contains only one sentence
            if not isinstance(row["output"], list):
                row["output"] = [
                    line.strip() for line in row["output"].split("\n") if line.strip()
                ]
                row["output"] = row["output"][:1]  # Keep only the first sentence

            # Ensure "predicted_text" is a list and contains only one sentence
            if not isinstance(row["predicted_text"], list):
                row["predicted_text"] = [
                    line.strip()
                    for line in row["predicted_text"].split("\n")
                    if line.strip()
                ]
                row["predicted_text"] = row["predicted_text"][
                    :1
                ]  # Keep only the first sentence

        return df

    # Rouge 1 and Rouge L scores
    def rougeScores(self, reference, prediction):
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        # Join the lists into strings
        reference_text = " ".join(reference)
        prediction_text = " ".join(prediction)

        # Score the strings
        scores = scorer.score(reference_text, prediction_text)

        rougelist = [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]
        return rougelist

    # Bert Score
    def bertScore(self, references, predictions):
        bertscore = load("bertscore")
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type="distilbert-base-uncased",
        )
        # print(results)
        return results["f1"]

    def append_scores(self, file):
        df = pd.read_csv(file)
        df = self.formatter(df)

        for i, row in df.iterrows():
            reference = row["output"]
            prediction = row["predicted_text"]

            rouge_scores = self.rougeScores(reference, prediction)
            bert_scores = self.bertScore(reference, prediction)

            # Append scores to the DataFrame
            df.at[i, "ROUGE-1"] = rouge_scores[0]
            df.at[i, "ROUGE-2"] = rouge_scores[1]
            df.at[i, "ROUGE-L"] = rouge_scores[2]
            df.at[i, "BERTScore"] = bert_scores
            print(bert_scores)

        df.to_csv(file, index=False)
