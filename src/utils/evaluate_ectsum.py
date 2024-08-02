# utils/evaluate_metrics.py

import bert_score
import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer

from utils.formatter_ectsum import Formatter


class EvaluateMetrics:
    def __init__(self) -> None:
        pass

    def rouge_scores(self, reference, prediction):
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        reference_text = " ".join(reference)
        prediction_text = " ".join(prediction)

        scores = scorer.score(reference_text, prediction_text)

        return [scores["rouge1"][2], scores["rouge2"][2], scores["rougeL"][2]]

    def bert_score(self, references, predictions):
        bertscore = load("bertscore")
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            model_type="distilbert-base-uncased",
        )

        return results["f1"]

    def append_scores(self, file):
        df = pd.read_csv(file)
        formatter = Formatter()
        df = formatter.format_df(df)

        for i, row in df.iterrows():
            reference = row["output"]
            prediction = row["predicted_text"]

            rouge_scores = self.rouge_scores(reference, prediction)
            bert_scores = self.bert_score(reference, prediction)

            df.at[i, "ROUGE-1"] = rouge_scores[0]
            df.at[i, "ROUGE-2"] = rouge_scores[1]
            df.at[i, "ROUGE-L"] = rouge_scores[2]
            df.at[i, "BERTScore"] = bert_scores

        df.to_csv(file, index=False)
