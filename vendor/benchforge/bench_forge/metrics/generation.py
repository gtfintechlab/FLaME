"""Generation metrics for text generation tasks."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import Counter
import string

from bench_forge.metrics.base import BaseMetric, AveragedMetric


logger = logging.getLogger(__name__)


# Try to import advanced metrics libraries
try:
    from rouge_score import rouge_scorer

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    logger.warning("rouge-score not available, will use basic ROUGE implementation")

try:
    import sacrebleu

    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    logger.warning("sacrebleu not available, will use basic BLEU implementation")

try:
    from bert_score import score as bert_score

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    logger.warning("bert-score not available, BERTScore metric will not be available")


class BLEUScore(BaseMetric):
    """BLEU score for machine translation and text generation."""

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = True,
        lowercase: bool = False,
        tokenizer: str = "13a",
    ):
        """Initialize BLEU metric.

        Args:
            n_gram: Maximum n-gram order
            smooth: Whether to use smoothing
            lowercase: Whether to lowercase before scoring
            tokenizer: Tokenizer to use (for sacrebleu)
        """
        super().__init__(f"bleu_{n_gram}", higher_is_better=True)
        self.n_gram = n_gram
        self.smooth = smooth
        self.lowercase = lowercase
        self.tokenizer = tokenizer

    def compute(
        self, predictions: List[str], references: List[Union[str, List[str]]], **kwargs
    ) -> float:
        """Compute BLEU score.

        Args:
            predictions: Generated texts
            references: Reference texts (can be list of lists for multiple refs)
            **kwargs: Additional parameters

        Returns:
            BLEU score (0-100 scale)
        """
        if not predictions or not references:
            return 0.0

        if HAS_SACREBLEU:
            try:
                # Handle multiple references per sample
                if isinstance(references[0], list):
                    # Transpose to get list of refs for each position
                    refs_transposed = list(zip(*references))
                    bleu = sacrebleu.corpus_bleu(
                        predictions,
                        refs_transposed,
                        lowercase=self.lowercase,
                        tokenize=self.tokenizer,
                    )
                else:
                    bleu = sacrebleu.corpus_bleu(
                        predictions,
                        [references],
                        lowercase=self.lowercase,
                        tokenize=self.tokenizer,
                    )
                return bleu.score
            except Exception as e:
                logger.warning(f"sacrebleu failed: {e}, using basic implementation")

        # Basic implementation
        return self._compute_basic_bleu(predictions, references)

    def _compute_basic_bleu(
        self, predictions: List[str], references: List[Union[str, List[str]]]
    ) -> float:
        """Basic BLEU implementation.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            BLEU score
        """
        scores = []

        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                # Multiple references - take best score
                ref_scores = [self._bleu_sentence(pred, r) for r in ref]
                scores.append(max(ref_scores))
            else:
                scores.append(self._bleu_sentence(pred, ref))

        return np.mean(scores) * 100  # Convert to 0-100 scale

    def _bleu_sentence(self, prediction: str, reference: str) -> float:
        """Compute BLEU for a single sentence pair.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            BLEU score (0-1 scale)
        """
        if self.lowercase:
            prediction = prediction.lower()
            reference = reference.lower()

        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Calculate n-gram precisions
        precisions = []

        for n in range(1, min(self.n_gram + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                continue

            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += 1
                    ref_ngrams.remove(ngram)  # Each ref n-gram can only match once

            precision = matches / len(pred_ngrams)

            # Smoothing
            if self.smooth and precision == 0:
                precision = 1 / (2 * len(pred_ngrams))

            precisions.append(precision)

        if not precisions:
            return 0.0

        # Calculate brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))

        # Calculate BLEU
        bleu = bp * np.exp(
            np.mean([np.log(p) if p > 0 else -np.inf for p in precisions])
        )

        return bleu if not np.isinf(bleu) else 0.0

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            List of n-grams
        """
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class ROUGEScore(BaseMetric):
    """ROUGE scores for text summarization."""

    def __init__(
        self,
        rouge_types: List[str] = None,
        use_stemmer: bool = True,
        split_summaries: bool = False,
    ):
        """Initialize ROUGE metric.

        Args:
            rouge_types: ROUGE types to compute (default: rouge1, rouge2, rougeL)
            use_stemmer: Whether to use stemming
            split_summaries: Whether to split summaries into sentences
        """
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]

        super().__init__("rouge", higher_is_better=True)
        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer
        self.split_summaries = split_summaries

        if HAS_ROUGE:
            self.scorer = rouge_scorer.RougeScorer(
                rouge_types, use_stemmer=use_stemmer, split_summaries=split_summaries
            )
        else:
            self.scorer = None

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Dict[str, float]:
        """Compute ROUGE scores.

        Args:
            predictions: Generated summaries
            references: Reference summaries
            **kwargs: Additional parameters

        Returns:
            Dictionary of ROUGE scores
        """
        if not predictions or not references:
            return {f"{t}_f1": 0.0 for t in self.rouge_types}

        if HAS_ROUGE and self.scorer:
            try:
                scores = {f"{t}_f1": [] for t in self.rouge_types}
                scores.update({f"{t}_precision": [] for t in self.rouge_types})
                scores.update({f"{t}_recall": [] for t in self.rouge_types})

                for pred, ref in zip(predictions, references):
                    result = self.scorer.score(ref, pred)
                    for rouge_type in self.rouge_types:
                        scores[f"{rouge_type}_f1"].append(result[rouge_type].fmeasure)
                        scores[f"{rouge_type}_precision"].append(
                            result[rouge_type].precision
                        )
                        scores[f"{rouge_type}_recall"].append(result[rouge_type].recall)

                # Average scores
                return {k: np.mean(v) * 100 for k, v in scores.items()}

            except Exception as e:
                logger.warning(f"rouge-score failed: {e}, using basic implementation")

        # Basic implementation
        return self._compute_basic_rouge(predictions, references)

    def _compute_basic_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Basic ROUGE implementation.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary of ROUGE scores
        """
        all_scores = {f"{t}_f1": [] for t in self.rouge_types}

        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred.lower())
            ref_tokens = self._tokenize(ref.lower())

            for rouge_type in self.rouge_types:
                if rouge_type == "rouge1":
                    score = self._rouge_n(pred_tokens, ref_tokens, 1)
                elif rouge_type == "rouge2":
                    score = self._rouge_n(pred_tokens, ref_tokens, 2)
                elif rouge_type == "rougeL":
                    score = self._rouge_l(pred_tokens, ref_tokens)
                else:
                    score = 0.0

                all_scores[f"{rouge_type}_f1"].append(score)

        return {k: np.mean(v) * 100 for k, v in all_scores.items()}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Remove punctuation and split
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def _rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Compute ROUGE-N score.

        Args:
            pred_tokens: Prediction tokens
            ref_tokens: Reference tokens
            n: N-gram size

        Returns:
            F1 score
        """
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0

        pred_ngrams = Counter(self._get_ngrams(pred_tokens, n))
        ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))

        overlap = sum((pred_ngrams & ref_ngrams).values())

        if overlap == 0:
            return 0.0

        precision = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute ROUGE-L score using LCS.

        Args:
            pred_tokens: Prediction tokens
            ref_tokens: Reference tokens

        Returns:
            F1 score
        """
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)

        if lcs_length == 0:
            return 0.0

        precision = lcs_length / len(pred_tokens) if pred_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _lcs_length(self, s1: List[str], s2: List[str]) -> int:
        """Compute length of longest common subsequence.

        Args:
            s1: First sequence
            s2: Second sequence

        Returns:
            LCS length
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            List of n-grams
        """
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class BERTScore(BaseMetric):
    """BERTScore for semantic similarity."""

    def __init__(
        self,
        model_type: str = "bert-base-uncased",
        num_layers: Optional[int] = None,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """Initialize BERTScore metric.

        Args:
            model_type: BERT model to use
            num_layers: Number of layers to use
            batch_size: Batch size for scoring
            device: Device to use (cuda/cpu)
        """
        super().__init__("bertscore", higher_is_better=True)
        self.model_type = model_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device

        if not HAS_BERTSCORE:
            logger.warning("BERTScore not available - install bert-score package")

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Dict[str, float]:
        """Compute BERTScore.

        Args:
            predictions: Generated texts
            references: Reference texts
            **kwargs: Additional parameters

        Returns:
            Dictionary with precision, recall, and F1
        """
        if not predictions or not references:
            return {
                "bertscore_f1": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
            }

        if not HAS_BERTSCORE:
            logger.error("BERTScore not available")
            return {
                "bertscore_f1": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
            }

        try:
            P, R, F1 = bert_score(
                predictions,
                references,
                model_type=self.model_type,
                num_layers=self.num_layers,
                batch_size=self.batch_size,
                device=self.device,
                verbose=False,
            )

            return {
                "bertscore_f1": F1.mean().item(),
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
            }

        except Exception as e:
            logger.error(f"BERTScore computation failed: {e}")
            return {
                "bertscore_f1": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
            }


class ExactMatchScore(AveragedMetric):
    """Exact match metric for QA and generation tasks."""

    def __init__(
        self,
        normalize: bool = True,
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        ignore_articles: bool = False,
    ):
        """Initialize exact match metric.

        Args:
            normalize: Whether to normalize texts
            ignore_case: Whether to ignore case
            ignore_punctuation: Whether to ignore punctuation
            ignore_articles: Whether to ignore articles (a, an, the)
        """
        super().__init__("exact_match", higher_is_better=True)
        self.normalize = normalize
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_articles = ignore_articles

        self.articles = {"a", "an", "the"}

    def compute_single(self, prediction: str, reference: str, **kwargs) -> float:
        """Compute exact match for a single pair.

        Args:
            prediction: Generated text
            reference: Reference text
            **kwargs: Additional parameters

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if self.normalize:
            prediction = self._normalize(prediction)
            reference = self._normalize(reference)

        return 1.0 if prediction == reference else 0.0

    def _normalize(self, text: str) -> str:
        """Normalize text based on settings.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if self.ignore_case:
            text = text.lower()

        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.ignore_articles:
            tokens = text.split()
            tokens = [t for t in tokens if t.lower() not in self.articles]
            text = " ".join(tokens)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text.strip()


class TokenF1Score(BaseMetric):
    """Token-level F1 score for generation tasks."""

    def __init__(self, ignore_case: bool = True):
        """Initialize token F1 metric.

        Args:
            ignore_case: Whether to ignore case
        """
        super().__init__("token_f1", higher_is_better=True)
        self.ignore_case = ignore_case

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """Compute token F1 score.

        Args:
            predictions: Generated texts
            references: Reference texts
            **kwargs: Additional parameters

        Returns:
            Average token F1 score
        """
        if not predictions or not references:
            return 0.0

        scores = []

        for pred, ref in zip(predictions, references):
            if self.ignore_case:
                pred = pred.lower()
                ref = ref.lower()

            pred_tokens = set(pred.split())
            ref_tokens = set(ref.split())

            if not pred_tokens and not ref_tokens:
                scores.append(1.0)
                continue

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            common = pred_tokens & ref_tokens

            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            scores.append(f1)

        return np.mean(scores)


class Perplexity(BaseMetric):
    """Perplexity metric for language modeling."""

    def __init__(self, base: float = np.e):
        """Initialize perplexity metric.

        Args:
            base: Base for perplexity calculation (e or 2)
        """
        super().__init__("perplexity", higher_is_better=False)
        self.base = base

    def compute(
        self, predictions: List[float], references: Optional[List[Any]] = None, **kwargs
    ) -> float:
        """Compute perplexity from log probabilities.

        Args:
            predictions: Log probabilities
            references: Not used (for compatibility)
            **kwargs: Additional parameters

        Returns:
            Perplexity score
        """
        if not predictions:
            return float("inf")

        # Predictions should be log probabilities
        avg_log_prob = np.mean(predictions)

        if self.base == np.e:
            perplexity = np.exp(-avg_log_prob)
        else:
            perplexity = self.base ** (-avg_log_prob)

        return perplexity


class LengthMetrics(BaseMetric):
    """Length-based metrics for generation."""

    def __init__(self):
        """Initialize length metrics."""
        super().__init__("length_metrics", higher_is_better=False)

    def compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> Dict[str, float]:
        """Compute length-based metrics.

        Args:
            predictions: Generated texts
            references: Reference texts
            **kwargs: Additional parameters

        Returns:
            Dictionary of length metrics
        """
        if not predictions:
            return {
                "avg_pred_length": 0,
                "avg_ref_length": 0,
                "length_ratio": 1.0,
                "abs_length_diff": 0,
            }

        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references] if references else []

        metrics = {
            "avg_pred_length": np.mean(pred_lengths),
            "std_pred_length": np.std(pred_lengths),
            "min_pred_length": np.min(pred_lengths),
            "max_pred_length": np.max(pred_lengths),
        }

        if ref_lengths:
            metrics.update(
                {
                    "avg_ref_length": np.mean(ref_lengths),
                    "length_ratio": np.mean(pred_lengths) / np.mean(ref_lengths),
                    "abs_length_diff": np.mean(
                        [abs(p - r) for p, r in zip(pred_lengths, ref_lengths)]
                    ),
                }
            )

        return metrics


class EditDistance(AveragedMetric):
    """Edit distance metrics for generation."""

    def __init__(self, normalize: bool = True):
        """Initialize edit distance metric.

        Args:
            normalize: Whether to normalize by reference length
        """
        super().__init__("edit_distance", higher_is_better=False)
        self.normalize = normalize

    def compute_single(self, prediction: str, reference: str, **kwargs) -> float:
        """Compute edit distance for a single pair.

        Args:
            prediction: Generated text
            reference: Reference text
            **kwargs: Additional parameters

        Returns:
            Edit distance (normalized if requested)
        """
        distance = self._levenshtein_distance(prediction, reference)

        if self.normalize and len(reference) > 0:
            distance = distance / len(reference)

        return distance

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]


class GenerationMetricsReport:
    """Comprehensive report for generation metrics."""

    def __init__(
        self,
        include_bleu: bool = True,
        include_rouge: bool = True,
        include_bertscore: bool = False,
        include_exact_match: bool = True,
        include_token_f1: bool = True,
        include_length: bool = True,
    ):
        """Initialize generation metrics report.

        Args:
            include_bleu: Include BLEU score
            include_rouge: Include ROUGE scores
            include_bertscore: Include BERTScore
            include_exact_match: Include exact match
            include_token_f1: Include token F1
            include_length: Include length metrics
        """
        self.metrics = {}

        if include_bleu:
            self.metrics["bleu"] = BLEUScore()

        if include_rouge:
            self.metrics["rouge"] = ROUGEScore()

        if include_bertscore and HAS_BERTSCORE:
            self.metrics["bertscore"] = BERTScore()

        if include_exact_match:
            self.metrics["exact_match"] = ExactMatchScore()

        if include_token_f1:
            self.metrics["token_f1"] = TokenF1Score()

        if include_length:
            self.metrics["length"] = LengthMetrics()

    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Compute all generation metrics.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Dictionary of all metrics
        """
        results = {}

        for name, metric in self.metrics.items():
            try:
                score = metric.compute(predictions, references)

                if isinstance(score, dict):
                    results.update(score)
                else:
                    results[name] = score

            except Exception as e:
                logger.error(f"Failed to compute {name}: {e}")
                results[name] = 0.0

        return results

    def format_report(self, predictions: List[str], references: List[str]) -> str:
        """Generate formatted generation metrics report.

        Args:
            predictions: Generated texts
            references: Reference texts

        Returns:
            Formatted report string
        """
        results = self.compute(predictions, references)

        lines = ["Generation Metrics Report", "=" * 50]

        # Group metrics
        if "bleu_4" in results:
            lines.append("\nBLEU Score:")
            lines.append(f"  BLEU-4:             {results.get('bleu_4', 0):.2f}")

        # ROUGE scores
        rouge_scores = [k for k in results if k.startswith("rouge")]
        if rouge_scores:
            lines.append("\nROUGE Scores:")
            for key in sorted(rouge_scores):
                if "f1" in key:
                    lines.append(f"  {key:20s}: {results[key]:.2f}")

        # BERTScore
        if "bertscore_f1" in results:
            lines.append("\nBERTScore:")
            lines.append(f"  F1:                 {results['bertscore_f1']:.4f}")
            lines.append(f"  Precision:          {results['bertscore_precision']:.4f}")
            lines.append(f"  Recall:             {results['bertscore_recall']:.4f}")

        # Other metrics
        if "exact_match" in results:
            lines.append(f"\nExact Match:          {results['exact_match']:.4f}")

        if "token_f1" in results:
            lines.append(f"Token F1:             {results['token_f1']:.4f}")

        # Length metrics
        if "avg_pred_length" in results:
            lines.append("\nLength Statistics:")
            lines.append(f"  Avg Prediction:     {results['avg_pred_length']:.1f}")
            if "avg_ref_length" in results:
                lines.append(f"  Avg Reference:      {results['avg_ref_length']:.1f}")
                lines.append(f"  Length Ratio:       {results['length_ratio']:.2f}")

        lines.append("=" * 50)

        return "\n".join(lines)


# Convenience instances
bleu_score = BLEUScore()
rouge_score = ROUGEScore()
exact_match = ExactMatchScore()
token_f1 = TokenF1Score()
edit_distance = EditDistance()
length_metrics = LengthMetrics()
