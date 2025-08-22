"""FLAME-specific evaluation metrics for BenchForge integration."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

logger = logging.getLogger(__name__)


class FLAMEEvaluator:
    """Comprehensive evaluator for FLAME tasks with BenchForge integration."""

    def __init__(self):
        """Initialize FLAME evaluator."""
        self.qa_evaluator = QAEvaluator()
        self.ner_evaluator = NREvaluator()
        self.classification_evaluator = ClassificationEvaluator()

    def evaluate_task(
        self, 
        task_name: str, 
        predictions: List[Any], 
        ground_truth: List[Any],
        samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Evaluate predictions for a specific FLAME task.
        
        Args:
            task_name: Name of the FLAME task
            predictions: Model predictions
            ground_truth: Ground truth labels
            samples: Optional sample data for additional context
            
        Returns:
            Dictionary of evaluation metrics
        """
        task_name = task_name.lower()
        
        # Route to appropriate evaluator
        if task_name in ["convfinqa", "finqa", "edtsum"]:
            return self.qa_evaluator.evaluate(predictions, ground_truth, samples)
        elif task_name in ["finer", "finentity"]:
            return self.ner_evaluator.evaluate(predictions, ground_truth, samples)
        elif task_name in ["fomc", "fpb", "headlines", "numclaim"]:
            return self.classification_evaluator.evaluate(predictions, ground_truth, samples)
        else:
            # Default to classification evaluation
            logger.warning(f"Unknown task {task_name}, using classification evaluation")
            return self.classification_evaluator.evaluate(predictions, ground_truth, samples)


class QAEvaluator:
    """Evaluator for Question Answering tasks (ConvFinQA, FinQA, EDTSum)."""

    def evaluate(
        self, 
        predictions: List[Any], 
        ground_truth: List[Any],
        samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Evaluate QA predictions with multiple metrics.
        
        Args:
            predictions: Model predictions (strings or extracted values)
            ground_truth: Ground truth answers
            samples: Optional sample data
            
        Returns:
            Dictionary with QA evaluation metrics
        """
        # Filter out None predictions for metric calculation
        valid_pairs = [
            (pred, gt) for pred, gt in zip(predictions, ground_truth) 
            if pred is not None and gt is not None
        ]
        
        if not valid_pairs:
            logger.warning("No valid prediction-ground truth pairs found")
            return self._empty_qa_metrics()
        
        valid_predictions, valid_ground_truth = zip(*valid_pairs)
        
        # Calculate metrics
        exact_match = self._exact_match_score(valid_predictions, valid_ground_truth)
        f1_score = self._qa_f1_score(valid_predictions, valid_ground_truth)
        numeric_accuracy = self._numeric_accuracy(valid_predictions, valid_ground_truth)
        
        # Coverage metrics
        total_samples = len(predictions)
        valid_samples = len(valid_pairs)
        coverage = valid_samples / total_samples if total_samples > 0 else 0.0
        
        metrics = {
            "exact_match": exact_match,
            "f1_score": f1_score,
            "numeric_accuracy": numeric_accuracy,
            "coverage": coverage,
            "valid_predictions": valid_samples,
            "total_samples": total_samples,
        }
        
        logger.info(f"QA Evaluation: EM={exact_match:.3f}, F1={f1_score:.3f}, "
                   f"NumAcc={numeric_accuracy:.3f}, Coverage={coverage:.3f}")
        
        return metrics

    def _exact_match_score(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate exact match score with normalization."""
        exact_matches = 0
        
        for pred, gt in zip(predictions, ground_truth):
            if self._normalize_answer(str(pred)) == self._normalize_answer(str(gt)):
                exact_matches += 1
        
        return exact_matches / len(predictions) if predictions else 0.0

    def _qa_f1_score(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate token-level F1 score for QA."""
        f1_scores = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_tokens = self._tokenize_answer(str(pred))
            gt_tokens = self._tokenize_answer(str(gt))
            
            if not gt_tokens:
                f1_scores.append(1.0 if not pred_tokens else 0.0)
                continue
            
            if not pred_tokens:
                f1_scores.append(0.0)
                continue
            
            # Calculate token overlap
            common_tokens = set(pred_tokens) & set(gt_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(gt_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0

    def _numeric_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate accuracy for numeric answers."""
        correct = 0
        numeric_pairs = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_num = self._extract_number(str(pred))
            gt_num = self._extract_number(str(gt))
            
            if pred_num is not None and gt_num is not None:
                numeric_pairs += 1
                if abs(pred_num - gt_num) < 1e-6:  # Handle floating point comparison
                    correct += 1
        
        return correct / numeric_pairs if numeric_pairs > 0 else 0.0

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove articles and extra whitespace
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer.lower())
        answer = re.sub(r'\s+', ' ', answer).strip()
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        return answer

    def _tokenize_answer(self, answer: str) -> List[str]:
        """Tokenize answer for F1 calculation."""
        normalized = self._normalize_answer(answer)
        return normalized.split()

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text."""
        # Remove currency symbols and commas
        text = re.sub(r'[$,]', '', text)
        
        # Look for numbers (including percentages and decimals)
        number_pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
        match = re.search(number_pattern, text)
        
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        
        return None

    def _empty_qa_metrics(self) -> Dict[str, float]:
        """Return empty metrics for QA evaluation."""
        return {
            "exact_match": 0.0,
            "f1_score": 0.0,
            "numeric_accuracy": 0.0,
            "coverage": 0.0,
            "valid_predictions": 0,
            "total_samples": 0,
        }


class NREvaluator:
    """Evaluator for Named Entity Recognition tasks (FiNER, FinEntity)."""

    def evaluate(
        self, 
        predictions: List[Any], 
        ground_truth: List[Any],
        samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Evaluate NER predictions with entity-level metrics.
        
        Args:
            predictions: Model predictions (lists of tags or entity types)
            ground_truth: Ground truth labels
            samples: Optional sample data
            
        Returns:
            Dictionary with NER evaluation metrics
        """
        # Handle different NER task types
        if self._is_sequence_labeling(predictions, ground_truth):
            return self._evaluate_sequence_labeling(predictions, ground_truth)
        else:
            return self._evaluate_entity_classification(predictions, ground_truth)

    def _is_sequence_labeling(self, predictions: List[Any], ground_truth: List[Any]) -> bool:
        """Determine if this is sequence labeling (BIO tags) or entity classification."""
        # Check if first few samples are lists (sequence labeling) or strings (classification)
        for pred, gt in zip(predictions[:5], ground_truth[:5]):
            if pred is not None and gt is not None:
                if isinstance(pred, list) and isinstance(gt, list):
                    return True
                elif isinstance(pred, str) and isinstance(gt, str):
                    # Check if it looks like BIO tags
                    if any(tag.startswith(('B-', 'I-')) or tag == 'O' for tag in [pred, gt]):
                        return False  # Single tag, not sequence
                    return False
        return False

    def _evaluate_sequence_labeling(self, predictions: List[List[str]], ground_truth: List[List[str]]) -> Dict[str, float]:
        """Evaluate sequence labeling with entity-level metrics."""
        all_pred_entities = []
        all_gt_entities = []
        
        valid_pairs = 0
        total_pairs = len(predictions)
        
        for pred_tags, gt_tags in zip(predictions, ground_truth):
            if pred_tags is None or gt_tags is None:
                continue
                
            # Skip if length mismatch
            if len(pred_tags) != len(gt_tags):
                logger.debug(f"Length mismatch: pred={len(pred_tags)}, gt={len(gt_tags)}")
                continue
            
            valid_pairs += 1
            
            # Extract entities from BIO tags
            pred_entities = self._extract_entities_from_bio(pred_tags)
            gt_entities = self._extract_entities_from_bio(gt_tags)
            
            all_pred_entities.extend(pred_entities)
            all_gt_entities.extend(gt_entities)
        
        # Calculate entity-level metrics
        precision, recall, f1 = self._calculate_entity_metrics(all_pred_entities, all_gt_entities)
        
        # Token-level accuracy for valid sequences
        token_accuracy = self._calculate_token_accuracy(predictions, ground_truth)
        
        metrics = {
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "token_accuracy": token_accuracy,
            "coverage": valid_pairs / total_pairs if total_pairs > 0 else 0.0,
            "valid_sequences": valid_pairs,
            "total_sequences": total_pairs,
        }
        
        logger.info(f"NER Sequence Evaluation: P={precision:.3f}, R={recall:.3f}, "
                   f"F1={f1:.3f}, TokenAcc={token_accuracy:.3f}")
        
        return metrics

    def _evaluate_entity_classification(self, predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate entity type classification."""
        # Filter valid predictions
        valid_pairs = [
            (pred, gt) for pred, gt in zip(predictions, ground_truth)
            if pred is not None and gt is not None
        ]
        
        if not valid_pairs:
            return self._empty_ner_metrics()
        
        valid_predictions, valid_ground_truth = zip(*valid_pairs)
        
        # Convert to string and normalize
        pred_strings = [str(p).upper() for p in valid_predictions]
        gt_strings = [str(g).upper() for g in valid_ground_truth]
        
        # Calculate metrics
        accuracy = accuracy_score(gt_strings, pred_strings)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_strings, pred_strings, average='weighted', zero_division=0
        )
        
        # Coverage
        coverage = len(valid_pairs) / len(predictions) if len(predictions) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "coverage": coverage,
            "valid_predictions": len(valid_pairs),
            "total_samples": len(predictions),
        }
        
        logger.info(f"NER Classification Evaluation: Acc={accuracy:.3f}, P={precision:.3f}, "
                   f"R={recall:.3f}, F1={f1:.3f}")
        
        return metrics

    def _extract_entities_from_bio(self, bio_tags: List[str]) -> List[Tuple[int, int, str]]:
        """Extract entities from BIO tag sequence.
        
        Returns:
            List of (start, end, entity_type) tuples
        """
        entities = []
        current_entity = None
        
        for i, tag in enumerate(bio_tags):
            if tag.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = (i, i + 1, entity_type)
            elif tag.startswith('I-') and current_entity:
                # Continue current entity
                entity_type = tag[2:]
                if current_entity[2] == entity_type:
                    current_entity = (current_entity[0], i + 1, entity_type)
                else:
                    # Type mismatch, end current and start new
                    entities.append(current_entity)
                    current_entity = (i, i + 1, entity_type)
            else:
                # O tag or invalid I- tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def _calculate_entity_metrics(
        self, 
        pred_entities: List[Tuple[int, int, str]], 
        gt_entities: List[Tuple[int, int, str]]
    ) -> Tuple[float, float, float]:
        """Calculate entity-level precision, recall, and F1."""
        pred_set = set(pred_entities)
        gt_set = set(gt_entities)
        
        if not gt_set:
            return (1.0 if not pred_set else 0.0, 1.0, 1.0 if not pred_set else 0.0)
        
        if not pred_set:
            return (0.0, 0.0, 0.0)
        
        correct = len(pred_set & gt_set)
        precision = correct / len(pred_set)
        recall = correct / len(gt_set)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1

    def _calculate_token_accuracy(self, predictions: List[List[str]], ground_truth: List[List[str]]) -> float:
        """Calculate token-level accuracy for valid sequences."""
        correct_tokens = 0
        total_tokens = 0
        
        for pred_tags, gt_tags in zip(predictions, ground_truth):
            if pred_tags is None or gt_tags is None or len(pred_tags) != len(gt_tags):
                continue
            
            for pred_tag, gt_tag in zip(pred_tags, gt_tags):
                total_tokens += 1
                if pred_tag == gt_tag:
                    correct_tokens += 1
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0

    def _empty_ner_metrics(self) -> Dict[str, float]:
        """Return empty metrics for NER evaluation."""
        return {
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0,
            "token_accuracy": 0.0,
            "coverage": 0.0,
            "valid_sequences": 0,
            "total_sequences": 0,
        }


class ClassificationEvaluator:
    """Evaluator for classification tasks (FOMC, FPB, etc.)."""

    def evaluate(
        self, 
        predictions: List[Any], 
        ground_truth: List[Any],
        samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Evaluate classification predictions.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            samples: Optional sample data
            
        Returns:
            Dictionary with classification metrics
        """
        # Filter valid predictions
        valid_pairs = [
            (pred, gt) for pred, gt in zip(predictions, ground_truth)
            if pred is not None and gt is not None
        ]
        
        if not valid_pairs:
            return self._empty_classification_metrics()
        
        valid_predictions, valid_ground_truth = zip(*valid_pairs)
        
        # Calculate metrics
        accuracy = accuracy_score(valid_ground_truth, valid_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_ground_truth, valid_predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            valid_ground_truth, valid_predictions, average='macro', zero_division=0
        )
        
        # Coverage
        coverage = len(valid_pairs) / len(predictions) if len(predictions) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "coverage": coverage,
            "valid_predictions": len(valid_pairs),
            "total_samples": len(predictions),
        }
        
        logger.info(f"Classification Evaluation: Acc={accuracy:.3f}, F1_weighted={f1:.3f}, "
                   f"F1_macro={f1_macro:.3f}")
        
        return metrics

    def _empty_classification_metrics(self) -> Dict[str, float]:
        """Return empty metrics for classification evaluation."""
        return {
            "accuracy": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "coverage": 0.0,
            "valid_predictions": 0,
            "total_samples": 0,
        }