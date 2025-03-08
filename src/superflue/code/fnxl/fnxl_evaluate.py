import json
import pandas as pd
import logging
from datetime import date
from litellm import batch_completion
from superflue.utils.batch_utils import chunk_list, process_batch_with_retry
from superflue.utils.logging_utils import setup_logger
from superflue.code.extraction_prompts import fnxl_extraction_prompt
from superflue.config import LOG_DIR, LOG_LEVEL

logger = setup_logger(
    name="fnxl_evaluation",
    log_file=LOG_DIR / "fnxl_evaluation.log",
    level=LOG_LEVEL,
)


def normalize_taglist_json(json_input):
    """
    Convert the JSON string (or dict) into:
      { tag (lowercased): set_of_floats }
    ignoring any non-numeric items.
    """
    if isinstance(json_input, str):
        json_str = json_input.strip().strip("```").strip()
        json_str = json_str.replace("'", '"')
        try:
            data = json.loads(json_str)
        except:
            return {}
    elif isinstance(json_input, dict):
        data = json_input
    else:
        return {}

    normalized = {}
    for tag, val_list in data.items():
        floats_set = set()
        if isinstance(val_list, list):
            for v in val_list:
                try:
                    if isinstance(v, (int, float)):
                        floats_set.add(float(v))
                    elif isinstance(v, str):
                        val_str = v.replace(",", "").strip()
                        floats_set.add(float(val_str))
                except ValueError:
                    pass
        normalized[tag.lower().strip()] = floats_set
    return normalized


def compare_taglist_dicts(actual, predicted):
    """
    Partial-credit set comparison for "tag -> set_of_floats".
    Returns (tp, fp, fn, total_actual, total_predicted).
    """
    actual_dict = normalize_taglist_json(actual)
    pred_dict   = normalize_taglist_json(predicted)

    tp = 0
    fp = 0
    fn = 0

    all_tags = set(actual_dict.keys()).union(set(pred_dict.keys()))
    for tag in all_tags:
        actual_vals = actual_dict.get(tag, set())
        pred_vals   = pred_dict.get(tag, set())

        overlap = actual_vals.intersection(pred_vals)
        tp += len(overlap)
        fp += len(pred_vals - actual_vals)
        fn += len(actual_vals - pred_vals)

    total_actual = sum(len(s) for s in actual_dict.values())
    total_predicted = sum(len(s) for s in pred_dict.values())

    return tp, fp, fn, total_actual, total_predicted


def fnxl_evaluate(file_name, args):
    """
    Evaluate FNXL dataset:
      1) Load df from CSV (which has columns: 'actual_labels', 'llm_responses').
      2) For each row, run a second-pass "extraction_prompt" to parse raw 'llm_responses'
         into a uniform "tag -> [list_of_floats]" structure.
      3) Compare partial-credit for each row. Sum up micro-average metrics.
      4) Return (df_with_extractions, metrics_df).
    """
    logger.info(f"Loading file: {file_name}")
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")
    
    if "extracted_labels" not in df.columns:
        df["extracted_labels"] = None

    row_metrics = []

    batch_size = args.batch_size
    index_batches = chunk_list(list(df.index), batch_size)

    for batch_idx, batch_indices in enumerate(index_batches):
        raw_responses_batch = df.loc[batch_indices, "llm_responses"].tolist()
        actual_labels_batch = df.loc[batch_indices, "actual_labels"].tolist()

        messages_batch = []
        for raw_resp in raw_responses_batch:
            user_msg = fnxl_extraction_prompt(raw_resp)
            messages_batch.append([{"role": "user", "content": user_msg}])

        try:
            batch_responses = process_batch_with_retry(args, messages_batch, batch_idx, len(index_batches))
        except Exception as e:
            logger.error(f"Batch {batch_idx+1} second-pass extraction failed: {e}")
            for _ in messages_batch:
                row_metrics.append({
                    "tp": 0, "fp": 0, "fn": 0,
                    "total_actual": 0, "total_predicted": 0
                })
            continue
        
        for i, response in enumerate(batch_responses):
            try:
                cleaned_json_str = response.choices[0].message.content.strip()  # type: ignore
                tp, fp, fn, total_act, total_pred = compare_taglist_dicts(
                    actual_labels_batch[i], 
                    cleaned_json_str
                )
                row_metrics.append({
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "total_actual": total_act,
                    "total_predicted": total_pred
                })
                df.at[batch_indices[i], "extracted_labels"] = cleaned_json_str
            except Exception as e:
                logger.error(f"Error processing row {batch_indices[i]}: {e}")
                row_metrics.append({
                    "tp": 0, "fp": 0, "fn": 0,
                    "total_actual": 0, "total_predicted": 0
                })
                df.at[batch_indices[i], "extracted_labels"] = None

    total_tp = sum(m["tp"] for m in row_metrics)
    total_fp = sum(m["fp"] for m in row_metrics)
    total_fn = sum(m["fn"] for m in row_metrics)
    total_actual = sum(m["total_actual"] for m in row_metrics)
    total_predicted = sum(m["total_predicted"] for m in row_metrics)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    if (total_actual + total_predicted - total_tp) > 0:
        accuracy = total_tp / (total_actual + total_predicted - total_tp)
    else:
        accuracy = 0.0

    logger.info(f"Final micro-average metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Accuracy (Jaccard): {accuracy:.4f}")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy (Jaccard)", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    })

    return df, metrics_df
