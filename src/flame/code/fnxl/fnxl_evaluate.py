import json

import pandas as pd
from tqdm import tqdm

from flame.code.prompts.registry import PromptFormat, get_prompt
from flame.config import LOG_DIR, LOG_LEVEL
from flame.utils.batch_utils import chunk_list, process_batch_with_retry
from flame.utils.logging_utils import setup_logger

logger = setup_logger(
    name="fnxl_evaluation",
    log_file=LOG_DIR / "fnxl_evaluation.log",
    level=LOG_LEVEL,
)


def clean_json_response(response):
    """
    Clean JSON response by removing markdown formatting and extracting only the JSON portion.
    Handles cases like:
    - "Here is the output in JSON format:\n\n```json\n{...}\n```\n\nExplanation:..."
    - "json\n{...}"
    - "```json\n{...}\n```"
    """
    if not isinstance(response, str):
        return response

    response = response.strip()

    # Look for JSON content within markdown code blocks
    if "```json" in response:
        start_marker = response.find("```json")
        json_start = start_marker + 7  # Length of '```json\n'

        # Find the end of the JSON block
        end_marker = response.find("```", json_start)
        if end_marker != -1:
            json_content = response[json_start:end_marker].strip()
        else:
            # No closing ```, take everything after ```json
            json_content = response[json_start:].strip()

    # Look for JSON content after "Here is the output in JSON format:"
    elif "Here is the output in JSON format:" in response:
        json_start_marker = response.find("Here is the output in JSON format:")
        after_intro = response[
            json_start_marker + len("Here is the output in JSON format:") :
        ].strip()

        # Look for actual JSON content (starts with { or [)
        json_start = 0
        for i, char in enumerate(after_intro):
            if char in "{[":
                json_start = i
                break

        # Extract JSON until we hit explanation text or end
        json_content = after_intro[json_start:]

        # Try to find the end of JSON by looking for closing brace followed by explanation
        if json_content.startswith("{"):
            brace_count = 0
            json_end = 0
            for i, char in enumerate(json_content):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                json_content = json_content[:json_end]

    # Fallback: try to extract JSON from any response
    else:
        # Look for JSON starting with { or [
        json_start = 0
        for i, char in enumerate(response):
            if char in "{[":
                json_start = i
                break

        json_content = response[json_start:]

        # Try to find the end of JSON
        if json_content.startswith("{"):
            brace_count = 0
            json_end = 0
            for i, char in enumerate(json_content):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                json_content = json_content[:json_end]

    # Clean up the extracted JSON
    json_content = json_content.strip()
    json_content = json_content.replace(
        "'", '"'
    )  # Replace single quotes with double quotes

    logger.debug(f"Cleaned JSON response: {json_content[:100]}...")
    return json_content


def normalize_taglist_json(json_input):
    """
    Convert the JSON string (or dict) into:
      { tag (lowercased): set_of_floats }
    ignoring any non-numeric items.
    """
    if isinstance(json_input, str):
        # Apply JSON cleanup first
        json_str = clean_json_response(json_input)
        json_str = json_str.replace("'", '"')
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON string after cleanup: {json_str[:200]}... Error: {e}"
            )
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
    pred_dict = normalize_taglist_json(predicted)

    tp = 0
    fp = 0
    fn = 0

    all_tags = set(actual_dict.keys()).union(set(pred_dict.keys()))
    for tag in all_tags:
        actual_vals = actual_dict.get(tag, set())
        pred_vals = pred_dict.get(tag, set())

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
    # support legacy args.dataset for tests, prefer args.task
    task = getattr(args, "task", None) or getattr(args, "dataset", None) or "fnxl"
    logger.info(f"Starting evaluation for {task} using model {args.model}.")

    logger.info(f"Loading file: {file_name}")
    df = pd.read_csv(file_name)
    logger.info(f"Loaded {len(df)} rows from {file_name}.")

    row_metrics = []

    all_responses = df["llm_responses"].tolist()
    batches = chunk_list(all_responses, args.batch_size)
    actual_labels = df["actual_labels"].tolist()
    actual_labels_batches = chunk_list(actual_labels, args.batch_size)
    total_batches = len(batches)

    logger.info(f"Processing {len(df)} rows in {total_batches} batches.")
    pbar = tqdm(batches, desc="Processing batches")
    for batch_idx, batch in enumerate(pbar):
        extraction_prompt_func = get_prompt("fnxl", PromptFormat.EXTRACTION)
        messages_batch = [
            [{"role": "user", "content": extraction_prompt_func(response)}]
            for response in batch
        ]

        try:
            batch_responses = process_batch_with_retry(
                args, messages_batch, batch_idx, total_batches
            )

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} second-pass extraction failed: {e}")
            for _ in messages_batch:
                row_metrics.append(
                    {"tp": 0, "fp": 0, "fn": 0, "total_actual": 0, "total_predicted": 0}
                )
            continue

        actual_labels_batch = actual_labels_batches[batch_idx]
        for i, response in enumerate(batch_responses):
            try:
                cleaned_json_str = response.choices[0].message.content.strip()  # type: ignore
                tp, fp, fn, total_act, total_pred = compare_taglist_dicts(
                    actual_labels_batch[i], cleaned_json_str
                )
                row_metrics.append(
                    {
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "total_actual": total_act,
                        "total_predicted": total_pred,
                    }
                )
                # [Glenn] the old code here was `df.at[batch_indices[i], "extracted_labels"] = cleaned_json_str`
                # But batch_indices was never defined so I calculate the correct index in the original dataframe
                # TODO: check with @Huzaifa about this and ensure my patch is correct
                idx = batch_idx * args.batch_size + i
                df.at[idx, "extracted_labels"] = cleaned_json_str
            except Exception as e:
                logger.error(f"Error processing: {e}")
                row_metrics.append(
                    {"tp": 0, "fp": 0, "fn": 0, "total_actual": 0, "total_predicted": 0}
                )
                idx = batch_idx * args.batch_size + i
                df.at[idx, "extracted_labels"] = None

        pbar.set_description(f"Batch {batch_idx + 1}/{total_batches}")
        logger.info(f"Processed responses for batch {batch_idx + 1}.")

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

    logger.info("Final micro-average metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Accuracy (Jaccard): {accuracy:.4f}")

    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy (Jaccard)", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1],
        }
    )

    return df, metrics_df
