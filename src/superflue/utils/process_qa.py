import pandas as pd
# TODO: (Glenn) This function is used to process QAs, but its one file one function. we can move it elsewhere or put it in a folder.

def process_qa_pairs(data):
    inputs, outputs = [], []

    for _, row in data.iterrows():

        input_str = row["pre_text"]
        if pd.notna(row["table_ori"]):
            input_str += " " + row["table_ori"]
        if pd.notna(row["post_text"]):
            input_str += " " + row["post_text"]

        # Multiple QA pairs (convfinqa)
        if "qa_1" in row:
            question_0 = row["qa_0"].get("question") if pd.notna(row["qa_0"]) else ""
            answer_0 = row["qa_0"].get("answer") if pd.notna(row["qa_0"]) else ""
            question_1 = row["qa_1"].get("question") if pd.notna(row["qa_1"]) else ""
            answer_1 = row["qa_1"].get("answer") if pd.notna(row["qa_1"]) else ""

            input_str += " " + question_0 + " " + question_1
            outputs.append(answer_0 + " " + answer_1)

        # Single QA pair (finqa)
        else:
            question = row["qa"].get("question") if pd.notna(row["qa"]) else ""
            answer = row["qa"].get("answer") if pd.notna(row["qa"]) else ""

            input_str += " " + question
            outputs.append(answer)

        inputs.append(input_str)

    return pd.DataFrame({"input": inputs, "output": outputs})
