import time
from datetime import date
import pandas as pd
from datasets import load_dataset
import together
from prompts import convfinqa_prompt
from tokens import tokens
from pathlib import Path

def convfinqa_inference(args):
    together.api_key = args.api_key
    today = date.today()
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    results_path = (
        ROOT_DIR
        / "results"
        / args.task
        / f"{args.task}_{args.model}_{today.strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("gtfintechlab/ConvFinQa", token=args.hf_token)

    # Initialize lists to store actual labels and model responses
    context = []
    llm_responses = []
    actual_labels = []
    complete_responses = []

    start_t = time.time()
    
    for entry in dataset["train"]:  # type: ignore
        try:
            pre_text = " ".join(entry["pre_text"])
            post_text = " ".join(entry["post_text"])
            table_text = " ".join([" ".join(map(str, row)) for row in entry["table_ori"]])
            question_0 = str(entry["question_0"]) if entry["question_0"] is not None else ""
            question_1 = str(entry["question_1"]) if entry["question_1"] is not None else ""
            answer_1 = str(entry["answer_1"]) if entry["answer_1"] is not None else ""

            combined_text = f"{pre_text} {post_text} {table_text} {question_0} {question_1}"

            # Prepare the context and actual label only if the rest is successful
            actual_label = answer_1
            
            # Call the API for model response
            model_response = together.Complete.create(
                prompt=convfinqa_prompt(combined_text),
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                stop=tokens(args.model),
            )

            response_label = model_response["output"]["choices"][0]["text"].strip()

       
            context.append(combined_text)
            actual_labels.append(actual_label)
            llm_responses.append(response_label)
            complete_responses.append(model_response)

      
            df = pd.DataFrame(
                {
                    "context": context,
                    "response": llm_responses,
                    "actual_label": actual_labels,
                    "complete_responses": complete_responses,
                }
            )
            df.to_csv(results_path, index=False)
            time.sleep(10)

        except Exception as e:
            print(f"Error encountered: {e}")
            time.sleep(10.0)  # Adding a delay before continuing the loop

    return df
