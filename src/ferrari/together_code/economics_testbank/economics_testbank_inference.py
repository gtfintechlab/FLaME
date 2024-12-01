from datetime import date
import pandas as pd
from datasets import load_dataset
from ferrari.config import LOG_DIR, LOG_LEVEL, RESULTS_DIR
from ferrari.together_code.prompts import economics_testbank_prompt
from ferrari.utils.logging_utils import setup_logger
from together import Together
from tqdm import tqdm

# Set up logger
logger = setup_logger(
    name="economics_testbank",
    log_file=LOG_DIR / "economics_testbank.log",
    level=LOG_LEVEL,
)

def format_prompt(question, choices):
    """
    Format the prompt in the required Q&A style.
    """
    formatted_choices = "\n".join(
        f"({chr(97 + i)}) {choice}" for i, choice in enumerate(choices)
    )
    return f"Q: {question}\nAnswer Choices:\n{formatted_choices}"

def format_answer(answer, rationale):
    """
    Format the answer and rationale in the required style.
    """
    return f"A: {rationale} Therefore, the answer is {answer}."

def generate_rationale(client, args, prompt):
    """
    Generate a rationale for a given prompt using the Together API.
    """
    try:
        model_response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop=None,
        )
        out = model_response.choices[0].message.content
        answer = out.splitlines()[0]
        rationale = out.splitlines()[1:]
        return answer, rationale
    except Exception as e:
        logger.error(f"Error generating rationale: {str(e)}")
        return "Error", "No rationale"

def refine_rationale(client, args, question, choices, correct_answer):
    """
    Refine a rationale using the new Q&A format.
    """
    formatted_prompt = format_prompt(question, choices)
    formatted_choices = [f"{choice} {'(CORRECT)' if i == correct_answer else ''}" 
                        for i, choice in enumerate(choices)]
    refined_prompt = (
        f"Q: {question}\n"
        f"Answer Choices:\n" + 
        "\n".join(f"({chr(97 + i)}) {choice}" for i, choice in enumerate(formatted_choices))
    )
    
    try:
        model_response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": refined_prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop=None,
        )
        return model_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error refining rationale: {str(e)}")
        return "Error"

def star_inference(args, dataset_path):
    """
    Perform STaR inference by first running inference and then refining incorrect outputs.
    """
    today = date.today()
    logger.info(f"Starting STaR inference on {today}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("gtfintechlab/Economics_TestBank", trust_remote_code=True)[
        "train"
    ]

    # Initialize Together API client
    client = Together()

    # Step 1: Generate initial rationales
    logger.info("Generating initial rationales...")
    results = []
    for i in tqdm(range(len(dataset)), desc="Generating Rationales"):
        row = dataset.iloc[i]
        question, choices = row["question"], row["choices"]
        correct_answer = row["answer"]
        prompt = format_prompt(question, choices)
        answer, rationale = generate_rationale(client, args, prompt)
        results.append({
            "Question": question,
            "Choices": choices,
            "Answer": answer,
            "Correct_Answer": correct_answer,
            "Generated_Rationale": rationale,
        })

    # Step 2: Evaluate and refine incorrect rationales
    logger.info("Refining incorrect rationales...")
    refined_results = []
    for result in tqdm(results, desc="Refining Incorrect Responses"):
        if result["Answer"] != result["Correct_Answer"]:
            refined_rationale = refine_rationale(
                client,
                args,
                result["Question"],
                result["Choices"],
                result["Correct_Answer"]
            )
            result["Refined_Rationale"] = refined_rationale
        else:
            result["Refined_Rationale"] = result["Generated_Rationale"]

        refined_results.append(result)

    return pd.DataFrame(refined_results)