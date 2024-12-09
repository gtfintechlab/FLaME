import csv
import io
import os
import re

from docx import Document


def parse_multiple_choice(text):
    """Parse multiple choice questions and answers from test bank text."""
    question_pattern = r"(\d+)\.\s+(.*?)\n(?:A\.\s+(.*?)\nB\.\s+(.*?)\nC\.\s+(.*?)\nD\.\s+(.*?))\nAnswer:\s*([A-D])"
    questions = re.finditer(question_pattern, text, re.DOTALL)

    parsed_questions = []
    for match in questions:
        # print("REACHES HERE")
        question_num = match.group(1)
        question_text = match.group(2).strip()
        options = (
            "A: "
            + match.group(3).strip()
            + "\nB: "
            + match.group(4).strip()
            + "\nC: "
            + match.group(5).strip()
            + "\nD: "
            + match.group(6).strip()
        )
        answer = match.group(7).strip()
        answer_match = re.search(rf"({answer}\:\s+.*?)(?:\n|$)", options)
        full_answer = answer_match.group(1).strip() if answer_match else answer

        # print(full_answer)

        # Remove extra newlines and spaces
        # options = re.sub(r"\n+", "\n", options)
        # options = re.sub(r"\s+", " ", options)

        # Format each option on a new line
        options = "\n".join(
            line.strip() for line in options.split("\n") if line.strip()
        )

        # Format the full prompt with question and options
        prompt = f"{question_num}. {question_text}\n\n{options}"

        # Clean up the prompt by removing extra whitespace
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        prompt = prompt.strip()

        parsed_questions.append({"prompt": prompt, "answer": full_answer})
        # break

    return parsed_questions


def write_to_csv(questions, output_file="test_bank.csv"):
    """Write parsed questions to CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "answer"])
        writer.writeheader()
        writer.writerows(questions)


def read_docx(file_path):
    """Read content from a docx file."""
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:  # Only add non-empty paragraphs
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        raise Exception(f"Error reading DOCX file: {str(e)}")


def parse_file(input_path):
    """
    Main function to process the test bank file.
    Args:
        input_path: Path to the input docx file
    """
    try:
        # Check if file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # Check file extension
        if not input_path.lower().endswith(".docx"):
            raise ValueError("Input file must be a .docx file")

        # Read the docx file
        print(f"Reading file: {input_path}")
        content = read_docx(input_path)

        # Parse the questions
        questions = parse_multiple_choice(content)

        if not questions:
            print(
                "Warning: No questions were parsed. Check if the file format matches the expected pattern."
            )
            return

        # Generate output filename
        output_file = os.path.splitext(input_path)[0] + "_parsed.csv"

        # Write to CSV
        write_to_csv(questions, output_file)

        # print(
        #     f"Successfully parsed {len(questions)} questions and saved to {output_file}"
        # )

        # Preview first few entries
        # print("\nFirst few entries:")
        # for q in questions[:3]:
        #     print("\nPrompt:")
        #     print(q["prompt"])
        #     print(f"Answer: {q['answer']}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parse_test_bank.py <path_to_docx_file>")
        sys.exit(1)

    parse_file(sys.argv[1])
