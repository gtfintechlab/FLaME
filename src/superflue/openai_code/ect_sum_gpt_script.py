
from datasets import load_dataset

from superflue.openai_code.ect_sum_main import Evaluate


def process_dataset(evaluator, dataset_file, model_name):

    results = evaluator.iterate_df(dataset_file)

    output_path = evaluator.save_data(dataset_file, model_name, results)

    evaluator.append_scores(output_path)


DATA_DIR = "data"

evaluator = Evaluate()

"""datasets = ["train.csv", "test.csv", "val.csv"]
model_name = "GPT-3.5-Turbo"

for dataset in datasets:
    dataset_file = os.path.join(DATA_DIR, dataset)
    process_dataset(evaluator, dataset_file, model_name)"""

dataset = load_dataset("https://huggingface.co/datasets/gtfintechlab/ECTSum")
data_files = {"train": "train.csv", "test": "test.csv", "val": "val.csv"}
