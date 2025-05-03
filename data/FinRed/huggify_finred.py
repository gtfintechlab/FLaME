import os
from huggingface_hub import login
from datasets import Dataset, DatasetDict
import logging
from flame.config import DATA_DIR, LOG_LEVEL

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_ORGANIZATION = "gtfintechlab"
DATASET = "FinRed"
login(HUGGINGFACEHUB_API_TOKEN)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def parse_file(file_path):
    sentences = []
    entities = []
    relations = []

    # Specify encoding when opening the file
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                parts = line.strip().split(" | ")
                sentence_parts = []
                triplets = []
                for part in parts:
                    if part.count(" ; ") == 2:
                        triplets.append(part)
                    else:
                        sentence_parts.append(part)

                sentence = " | ".join(sentence_parts)  # Reconstruct the sentence
                entities_list = []
                relations_list = []
                for triplet in triplets:
                    try:
                        entity1, entity2, relation = triplet.split(" ; ")
                        entities_list.append((entity1, entity2))
                        relations_list.append(relation)
                    except ValueError as e:
                        logger.error(f"Error parsing triplet '{triplet}': {e}")
                        print(f"Problematic triplet: {triplet}")
                        continue
                sentences.append(sentence)
                entities.append(entities_list)
                relations.append(relations_list)

    return sentences, entities, relations


def huggify_data_finred(push_to_hub=False):
    try:
        directory_path = DATA_DIR / "FinRed"
        logger.debug(f"Directory path: {directory_path}")

        train_sentences, train_entities, train_relations = parse_file(
            f"{directory_path}/train.txt"
        )
        test_sentences, test_entities, test_relations = parse_file(
            f"{directory_path}/test.txt"
        )
        val_sentences, val_entities, val_relations = parse_file(
            f"{directory_path}/dev.txt"
        )

        splits = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "sentence": train_sentences,
                        "entities": train_entities,
                        "relations": train_relations,
                    }
                ),
                "test": Dataset.from_dict(
                    {
                        "sentence": test_sentences,
                        "entities": test_entities,
                        "relations": test_relations,
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "sentence": val_sentences,
                        "entities": val_entities,
                        "relations": val_relations,
                    }
                ),
            }
        )

        # Push to HF Hub
        if push_to_hub:
            splits.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name="main",
                private=True,
                token=HUGGINGFACEHUB_API_TOKEN,
            )

        logger.info("Finished processing FinRed dataset")
        return splits

    except Exception as e:
        logger.error(f"Error processing FinRed dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    huggify_data_finred(push_to_hub=True)
