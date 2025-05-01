"""MMLU dataset loader and processor."""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from superflue.code.mmlu.mmlu_constants import ECONOMICS_SUBJECTS, SPLITS
from superflue.config import LOG_DIR, LOG_LEVEL

# Configure logging to show on console with timestamp and level
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(LOG_DIR) / "mmlu_loader.log"),
    ],
)
logger = logging.getLogger(__name__)


class MMLULoader:
    """Loader for the MMLU dataset."""

    def __init__(
        self,
        subjects: Optional[List[str]] = None,
        split: str = "test",
        num_few_shot: int = 5,
    ):
        """
        Initialize MMLU loader.

        Args:
            subjects: List of MMLU subjects to load. If None, loads economics subjects.
            split: Dataset split to load ('dev', 'validation', or 'test').
            num_few_shot: Number of few-shot examples to load from dev set.

        Raises:
            ValueError: If split is not valid
        """
        self.subjects = subjects or ECONOMICS_SUBJECTS
        if split not in SPLITS:
            raise ValueError(f"Split must be one of {SPLITS}")
        self.split = split
        self.num_few_shot = num_few_shot
        self.dataset = None
        self.few_shot_examples = None

    def load_few_shot_examples(self) -> List[Dict]:
        """
        Load few-shot examples from dev set.

        Returns:
            List of dictionaries containing few-shot examples
        """
        logger.info(f"Loading {self.num_few_shot} few-shot examples from dev set")
        examples = []

        for subject in self.subjects:
            try:
                dev_set = load_dataset("cais/mmlu", subject, split="dev")
                # Take num_few_shot/len(subjects) examples from each subject
                num_examples = max(1, self.num_few_shot // len(self.subjects))

                for i in range(min(num_examples, len(dev_set))):
                    examples.append(
                        {
                            "subject": subject,
                            "question": dev_set[i]["question"],
                            "choices": dev_set[i]["choices"],
                            "answer": dev_set[i]["answer"],
                        }
                    )

                logger.info(f"Loaded {num_examples} examples from {subject}")

            except Exception as e:
                logger.error(f"Error loading dev examples for {subject}: {str(e)}")
                continue

        if not examples:
            raise ValueError("No few-shot examples could be loaded")

        # Trim to exact number requested
        examples = examples[: self.num_few_shot]
        self.few_shot_examples = examples
        return examples

    def load(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Load MMLU dataset for specified subjects and split.

        Returns:
            Tuple containing:
            - DataFrame with the loaded dataset
            - List of few-shot examples from dev set

        Raises:
            ValueError: If no data could be loaded
        """
        logger.info(f"Loading MMLU dataset for subjects: {self.subjects}")

        # Load few-shot examples first
        few_shot_examples = self.load_few_shot_examples()

        # Load main dataset
        all_data = []
        for subject in self.subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split=self.split)

                # Convert to DataFrame
                df = pd.DataFrame(
                    {
                        "subject": subject,
                        "question": dataset["question"],
                        "choices": dataset["choices"],
                        "answer": dataset["answer"],
                    }
                )
                all_data.append(df)
                logger.info(f"Successfully loaded {subject} ({len(df)} questions)")

            except Exception as e:
                logger.error(f"Error loading subject {subject}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data was successfully loaded")

        self.dataset = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total loaded questions: {len(self.dataset)}")
        return self.dataset, few_shot_examples

    def get_subjects_summary(self) -> Dict[str, int]:
        """
        Get summary of number of questions per subject.

        Returns:
            Dictionary mapping subject names to number of questions.

        Raises:
            ValueError: If dataset has not been loaded yet
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        return self.dataset.groupby("subject").size().to_dict()
