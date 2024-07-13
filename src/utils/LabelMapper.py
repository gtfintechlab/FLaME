class LabelMapper:
    def __init__(self, task):
        self.mappings = {
            "finer_ord": {
                0: "other",
                1: "person_b",
                2: "person_i",
                3: "location_b",
                4: "location_i",
                5: "organisation_b",
                6: "organisation_i",
            },
            "fomc_communication": {0: "dovish", 1: "hawkish", 2: "neutral"},
            "numclaim_detection": {0: "outofclaim", 1: "inclaim"},
            "sentiment_analysis": {0: "positive", 1: "negative", 2: "neutral"},
        }
        if task not in self.mappings:
            raise ValueError(f"Task {task} not found in mappings.")
        self.task = task

    def encode(self, label_name):
        reversed_mapping = {v: k for k, v in self.mappings[self.task].items()}
        return reversed_mapping.get(label_name, -1)

    def decode(self, label_number):
        return self.mappings[self.task].get(label_number, "undefined").upper()
