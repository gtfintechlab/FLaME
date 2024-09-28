# TODO: (Glenn) If this is a formatter just for ectsum we should keep it in the associated data\task folder not here

class Formatter:
    def __init__(self):
        pass

    def format_df(self, input_df):
        df = input_df.copy()

        for i, row in df.iterrows():
            if not isinstance(row["output"], list):
                row["output"] = [
                    line.strip() for line in row["output"].split("\n") if line.strip()
                ]
                row["output"] = row["output"][:1]

            if not isinstance(row["predicted_text"], list):
                row["predicted_text"] = [
                    line.strip()
                    for line in row["predicted_text"].split("\n")
                    if line.strip()
                ]
                row["predicted_text"] = row["predicted_text"][:1]

        return df
