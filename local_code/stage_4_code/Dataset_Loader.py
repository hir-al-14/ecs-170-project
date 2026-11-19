import csv
from pathlib import Path
from local_code.stage_4_code.Config import (
    CLASSIFICATION_DIR,
    GENERATION_DIR
)

class Stage4DatasetLoader:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        self.data_dir = (
            self.project_root / "data" / "stage_4_data"
        )

        self.classification_dir = (
            self.data_dir / "text_classification"
        )

        self.generation_dir = (
            self.data_dir / "text_generation" / "data"
        )

    def load_classification(self):
        """
        Load IMDb positive/negative movie reviews.

        Labels:
            0 = negative
            1 = positive
        """

        dataset = {}

        for split in ["train", "test"]:
            X = []
            y = []

            for label_name, label_id in [
                ("neg", 0),
                ("pos", 1)
            ]:
                folder = (
                    self.classification_dir
                    / split
                    / label_name
                )

                if not folder.is_dir():
                    raise FileNotFoundError(folder)

                for file in sorted(folder.glob("*.txt")):
                    text = file.read_text(
                        encoding="utf-8"
                    )

                    X.append(text)
                    y.append(label_id)

            dataset[split] = {
                "X": X,
                "y": y
            }

        return dataset

    def load_generation(self):
        """Read the instructor's CSV into individual jokes (not token windows).

        The supplied file is named ``data`` without a .csv extension.
        csv.DictReader preserves quoted commas and multiline joke fields.
        """
        if not self.generation_dir.is_file():
            raise FileNotFoundError(self.generation_dir)

        with self.generation_dir.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            if not reader.fieldnames or "Joke" not in reader.fieldnames:
                raise ValueError("Generation CSV must contain a Joke column")
            jokes = []
            for row_number, row in enumerate(reader, start=2):
                if None in row or row["Joke"] is None:
                    raise ValueError(f"Malformed generation CSV record {row_number}")
                joke = row["Joke"].strip()
                if joke:
                    jokes.append(joke)
        if not jokes:
            raise ValueError("Generation CSV contains no nonempty jokes")
        return jokes
