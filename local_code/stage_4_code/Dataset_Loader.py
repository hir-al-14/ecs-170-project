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
        """
        Load text files used for joke generation.
        """

        if not self.generation_dir.is_dir():
            raise FileNotFoundError(
                self.generation_dir
            )

        files = sorted(
            self.generation_dir.rglob("*.txt")
        )

        if not files:
            raise FileNotFoundError(
                "No .txt files found in "
                f"{self.generation_dir}"
            )

        texts = []

        for file in files:
            content = file.read_text(
                encoding="utf-8"
            )
            texts.append(content)

        return texts