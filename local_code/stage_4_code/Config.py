
from pathlib import Path

# Project root (independent of working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset directories
DATA_DIR = PROJECT_ROOT / "data" / "stage_4_data"

CLASSIFICATION_DIR = DATA_DIR / "text_classification"
GENERATION_DIR = DATA_DIR / "text_generation" / "data"

# GloVe directory and filenames
# Download the GloVe files in the data/stage_4_data
GLOVE_DIR = DATA_DIR / "glove"

GLOVE_FILES = {
    50: "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt",
    100: "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt",
    200: "wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt",
    300: "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt",
}

# Default experiment settings
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 10
MAX_SEQUENCE_LENGTH = 200

# Selected GloVe file
GLOVE_PATH = GLOVE_DIR / GLOVE_FILES[EMBEDDING_DIM]

def get_glove_path(dim):
    """Return the corresponding GloVe file path."""
    if dim not in GLOVE_FILES:
        raise ValueError(
            f"Unsupported dimension: {dim}. "
            f"Choose from {list(GLOVE_FILES)}"
        )

    return GLOVE_DIR / GLOVE_FILES[dim]