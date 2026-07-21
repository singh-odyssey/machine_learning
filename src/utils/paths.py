from pathlib import Path

# Root of the repository.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Data buckets.
DATA_DIR = ROOT_DIR / "data"
DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_PROCESS = DATA_DIR / "processed"

# Model buckets.
MODELS_DIR = ROOT_DIR / "models"
CML_MODELS_DIR = MODELS_DIR / "classical_ml"
DEEP_LEARNING_MODELS_DIR = MODELS_DIR / "deep_learning"
VISION_MODELS_DIR = DEEP_LEARNING_MODELS_DIR / "vision"
TEXT_MODELS_DIR = DEEP_LEARNING_MODELS_DIR / "text"

# Result buckets.
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Source tree.
SRC_DIR = ROOT_DIR / "src"

for directory in [
    DATA_DIR_RAW,
    DATA_DIR_PROCESS,
    CML_MODELS_DIR,
    VISION_MODELS_DIR,
    TEXT_MODELS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)