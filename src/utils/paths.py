from pathlib import Path

# 1. This is the ROOT (The absolute top level of project)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 2. These are  sub-divisions (Relative to the root)
DATA_DIR_PROCESS = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SRC_DIR = ROOT_DIR / "src"

# Define the sub-directories inside your results folder
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"


# Example: Ensuring they all exist
for directory in [DATA_DIR_PROCESS, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)