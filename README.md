# Machine Learning Models Repository

This repository contains classical machine-learning and deep-learning experiments, data, notebooks, and trained artifacts used for learning and prototyping.
---

## 🎯 Current repository layout (directories only)

```
machine_learning/
├── config/
│   └── training_configs/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── classical_ml/
│   └── deep_learning/
│       ├── text/
│       └── vision/
├── notebooks/
│   ├── classical_ml/
+│   └── deep_learning/
├── results/
│   ├── logs/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── classical_ml/
│   ├── student_marks_pred/
│   ├── data_processing/
│   ├── deep_learning/
│   │   ├── text/
│   │   └── vision/
│   └── utils/
├── tests/
```

---

## 📁 Quick folder guide

- **config/**: Experiment and training configuration files (e.g., `training_configs/`).
- **data/raw/**: Original source data files (CSV).
- **data/processed/**: Cleaned, model-ready datasets.
- **models/**: Saved model artifacts organized by family.
- **notebooks/**: Analysis and experimentation notebooks.
- **results/**: Experiment outputs (`logs/`, `metrics/`, `plots/`).
- **src/**: Project code organized into subpackages (`classical_ml/`, `student_marks_pred/`, `data_processing/`, `deep_learning/`, `utils/`).
- **tests/**: Unit and integration tests.

---

## 🚀 Quick start

1. Create and activate a Python virtualenv:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Explore notebooks or run a script. Example: run the iris script

```bash
python src/classical_ml/iris_.py
```

4. Run tests (if any):

```bash
pytest tests/
```

---

## 📝 Notes

- Python 3.8+ is recommended.
- Use `src/utils/paths.py` to build reproducible file paths across the project.
- If you add or move files, update this README to keep the structure accurate.

---

If you'd like, I can also add a short CONTRIBUTING section, update `requirements.txt` with exact pinned versions, or generate a small example script that loads `data/processed/Mall_Customers_Data.csv` and trains the KMeans model located in `models/classical_ml/`.