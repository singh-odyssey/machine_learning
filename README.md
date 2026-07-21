# Machine Learning Models Repository

A machine learning project repository designed for learning, practicing, and building classical machine learning and deep learning projects.

---

## 🎯 Repository Structure Diagram

```
machine_learning/
├── 📂 config/
│   └── training_configs/         # YAML/JSON configs for experiments
│
├── 📂 data/                      # Segregated by state
│   ├── raw/                      # Original data (don't modify)
│   └── processed/                # Cleaned, ready-to-use data
│
├── 📂 models/                    # Saved artifacts by family
│   ├── classical_ml/
│   └── deep_learning/
│       ├── vision/
│       └── text/
│
├── 📂 notebooks/
│   ├── classical_ml/
│   └── deep_learning/
│       ├── vision/
│       └── text/
│
├── 📂 results/
│   ├── logs/
│   ├── metrics/
│   └── plots/
│
├── 📂 src/
│   ├── classical_ml/
│   ├── deep_learning/
│   │   ├── vision/
│   │   └── text/
│   └── utils/
│
├── 📂 tests/                     # ✅ TEST CODE
│
├── 📄 pyproject.toml             # Makes the src tree installable
├── 📄 requirements.txt           # 📦 Python dependencies
├── 📄 LICENSE                    # 📋 License
└── 📄 README.md                  # 📖 This file
```

---

## 📁 What's Inside - Folder Guide

### **data/** - Your Data Storage
This folder holds all the data for your project:
- **raw/**: Keep your original data files here. Don't change these files!
- **processed/**: Store your cleaned and prepared data here. This is what your models will use.

### **src/** - Your Project Code
All your Python code goes here, organized by workflow:
- **classical_ml/**: Tabular ML pipelines, preprocessing, training, and evaluation
- **deep_learning/**: Shared neural utilities plus vision/text specific modules
- **utils/**: Helper functions and shared path/tracking utilities

### **notebooks/** - Experimentation Area
Jupyter notebooks for exploring your data and testing new ideas. You can write code, see results immediately, and add notes. Existing classical ML notebooks now live under `notebooks/classical_ml/`.

### **models/** - Your Trained Models
Store serialized outputs by family:
- **classical_ml/**: scikit-learn and joblib files
- **deep_learning/vision/**: CNN, ViT, and diffusion checkpoints
- **deep_learning/text/**: transformer, LLM, and RAG artifacts

### **results/** - Your Experiment Results
All outputs from your machine learning experiments go here:
- **plots/**: Charts and graphs showing your data and results
- **metrics/**: Numbers showing how well your models performed
- **logs/**: Records of what happened during training for debugging

### **tests/** - Quality Control
Test files to make sure your code works correctly. You can run tests to catch bugs early.

### **config/** - Settings
Configuration files for your project settings and parameters.

### **requirements.txt** - Dependencies List
A list of all Python packages your project needs. Install them with:
```bash
pip install -r requirements.txt
```

The main packages used are:
- **numpy**: For working with numbers and arrays
- **pandas**: For working with data tables
- **scikit-learn**: Pre-built machine learning models
- **matplotlib & seaborn**: For creating charts and visualizations

### **pyproject.toml** - Package Setup
A modern packaging file that makes your `src` tree installable.

---

## 🚀 Quick Start Guide

### 1. Get Ready
Clone the project and navigate to it:
```bash
git clone <repository-url>
cd machine_learning
```

### 2. Create a Safe Python Environment
This keeps your project separate from other projects on your computer:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Everything You Need
```bash
pip install -r requirements.txt
```

---

## 📊 Typical Workflow - How to Use This Project

1. **Get your data** → Put raw files in `data/raw/`

2. **Clean your data** → Write Python scripts in `src/classical_ml/` or `src/utils/` to clean the data and save results to `data/processed/`

3. **Build models** → Write classical ML code in `src/classical_ml/` and deep learning code in `src/deep_learning/`

4. **Test your code** → Run `pytest tests/` to make sure everything works

5. **Train and test** → Use Jupyter notebooks in `notebooks/` to experiment with your models

6. **Save everything** → 
   - Save trained models to `models/classical_ml/` or `models/deep_learning/`
   - Save charts to `results/plots/`
   - Save performance numbers to `results/metrics/`
   - Save training logs to `results/logs/`

---

## 📝 Project Information

- **Python Version**: Requires Python 3.7 or higher
- **License**: MIT License
- **Purpose**: A learning and practice repository for machine learning projects following industry standards.