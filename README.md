# Machine Learning Models Repository
This Repository contains various types of machine learning model for practicing and learning purposes. 

### Included Model Types:
- Supervised 
- Unsupervised 
- Reinforcement 
---

## 📁 Repository Structure

```
machine_learning/
│
├── data/                          # Data Management
│   ├── raw/                       # Original, unmodified data
│   ├── processed/                 # Cleaned, preprocessed data
│   └── README.md                  # Data documentation
│
├── src/                           # Source Code
│   ├── __init__.py
│   ├── config.py                  # Project configuration
│   ├── data_processing/           # Data loading & preprocessing
│   │   ├── data_loader.py         # Load data from various sources
│   │   └── preprocessing.py       # Clean & transform data
│   ├── models/                    # Machine learning models
│   │   ├── regression.py          # Regression algorithms
│   │   ├── classification.py      # Classification algorithms
│   │   ├── unsupervised.py        # Clustering & dimensionality reduction
│   │   └── reinforcement.py       # RL algorithms
│   └── utils/                     # Utility functions
│       ├── metrics.py             # Performance evaluation metrics
│       └── visualization.py       # Plotting & visualization
│
├── notebooks/                     # Jupyter Notebooks
│   └── 01_exploratory_analysis.ipynb  # EDA examples
│
├── models/                        # Trained Model Storage
│   └── README.md                  # Model documentation
│
├── results/                       # Outputs & Results
│   ├── plots/                     # Generated visualizations
│   ├── metrics/                   # Performance metrics
│   └── logs/                      # Training logs
│
├── tests/                         # Unit Tests
│   ├── test_models.py             # Model tests
│   └── test_data_processing.py    # Data processing tests
│
├── config/                        # Configuration Files
│   └── __init__.py
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🏭 Industry Best Practices Explained

### 1. **Data Organization** (`/data`)
- **raw/**: Original data sources never modified, ensuring reproducibility
- **processed/**: Clean data ready for modeling, versioned and tracked
- **Benefit**: Maintains data integrity and allows easy rollback to raw data

### 2. **Modular Source Code** (`/src`)
- **Separation of Concerns**: Data processing, models, and utilities are separated
- **data_processing/**: Handles loading, cleaning, and transformation
- **models/**: Contains all ML algorithms organized by type
- **utils/**: Reusable functions for metrics and visualization
- **config.py**: Centralized configuration management
- **Benefit**: Easy to maintain, test, and reuse code across projects

### 3. **Organized Models** (`/models`)
- Store trained models with versioning
- Include metadata about training and performance
- **Benefit**: Easy model management, comparison, and deployment

### 4. **Comprehensive Testing** (`/tests`)
- Unit tests for models and data processing
- Validates correctness and prevents regressions
- **Benefit**: Ensures code reliability and facilitates refactoring

### 5. **Results Tracking** (`/results`)
- **plots/**: Visualizations for analysis and reporting
- **metrics/**: Performance metrics for model comparison
- **logs/**: Training and execution logs for debugging
- **Benefit**: Complete experiment tracking and reproducibility

### 6. **Documentation & Configuration**
- **requirements.txt**: Explicit dependency management
- **setup.py**: Package configuration and distribution
- **README files**: Project and module documentation
- **config.py**: Centralized settings management
- **Benefit**: Reproducibility across different environments

### 7. **Notebooks** (`/notebooks`)
- Used for exploratory analysis and experimentation
- Separate from production code for clarity
- **Benefit**: Easy sharing and collaboration on analysis

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/singh-odyssey/machine_learning.git
cd machine_learning
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Usage Examples

#### 1. **Data Processing**
```python
from src.data_processing import DataLoader, DataPreprocessor

# Load data
loader = DataLoader(data_path='data')
df = loader.load_csv('your_dataset.csv')

# Preprocess data
preprocessor = DataPreprocessor(test_size=0.2)
df_clean = preprocessor.handle_missing_values(df)
df_clean = preprocessor.remove_duplicates(df_clean)
X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(X, y)
```

#### 2. **Regression Model**
```python
from src.models.regression import RegressionModels
from src.utils.metrics import ModelMetrics

# Train model
model, predictions = RegressionModels.random_forest_regression(
    X_train, y_train, X_test, n_estimators=100
)

# Evaluate
metrics = RegressionModels.evaluate_regression(y_test, predictions)
ModelMetrics.print_metrics(metrics, metric_type='Regression')
```

#### 3. **Classification Model**
```python
from src.models.classification import ClassificationModels

# Train model
model, predictions = ClassificationModels.random_forest_classification(
    X_train, y_train, X_test, n_estimators=100
)

# Evaluate
metrics = ClassificationModels.evaluate_classification(y_test, predictions)
```

#### 4. **Unsupervised Learning**
```python
from src.models.unsupervised import UnsupervisedModels

# Clustering
model, labels = UnsupervisedModels.kmeans_clustering(X, n_clusters=3)
metrics = UnsupervisedModels.evaluate_clustering(X, labels)

# Dimensionality Reduction
pca_model, X_reduced = UnsupervisedModels.pca_dimensionality_reduction(X, n_components=2)
```

---

## 🔍 Detailed Component Explanations

### **Regression Models** (`src/models/regression.py`)
Used for predicting continuous values (e.g., house prices, temperature)
- **Linear Regression**: Simple baseline, interpretable
- **Ridge/Lasso**: Regularized versions to prevent overfitting
- **Random Forest**: Ensemble method, handles non-linearity
- **Gradient Boosting**: Sequential ensemble, high accuracy

### **Classification Models** (`src/models/classification.py`)
Used for predicting categories (e.g., spam/not spam, disease/no disease)
- **Logistic Regression**: Simple, interpretable baseline
- **Decision Trees**: Rule-based, interpretable
- **SVM**: Effective for high-dimensional data
- **KNN**: Simple instance-based method
- **Random Forest & Gradient Boosting**: Ensemble methods for higher accuracy

### **Unsupervised Learning** (`src/models/unsupervised.py`)
Finds patterns without labeled data
- **K-Means**: Partitions data into K clusters
- **DBSCAN**: Density-based clustering for arbitrary shapes
- **Hierarchical Clustering**: Creates cluster hierarchy
- **PCA**: Reduces dimensionality while preserving variance

### **Reinforcement Learning** (`src/models/reinforcement.py`)
Learns optimal actions through trial and reward
- **Q-Learning**: Off-policy value-based method
- **Policy Gradient**: On-policy, learns policy directly
- **Monte Carlo**: First-visit Monte Carlo control
- **TD Learning**: Temporal Difference learning

### **Data Processing** (`src/data_processing/`)
- **DataLoader**: Loads data from CSV, JSON, etc.
- **DataPreprocessor**: Handles missing values, scaling, encoding

### **Utilities** (`src/utils/`)
- **ModelMetrics**: Calculates and displays performance metrics
- **ModelVisualizer**: Creates plots for analysis

---

## 📊 Model Selection Guide

| Problem Type | Recommended Models | Use When |
|---|---|---|
| **Regression** | Linear, Ridge, Random Forest | Predicting continuous values |
| **Binary Classification** | Logistic, SVM, Random Forest | Two categories output |
| **Multi-Class Classification** | Random Forest, Gradient Boosting | Multiple categories output |
| **Clustering** | K-Means, DBSCAN | Grouping similar data points |
| **Dimensionality Reduction** | PCA | Reducing features/visualization |
| **Control/Decision Making** | Q-Learning, Policy Gradient | Sequential decision tasks |

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
# or
python -m unittest discover tests
```

---

## 📦 Dependencies

Core libraries:
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms
- **matplotlib & seaborn**: Visualization
- **jupyter**: Interactive notebooks
- **pytest**: Testing framework

See [requirements.txt](requirements.txt) for complete list.

---

## 🔄 Typical Workflow

1. **Data Preparation**
   - Place raw data in `data/raw/`
   - Use `DataLoader` and `DataPreprocessor` to clean data
   - Save processed data to `data/processed/`

2. **Exploratory Analysis**
   - Use Jupyter notebooks in `notebooks/`
   - Visualize data and relationships

3. **Model Development**
   - Select appropriate model from `src/models/`
   - Train and evaluate using utilities
   - Log results to `results/`

4. **Model Evaluation**
   - Use `ModelMetrics` for performance calculation
   - Use `ModelVisualizer` for visualization
   - Compare models and select best performer

5. **Model Storage**
   - Save best model to `models/` with metadata
   - Document hyperparameters and performance

6. **Testing & Validation**
   - Write unit tests in `tests/`
   - Validate on hold-out test set

---

## 💡 Best Practices Implemented

✅ **Modular Code**: Separate concerns for maintainability  
✅ **Configuration Management**: Centralized settings  
✅ **Data Versioning**: Track data transformations  
✅ **Model Versioning**: Store trained models with metadata  
✅ **Reproducibility**: Random seeds, documented processes  
✅ **Testing**: Unit tests for validation  
✅ **Documentation**: Comprehensive docstrings and READMEs  
✅ **Scalability**: Structure supports adding new models easily  
✅ **Environment Isolation**: Virtual environment with `requirements.txt`  

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

Created as an industry-standard template for machine learning projects.

---

## 🤝 Contributing

1. Create a new branch for your feature
2. Make your changes
3. Add tests for new functionality
4. Submit a pull request

---

**Happy Machine Learning! 🚀**