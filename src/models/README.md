# Models Directory

This directory stores trained machine learning models.

## Structure

Models are organized by type and date:
- `student_marks_predict_regression_model_v1.pkl`
- `spam_email_classification_model_v1.pkl`
- `geospatial_clustering_model_v1.pkl`
- etc.
## Best Practices

1. **Naming Convention**: `{project_name}_{model_type}_{version}.pkl` or `.h5` or .joblib
2. **Version Control**: Increment version numbers for model improvements
3. **Metadata**: Keep a `models_metadata.json` with model details:
   - Training date
   - Accuracy/performance metrics
   - Dataset used
   - Hyperparameters
   - Training time

## Model Formats

- `.pkl` - Pickle format for scikit-learn models
- `.h5` - HDF5 format for Keras/TensorFlow models
- `.pt` - PyTorch model files
- `.joblib` - Joblib format (recommended for large sklearn models)

## Loading Models

```python
import joblib

model = joblib.load('models/regression_model_v1.pkl')
```
