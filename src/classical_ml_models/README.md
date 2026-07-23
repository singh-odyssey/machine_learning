# Models Directory

This directory stores trained machine learning models.

# Classical ML Source

This folder contains the classical machine learning pipelines that were previously under `src/models/`.

Use this area for tabular preprocessing, training scripts, clustering, regression, and classification workflows built with scikit-learn and related libraries.

Saved artifacts from these scripts should go to `models/classical_ml/`.
## Best Practices
Models should be saved with a clear and consistent naming convention: `[problem_domain]_[model_algorithm]_[version].joblib`.
For example: `iris_knn_v1.joblib` or `Mall_Customer_KMEANS_v1.joblib`.