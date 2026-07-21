# Models Directory

This folder is the top-level container for stored model artifacts.

## Structure

- `classical_ml/`: scikit-learn and joblib outputs
- `deep_learning/vision/`: CNN, ViT, diffusion, and other image checkpoints
- `deep_learning/text/`: transformer, LLM, and RAG artifacts

## Best Practices

1. Keep generated weights and datasets out of version control.
2. Use versioned names for saved artifacts.
3. Store training metadata alongside metrics in `results/metrics/`.

## Common Formats

- `.joblib` and `.pkl` for classical ML
- `.keras` and `.h5` for Keras / TensorFlow
- `.pt` and `.pth` for PyTorch
- `.onnx` for portable inference exports
