import pandas as pd
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from src.utils.paths import METRICS_DIR 

def log_experiment(model, metrics, experiment_name="Experiment_Log"):
    """
    Universally saves model metrics and parameters for both Scikit-Learn and Keras.
    The CSV file will be named exactly what you pass as 'experiment_name'.
    """
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    algo_name = estimator.__class__.__name__
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Start the row with the basics
    row_data = {
        "Timestamp": [timestamp],
        "Algorithm": [algo_name],
    }
    
    # 2. Add the parameters (Safely handle Scikit-Learn vs Keras)
    if hasattr(estimator, 'get_params'):
        # Scikit-Learn behavior
        params_dict = estimator.get_params()
    elif hasattr(estimator, 'get_config'):
        # Keras behavior - Extract basic high-level architecture
        params_dict = {
            "keras_layers": len(estimator.layers),
            "keras_trainable_params": estimator.count_params()
        }
    else:
        # Fallback for unknown model types
        params_dict = {"Parameters": "Unknown"}

    for param_name, param_value in params_dict.items():
        # Convert complex objects (like functions) to strings to prevent CSV errors
        row_data[f"Param_{param_name}"] = [str(param_value)]
    
    # 3. Add the metrics to the end of the row
    for metric_name, value in metrics.items():
        row_data[metric_name] = [round(value, 4)]

    df = pd.DataFrame(row_data)
   
    csv_path = METRICS_DIR / f"{experiment_name}.csv"

    # Save/Append data
    if os.path.exists(csv_path):
        # Concatenate allows pandas to handle new columns dynamically if parameters change
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        
    print(f"\n--- Metrics successfully logged to {csv_path} ---")