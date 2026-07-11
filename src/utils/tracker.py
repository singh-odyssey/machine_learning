# algo to save performance of model with different parameters on each run 

import pandas as pd
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from src.utils.paths import METRICS_DIR 

def log_experiment(model, metrics, experiment_name="Experiment_Log"):
    """
    Universally saves model metrics and parameters.
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
    
    # 2. Add the parameters
    params_dict = estimator.get_params()
    for param_name, param_value in params_dict.items():
        row_data[f"Param_{param_name}"] = [param_value]
    
    # 3. Add the metrics to the end of the row
    for metric_name, value in metrics.items():
        row_data[metric_name] = [round(value, 4)]

    df = pd.DataFrame(row_data)
   
    csv_path = METRICS_DIR / f"{experiment_name}.csv"

    # Save/Append data
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
        
    print(f"\n--- Metrics successfully logged to {csv_path} ---")
    