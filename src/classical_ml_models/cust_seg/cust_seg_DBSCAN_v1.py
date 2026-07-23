import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from src.utils.paths import DATA_DIR_PROCESS
from src.utils.tracker import log_experiment

try:
    data = pd.read_csv(DATA_DIR_PROCESS / "Mall_Customers_Data.csv")
except FileNotFoundError:
    print(f"Error : file not found at {DATA_DIR_PROCESS} ")
    sys.exit(1)

# cols to use for model
features = ["Annual Income (k)", "Spending Score"]
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# model
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(X_scaled)

data['Cluster'] = cluster_labels

print(data['Cluster'].value_counts())

print(data)