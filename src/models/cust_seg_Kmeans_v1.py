import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from src.utils.paths import DATA_DIR_PROCESS, MODELS_DIR
from src.utils.tracker import log_experiment
import joblib

try:
    data = pd.read_csv(DATA_DIR_PROCESS / "Mall_Customers_Data.csv")
except FileNotFoundError:
    print(f"Error : file not found at {DATA_DIR_PROCESS} ")
    sys.exit(1) 

# cols to use for model
features = ["Annual Income (k)", "Spending Score"]
X = data[features]

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# finding value of k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, init="k-means++", n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

kl = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
optimal_k = kl.elbow
print(f"Number of optimal clusters found for Dataset -> {optimal_k}")

# final model
kmeans = KMeans(n_clusters=optimal_k, init="k-means++" , random_state=42 , n_init=10)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled) # here predicting on same data we trained just to see how well it cluster and find silhouette

# performance metrics
silhouette_avg = silhouette_score(X_scaled, clusters)

print(f"Silhouette Score of Model is -> {silhouette_avg}")

# saving model
joblib.dump(kmeans, MODELS_DIR / "Mall_Customer_KMEANS_v1.joblib")

# saving metrics
metrics = {"Silhouette_Score": silhouette_avg, "WCSS": kmeans.inertia_}
log_experiment(kmeans, metrics, "Mall_Customers_KMeans_v1")  

# need to build pipeline for future prediction
# as new data will not be standardize