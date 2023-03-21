import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Cargar el dataset
url = 'bank_transactions.csv'
data = pd.read_csv(url)

# Preprocesamiento
data = data.drop(['CustomerID'], axis=1) # Eliminar la columna 'CustomerID'
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Implementación básica de K-Means
class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        if self.random_state:
            random.seed(self.random_state)
        initial_indices = random.sample(range(X.shape[0]), self.n_clusters)
        self.cluster_centers_ = X[initial_indices]

        for _ in range(self.max_iter):
            self.labels_ = [np.argmin(np.linalg.norm(x - self.cluster_centers_, axis=1)) for x in X]
            self.cluster_centers_ = [X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)]

        return self

    def predict(self, X):
        return np.array([np.argmin(np.linalg.norm(x - self.cluster_centers_, axis=1)) for x in X])

# Entrenar y predecir clusters con KMeansCustom
kmeans_custom = KMeansCustom(n_clusters=4, random_state=42)
kmeans_custom.fit(data_scaled)
clusters = kmeans_custom.predict(data_scaled)

# Evaluar el desempeño con el índice de Silueta
silhouette_avg = silhouette_score(data_scaled, clusters)
print("El índice de Silueta promedio es:", silhouette_avg)

# PCA para visualización
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# Graficar los clusters
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Visualización de clusters usando PCA')
plt.show()
