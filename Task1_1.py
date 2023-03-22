import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Carga del dataset
data = pd.read_csv('bank_transactions.csv')
#print(data.head()) 


class KMeans:
    
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        # Inicializar los centroides aleatoriamente
        self.centroids = random.sample(list(X), self.n_clusters)
        
        for i in range(self.max_iter):
            # Asignar cada punto al centroide más cercano
            clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                distances = [self.distance(x, c) for c in self.centroids]
                closest_centroid_index = distances.index(min(distances))
                clusters[closest_centroid_index].append(x)
                
            # Recalcular los centroides como el promedio de los puntos asignados a cada centroide
            prev_centroids = self.centroids.copy()
            for i, cluster in enumerate(clusters):
                if cluster:
                    self.centroids[i] = tuple(np.mean(cluster, axis=0))
            
            # Comprobar si los centroides han dejado de cambiar de posición
            if self.centroids == prev_centroids:
                break
    
    def predict(self, X):
        # Asignar cada punto al centroide más cercano
        y = []
        for x in X:
            distances = [self.distance(x, c) for c in self.centroids]
            closest_centroid_index = distances.index(min(distances))
            y.append(closest_centroid_index)
        return y
    
    def distance(self, a, b):
        return sum([(ai - bi) ** 2 for ai, bi in zip(a, b)]) ** 0.5


# Eliminación de la columna de categoría, porque la columna TransactionID y CustumerID no tienen datos relevantes al analisis de datos.
data = data.drop('TransactionID', axis=1)
data = data.drop('CustomerID', axis=1)
data = data.drop('CustomerDOB', axis =  1)


""" # Convertir las columnas 'CustomerDOB' y 'TransactionDate' a valores numéricos
data['CustomerDOB'] = pd.to_datetime(data['CustomerDOB'], format='%d/%m/%Y', errors='coerce')
data['CustomerDOB'] = data['CustomerDOB'].fillna(0).apply(lambda x: x.toordinal()) """

data['TransactionDate'] = pd.to_datetime(data['TransactionDate']).apply(lambda x: x.toordinal())

pca = PCA(n_components=2)

# Convertir las columnas restantes en variables binarias
categorical = ['CustGender', 'CustLocation']
new_cols = pd.get_dummies(data[categorical])
data = pd.concat([data.drop(categorical, axis=1), new_cols], axis = 1)
print(data)
data = pca.fit_transform(data)

# Ajuste del modelo K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
labels = kmeans.predict(data)

# Visualización de los clusters
colors = ['r', 'g', 'b']
for i in range(len(data)):
    plt.scatter(data[i, 0], data[i, 1], color=colors[labels[i]])
plt.scatter(kmeans.centroids[0][0], kmeans.centroids[0][1], marker='*', s=200, color='black')
plt.scatter(kmeans.centroids[1][0], kmeans.centroids[1][1], marker='*', s=200, color='black')
plt.scatter(kmeans.centroids[2][0], kmeans.centroids[2][1], marker='*', s=200, color='black')
plt.title('Clusters')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
