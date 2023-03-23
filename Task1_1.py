import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Carga del dataset
data = pd.read_csv('bank_transactions.csv')
#print(data.head()) 


class KMeans:
    
    def __init__(self, X, n_clusters= 4, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X = X
            
    
    def compare_centroids(self, prev_centroids, curr_centroids):
        for i in range(len(prev_centroids)):
            for j in range(len(prev_centroids[i])):
                if prev_centroids[i][j] != curr_centroids[i][j]:
                    return False
        return True
    
    def find_initial_centroids(self, k):
        centroids = []
        maxs = np.max(self.X, axis = 0)
        mins = np.min(self.X, axis = 0)
        iter = k+1
        for i in range(1, iter):
            point_perc = i/iter
            centroids.append([point_perc * (maxs[j] - mins[j]) + mins[j] for j in range(self.X.shape[1])])
        return np.array(centroids)


    def fit(self, k = None):

        if not k: k = self.n_clusters
        
        self.centroids = self.find_initial_centroids(k)
        for i in range(self.max_iter):
            # Asignar cada punto al centroide más cercano
            clusters = [[] for _ in range(k)]
            wss = 0
            for x in self.X:
                distances = [self.distance(x, c) for c in self.centroids]
                closest_centroid_index = distances.index(min(distances))
                wss += self.distance(x, self.centroids[closest_centroid_index])
                clusters[closest_centroid_index].append(x)
            wss = wss/self.X.shape[0]
            # Recalcular los centroides como el promedio de los puntos asignados a cada centroide
            prev_centroids = self.centroids.copy()
            for i, cluster in enumerate(clusters):
                if cluster:
                    self.centroids[i] = np.mean(cluster, axis=0)
            
            # Comprobar si los centroides han dejado de cambiar de posición
            if self.compare_centroids(prev_centroids, self.centroids) or i==self.max_iter-1:
                return self.centroids, clusters, wss
            
            return self.centroids, clusters, wss
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

    def slope(self, p1, p2):
        return (p2[1] - p1[1])/(p2[0] - p1[0])

    def best_k(self, min_k = 1, max_k = 10):
        x_axis = []
        y_axis = []
        max_k += 1 
        for k in range(min_k, max_k):
            x_axis.append(k)
            wss = self.fit(k)[2]
            y_axis.append(wss)
        slopes = {}
        for i in range(1, len(x_axis)-1):
            slopes[i] = abs(self.slope([x_axis[i-1], y_axis[i-1]], [x_axis[i], y_axis[i]])) - abs(self.slope([x_axis[i], y_axis[i]], [x_axis[i+1], y_axis[i+1]]))
        plt.plot(x_axis, y_axis)

        # set the labels and title
        plt.xlabel('k')
        plt.ylabel('WSS')
        plt.title('WSS - k (Codo)')

        # display the graph
        plt.show()
        best_k = max(slopes, key = slopes.get)
        return best_k

    def set_n_clusters(self, k):
        self.n_clusters = k

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
categorical = ['CustGender']
new_cols = pd.get_dummies(data[categorical])
data = pd.concat([data.drop(categorical, axis=1), new_cols], axis = 1)
data = data.drop(["CustLocation"], axis = 1)
data = data.dropna()
print(data)
data = pca.fit_transform(data)

# Ajuste del modelo K-Means
kmeans = KMeans(data, n_clusters=3, max_iter=100)
kmeans.best_k(min_k=1, max_k=15)
print("a")