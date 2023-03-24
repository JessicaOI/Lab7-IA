import numpy as np
from sklearn.cluster import KMeans
from Task1_1 import run, data_explore, cluster_graphic


def sklearn_kmeans(X, k):
    # Initialize the KMeans object with the desired number of clusters
    kmeans = KMeans(n_clusters=k)

    # Fit the KMeans model to the data
    kmeans.fit(X)


    centroids = kmeans.cluster_centers_

    print(centroids)


def main():
    data = data_explore()
    k = 6
    print("Sklearn")
    sklearn_kmeans(data, k)
    print("Native implementation")
    centroids, clusters, wss = run(data=data, k=k)
    print(centroids)
    print("La creación de la gráfica puede tomar un largo tiempo (10-20min)")
    cluster_graphic(clusters)



if __name__ == "__main__":
    main()