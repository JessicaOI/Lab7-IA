import numpy as np
from sklearn.cluster import KMeans
from Task1_1 import run, data_explore, graphic


def sklearn_kmeans(X, k):
    # Initialize the KMeans object with the desired number of clusters
    kmeans = KMeans(n_clusters=k)

    # Fit the KMeans model to the data
    kmeans.fit(X)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Print the cluster labels and centroids
    print(labels)
    print(centroids)


def main():
    data = data_explore()
    k = 6
    print("Sklearn")
    sklearn_kmeans(data, k)
    print("Native implementation")
    centroids, clusters, wss = run()
    print(centroids)
    graphic(centroids, clusters, wss)



if __name__ == "__main__":
    main()