import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


# Colors and markers
colors = ['orange', 'brown', 'red', 'green', 'blue', 'black', 'magenta', 'yellow', 'gray', 'crimson']
markers = ['o', 's', 'p', 'P', '*', 'h', 'D', 'd', '+', '_']


def get_clusters_number():
    valid_value = 0
    user_number = 5
    while valid_value == 0:
        user_number = int(input("Set number of clusters (from 2 to 10): "))
        if 1 < user_number < 11:
            valid_value = 1
        else:
            print("\nValue must be between [2;10]")
    return user_number


def show_kmeans_chart(initialize_method, name):
    # Using KMeans
    kmeans = KMeans(n_clusters=n_clusters, init=initialize_method)
    y_predicted = kmeans.fit_predict(transactions[["W0", "Normalized 0"]])
    transactions["cluster"] = y_predicted
    transactions.head()

    # Draw each cluster
    for index in range(n_clusters):
        filtered_items = transactions[transactions.cluster == index]
        plt.scatter(filtered_items["W0"], filtered_items["Normalized 0"], color=colors[index], marker=markers[index],
                    label="Skupienie " + str(index + 1))

    # Draw center points and show legend
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='purple', marker='X',
                label="Centroidy")
    plt.title(name + ' - wykres z podziałem na klastry: %d' % n_clusters)
    plt.legend()
    plt.show()


def show_dbscan_chart():
    # DBSCAN - Create matrix from columns with analyzed data and use DBSCAN
    matrix = np.column_stack((transactions["W0"], transactions["Normalized 0"]))
    model = DBSCAN(eps=0.25, min_samples=400).fit(matrix)
    plt.scatter(matrix[:, 0], matrix[:, 1], c=model.labels_)
    plt.title('DBSCAN - wykres')
    plt.show()


def agglomerative_clustering(transactions):
    transactions = transactions.drop('Product_Code', axis=1)
    X_normalized = normalize(transactions)
    X_normalized = pd.DataFrame(X_normalized)
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    plt.figure(figsize=(8, 8))
    plt.title('Dendogram - wykres')
    shc.dendrogram((shc.linkage(X_principal, method='ward')))
    plt.show()


def scale_week_column():
    # Scale values to 0-1
    scaler = MinMaxScaler()
    scaler.fit(transactions[['W0']])
    transactions['W0'] = scaler.transform(transactions[['W0']])


# Read CSV file
transactions = pd.read_csv("sources/Sales_Transactions_Dataset_Weekly.csv")

# User can set clusters number
n_clusters = get_clusters_number()

scale_week_column()

# KMeans algorithm
show_kmeans_chart('random', "KMeans")

# KMeans++ algorithm
show_kmeans_chart('k-means++', "KMeans++")

# DBScan
show_dbscan_chart()

# Agglomerative Clustering
agglomerative_clustering(transactions)
