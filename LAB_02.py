import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Colors and markers
colors = ['orange', 'brown', 'red', 'green', 'blue', 'black', 'magenta', 'yellow', 'gray', 'crimson']
markers = ['o', 's', 'p', 'P', '*', 'h', 'D', 'd', '+', '_']

# Read CSV file
transactions = pd.read_csv("sources/Sales_Transactions_Dataset_Weekly.csv")

# Scale values to 0-1
scaler = MinMaxScaler()
scaler.fit(transactions[['W0']])
transactions['W0'] = scaler.transform(transactions[['W0']])

# User can set clusters and week number
valid_value = 0
n_clusters = 5
while valid_value == 0:
    n_clusters = int(input("Set number of clusters (from 2 to 10): "))
    if 1 < n_clusters < 11:
        valid_value = 1
    else:
        print("\nValue must be between [2;10]")

# Using KMeans
kmeans = KMeans(n_clusters=n_clusters)
y_predicted = kmeans.fit_predict(transactions[["W0", "Normalized 0"]])
transactions["cluster"] = y_predicted
transactions.head()

# Draw each cluster
for index in range(n_clusters):
    filtered_items = transactions[transactions.cluster == index]
    plt.scatter(filtered_items["W0"], filtered_items["Normalized 0"], color=colors[index], marker=markers[index],
                label="Skupienie " + str(index + 1))

# Draw center points and show legend
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='purple', marker='X', label="Centroidy")
plt.legend()
plt.show()