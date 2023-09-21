import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\User\Downloads\Iris.csv")
X = df["SepalLengthCm"]
Y = df["SepalWidthCm"]
plt.scatter(X, Y, color="red")
plt.xlabel("Sepal Length(cm)--->")
plt.ylabel("Sepal Width(cm)--->")
plt.title("Sepal Length vs Sepal Width")
plt.show()
# Finding Optimum value for the number of clusters.
x = df.iloc[:, [0, 1, 2, 3]].values
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=600, n_init=10, random_state=3)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 16), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.show()
# So we can see that the optimum value of clusters must be 3.
# Now using KMeans to classify the clusters.
x = df.iloc[:, [0, 1, 2, 3]].values
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=600, n_init=10, random_state=3)
y_kmeans = kmeans.fit_predict(x)
# Plotting the clusters.
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c="red", label="Iris-Setosa")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c="green", label="Iris-Versicolor")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c="blue", label="Iris-Virginia")
plt.title("IRIS-CLUSTERS")
plt.legend(loc='best')
plt.show()
