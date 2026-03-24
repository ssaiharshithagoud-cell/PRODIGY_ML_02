# Task-02: Customer Segmentation using K-Means Clustering
# Prodigy InfoTech ML Internship

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customer_data.csv")

# Display dataset
print("Dataset Preview:")
print(data.head())

# Select features (Annual Income & Spending Score)
X = data[['AnnualIncome', 'SpendingScore']]

# Find optimal clusters using Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means with optimal clusters (example: 5)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to dataset
data['Cluster'] = y_kmeans

# Plot clusters
plt.scatter(X['AnnualIncome'], X['SpendingScore'], c=y_kmeans)
plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
