# Non-Parametric Models

The following `non-parametric` models are commonly used for classification, regression and clustering. Both models follow the general principle to compute the `similarity` and observe k nearest neighbors. Similarity directly affects the prediction, therefore `similarity metric` and `feature tranformations` should be carefully designed.

-   `K Nearest Neighbors (KNN)` (classification / regression): For classification the majority label within the k nearest neighbors will be the prediction. For regression, the average of k nearest neighbors will be the prediction.

-   `K Means Clustering`: The algorithm repeats two step,
    -   `Assign Cluster`: Compute the distance to all k centroids and assign a cluster id with the nearest centroid
    -   `Update Cluster Centroid`: Update all centroid coordinates as the average of all points with the same cluster id. Due to `randomly initialization` of cluster centroids, it is common to run K-Means multiple times and choose the best result

A common hyperparameter for both models is the `number of clusters, k`. In KNN, since the problem is supervised, prediction metrics can be used to inform the optimal k. In `clustering`, various techniques can be used to choose the optimal number of clusters,

-   `Visualization`: `Silhouette plots` provides the inter-cluster similarity for each cluster. `K v.s. Total Distance` provides when the infliction point (slope > -1, theta > 135 degree) occurs
-   `Cardinality` is the number of example per cluster
-   `Magnitude` is the sum of distance pre cluster
-   `Performance of Downstream Analysis`
