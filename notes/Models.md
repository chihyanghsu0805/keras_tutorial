# Model

In supervised learning, models use `parameters` θ and the `examples` x to form the `hypothesis` and make `predictions` h<sub>θ</sub>(x) and compare with the `labels` y. Examples X are often represented in `design matrix` with rows being examples (m) and columns as features (n). Parameters θ represents the weight for each feature. Predictions are therfore h<sub>θ</sub>(x) = X θ. Equations below uses single example for clarity.

A ML model consists of `model`, `loss`, `optimizer`, `hyperparameters`.

-   `Models` can be categorized into `Discriminative` and `Generative` based on their assumptions. Additionally, `parametric` models refer to models with fixed set of parameters while `non-parametric` models have number of parameters that grows with data size.

-   The `Loss` function (L) quantifies the difference between h<sub>θ</sub>(x) and y to provide updates to θ to minimize the difference. The `Cost` function (J) is the loss function for the entire train set. Loss functions can usually be derived from `Maximimum Likelihood Estimation (MLE)` in frequentist statistics and / or `Maximum A Posteriori (MAP)` in bayesian statistics.

-   Optimal parameters θ* an be found by `iterative` methods such as `Gradient Descent` that utilizes the differences between y and h<sub>θ</sub>(x) to update θ. `Parameter Initialization` is key to convergence of the optimization problem. For example, non-convex problems in `Neural Networks` strongly depends on the initialization and often prefers `random sampling from a uniform distribution of small numbers`, where random breaks symmetry and small prevents exploding gradients. On the other hand, `MLE in exponential family is always concave` with respect to η (convex with negative log likelihood) and any initialization would converge.

-   `Hyperparameters` are settings defined by users and not learned. The most important hyperparameter is `learning rate` which directly controls the parameter update rate. Even though, convergence is guaranteed with constant learning rate under `Lipschitz Continuous`, `Learning Rate Decay` is often applied for faster convergence. `Grid Search` and `Random Search` are commonly used to search for hyperparameters.  For both approaches, `appropriate scale` should be used for faster search. For example, learning rate should be searched on an exponential scale instead of linear.

Generally, model selection can be approached either carefully (Panda) or massively (Caviar).

## [Discriminative Models](./Discriminative.md)
## [Generative Models](./Generative.md)
### Non-Parametric Models

The following `non-parametric` models are commonly used for classification, regression and clustering. Both models follow the general principle to compute the `similarity` and observe k nearest neighbors. Similarity directly affects the prediction, therefore `similarity metric` and `feature tranformations` should be carefully designed.

-   `K Nearest Neighbors (KNN)` (classification / regression): For classification the majority label within the k nearest neighbors will be the prediction. For regression, the average of k nearest neighbors will be the prediction.

-   `K Means Clustering`: The algorithm repeats two step,
    -   `Assign Cluster`: Compute the distance to all k centroids and assign a cluster id with the nearest centroid
    -   `Update Cluster Centroid`: Update all centroid coordinates as the average of all points with the same cluster id. Due to `randomly initialization` of cluster centroids, it is common to run K-Means multiple times and choose the best result

A common hyperparamter for both models is the `number of clusters, k`. In KNN, since the problem is supervised, prediction metrics can be used to inform the optimal k. In `clustering`, various techniques can be used to choose the optimal number of clusters,

-   `Visualization`: `Silhouette plots` provides the inter-cluster similarity for each cluster. `K v.s. Total Distance` provides when the infliction point (slope > -1, theta > 135 degree) occurs
-   `Cardinality` is the number of example per cluster
-   `Magnitude` is the sum of distance pre cluster
-   `Performance of Downstream Analysis`
