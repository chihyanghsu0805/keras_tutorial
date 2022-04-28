Discriminative models find the conditional probability of y given x parameterized by θ, P(y | x; θ). The most common model is `Linear Models`.

-   [`Linear Models`](./LinearModels.md)

Linear models are very efficient but sometimes lack the capacity to model difficult problems. `Non-linearity` can be introduced into linear models by `feature crosses` but may be prohibitively cumbersome. Support Vector Machines and Neural Networks add non-linearity by using kernels and activation functions.

-   [`Support Vector Machines (SVMs)`](./SVMs.md)
-   [`Neural Networks (NN)`](./NNs.md)

Besides adding non-linearity, methods that partition the feature space also may provide better performance. `Decision Trees` find the optimal cutoff for each feature to separate the samples for in-group homoneneity. `Gini Impurity` is often used as the loss function to update the cutoffs. Decision Trees can easily overfit by using the same number of leaves and samples. It is regularized by `pruning` and setting the `maximum tree depth and leaves`. Decision Trees suffer from only finding decision boundaries that align with the feature axes.

A common way to boost the performance of weak performers is by `Ensembling`. Generally, there are two ways to ensemble,

-   `Boosting` improves the performance by `sequentially adjusting the importance` of misclassified samples. Common boosting techniques are
    -   Adaptive Boosting (AdaBoost) are used in decision trees to `upweight the decision stumps`
    -   Gradient Boosting Machines (GBM) takes the boosting idea and apply it to gradient descent.
    -   Extreme Gradient Boosting (XGBoost) used a more regularized model formalization to control over-fitting

-   `Bagging` improves the performance by `parallelization`. A common technique is `Random Forest (RF)` that random samples the features in decision trees to build multiple classifiers and combined their prediction for the final prediction.
