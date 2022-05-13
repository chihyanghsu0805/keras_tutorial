### Regularization

A ML model find the optimal parameters by minimizing the prediction and the provided labels. This is referred to as
`Empirical Risk Minimization`. A common way to address overfitting is to add regularization terms to the optimization problem, or `Structural Risk Minimization`. Regularization prevents overfitting by forcing the parameters to be small hence limits the `parameter ranges to be mostly linear`. Regularization can be seen as having a preference over certain models.

` Statistical Learning Theory`

Two regularization terms are commonly used,

-   `L1` (LASSO) pushes weights to zero to encourage `sparsity`, where the regularization term is the sum of absolute values of the weights
-   `L2` (ridge) pushes weights near zero to encourage `stability`, also known as `weight decay`, where the regularization term is the sum of squared values of the weights
-   Elastic nets combines both L1 and L2
-   L1 and L2 are also used for feature selection

Besides regularization, other techniques can help detect and reduce overfitting.

-   Data Augmentation: applying some tranformations (flipping, rotation, color shift) to the input allows the model to learn from a larger sample size. `Statistic Consistency` states that with infinite sample size, the variance should be 0. `RandAugment` remove the need for a separate search phase on a proxy task. [1]
-   Early Stopping: which the user defines criteria for stopping the model training
-   SVM utilizes the C coefficient along with L1 / L2
-   Decision Tress can be regularized by `pruning`, tree `depth` and number of trees
-   Neural networks can use `Dropout` layers

[1] Cubuk, E.D., Zoph, B., Shlens, J. and Le, Q.V., 2020. Randaugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 702-703).
