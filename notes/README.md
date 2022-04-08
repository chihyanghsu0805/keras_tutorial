This folder contains notes from various Machine Learning (ML) / Deep Learning (DL) courses.

Including,

-   Google Machine Learning Crash Course, https://developers.google.com/machine-learning/crash-course
-   Stanford CS229 - 2018 Fall, https://youtu.be/jGwO_UgTS7I, https://cs229.stanford.edu/syllabus-autumn2018.html
-   Coursera Deep Learning Specialization, https://www.coursera.org/specializations/deep-learning
-   Coursera Stanford Machine Learning, https://www.coursera.org/learn/machine-learning?page=2
-   Deep Learning, https://www.deeplearningbook.org/

# Overview

## Problem
ML problems can be catgorized into the following,

-   `Supervised` ML have features and labels. Depending on the labels, it can be further categorized into

    -   `Classification` with discrete labels
    -   `Regression` with continuous labels

-   `Unsupervised` ML are used to find patterns in the data. Some examples are,

    -   `Clustering`
    -   Network Analysis
    -   Market Segmentation

-   `Reinforcement Learning` (RL): RL provides an environment to learn value and policy functions.

## Data

Typically, there are two types of data, `structured` and `unstructured`. Structured data refers to data that is represented with rows as subjects and columns as features. Unstructured data refers to data that do not follow this format, such as images and audios.

To build a good solution to a ML problem, it is important to understand the data, or `exploratory data analysis (EDA)`. In EDA, it is typical to collect representative statistics (such as mean, minimum, maximum, etc) for each feature and detect outliers and errors. Histogram visualization also helps with understanding the feature distribution to determine possible means to normalize the features. After understanding the data, various approaches can be applied to the features to help with model training. For example, normalizing the features allows certain algorithms to converge faster. These approaches are referred to as `feature engineering`.

### Feature Engineering

-   `Features`: Features can usually be separated into numerical and categorical features

    -   `Numerical` Features

        -   `Normalization`: Normalization transforms the distribution for each feature to a similar range
            -   `Scaling`
            -   `Z-score`
        -   For distribution that follows power law, `log tranformations` can be used
        -   `Bucketing` can be used to group numerical values into categorical when the numerical value do not directly relate (e.g. longitude and latitude, zipcode)
            -   `Uniform`
            -   `Quantile`

    -   `Categorial` Features

        -   Categorical features can be converted into `one-hot` or `multi-hot` vectors
        -   `Sparse` representation can be used instead of dense representations
        -   They can also be represented as `embeddings` with continuous values

    -   `Feature Crosses`

        Feature crosses are used to introduce `non-linear` relationships between features to engineer new features.

- `Labels`

    For supervised learning, labels should also be inspected and reviewed. Labels can be categorized into `direct` or `derived` based on their source and ML problem definition. An example for direct labels is movie rating and an example for derived label is users watched movies. See https://developers.google.com/machine-learning/data-prep/construct/collect/label-sources for more information.

    For labels, check the distribution to identify `class imbalances`.

    -   Class Imbalance:

        To overcome class imbalance, downsampling and upsampling techniques can be used

        -   `Downsampling`: downsample the majority class but upweight the samples
        -   https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data
        -   `Upsampling`: upsample the minority class by bootstrapping
        -   Additionally, for image segmentation, biased random crop can be used
        -   https://developer.nvidia.com/blog/accelerating-medical-image-processing-with-dali/

        Additionally, one can use weighted loss function and other metrics (`precision` and `recall`) to evaluate the model with class imbalanced data.

        Labels can also be grouped to alleviate class imbalance and change the ML problem.

-   `Outliers`

    For both features and labels, check for outliers, missing values, and other mistakes.

    -   Feature outliers can be `clipped`
    -   Missing data can be `imputed` by mean or most frequent value
    -   If too many observations are missing the feature, the feature can be `discarded`
    -   Check for incorrect entries such as typos
    -   For labels, check whether all the entries are in `expected range`
    -   Check every case for label correctness if possible, otherwise spot check
    -   For missing label, ask for annotation or discard the observation

-   `Correlations`

    Additionally, for supervised learning, check the correlation between features and labels. Identify feature highly correlated with labels and use them to establish an early baseline performance.

    -   Check the correlation among all the features to identify highly-correlated feature, which can be removed
    -   Check the correlation between features and labels to gain a early understanding of the predictability of the features
    -   Note that correlation only finds linear relationships, `mutual information` can be used to find more complex relationships
    -   Correlation is only applicable to univariate analysis. For multi-variate, `feature selection` techniques can be used later in the pipeline

<!---  Data Centric --->

## Generalization

A successful ML model needs to perform well on new and unseen data. In other words, it needs to `generalize` well. In order to test the model generalization, the dataset is splitted into several partitions, `train`, `development (dev)` / `validation (val)`, `test` sets. Train set is used to train the model. Dev set is used to tune the hyperparameters to detect overfitting and quantifies generalizability. Test sets are used to evaluate performance. Dev sets are used so the model never sees the test set for unbiased evaluation. See https://developers.google.com/machine-learning/crash-course/validation/another-partition for more information. This is referred to as `hold-out` validation.

Ideally, distribution of train / dev / test should follow the following properties,

-   Random independent identically distributed (iid)
-   Stationary distribution
-   Same Distribution

Traditionally, the data is `randomly` splitted into approximately 70% for train 10% for dev and 20% for test. Note that for ML problems, the size of dev and test sets need to be big enough for an unbiased evalution of performance. When the data size is big enough, the majority of data can be used for train and leave smaller portions for dev and test sets. For example, a dataset with one million examples 98% can be used for training, and 1% for dev set, 1% for test set still has 10,000 examples.

Typically, the data is randomly shuffled before splitting. However, random splitting is not ideal in certain scenarios. For example, class imbalance, clustered, and time dependent data. See https://developers.google.com/machine-learning/crash-course/18th-century-literature for more details. For randomization, always `seed` the sampling for reproducible results.

When the dataset is small, `k-fold` cross validation is often used to give a better estimate. K-fold cross validation first splits the dataset into k partitions, and trains k models each with one of the partitions left out. The final performance is averaged across all models. `Leave-one-out` cross validation is a special case where k is the sample size.

The objective of cross validation is to check whether the model is overfitting to the train set and does not generalizes well.

### Variance / Overfitting

`Overfitting` happens when the model capacity is big enough and memorizes the train set. When tested with the dev or test set, the perfomance drops significantly and is therfore not useful. Overfitting is referred to model having `high variance`, with variance being a property of an estimator. It can be described as how much the estimator varies as a function of a data sample.

There are many ways to address overfitting, the most common is `regularization`. Increasing the dataset size can also help the model reach statistical efficiency.

### Bias / Underfitting

Another property of an estimator is the `bias`, it can be described as how much the estimator differs from the truth. When a model has high bias, it means the model is fitting the traing set poorly, also known as `underfitting`.

The most common way to address high bias is to increase the hypothesis space or increase model capacity. This can be done by `adding more features`, or `build bigger models` in neural network. The downside is the risk of overfitting.

The relationship between bias and variance is often seen as a trade-off. But it is more of finding the optimal model capacity so that both the bias and generalization error is low.

To detect overfitting, it is helpful to monitor the training loss and validation loss while training.

### Regularization

A ML model find the optimal parameters by minimizing the prediction and the provided labels. This is referred to as
`Empirical Risk Minimization`. A common way to address overfitting is to add regularization terms to the optimization problem, or `Structural Risk Minimization`.

Two regularization terms are commonly used,

-   `L1` (LASSO) pushes weights to zero to encourage `sparsity`
-   `L2` (ridge) pushes weights near zero to encourage `stability`, also known as `weight decay`
-   Elastic nets combines both L1 and L2
-   L1 and L2 are also used for feature selection

Besides regularization, other techniques can help detect and reduce overfitting.

-   Early Stopping: which the user defines criteria for stopping the model training
-   SVM utilizes the C coefficient along with L1 / L2
-   Decision Tress can be regularized by `pruning`, tree `depth` and number of trees
-   Neural networks can use `Dropout` layers

### Metrics

Metrics are used to compare model performance. It is important to choose a metric that truly represents the objective of the model. Choose a single metric to evaluate but others can also be collected. For classification problems, it is typical to use `accuracy: (TP + TN) / (TP + FP + TN + FN)`. For class imbalanced problems, `precision: TP / (TP + FP)` and `recall: TP / (TP + FN)` may be more informative. Increasing the threshold leads to less False Positives and result in higher precision, while decreasing the threshold leads to less False Negatives with higher Recall. `F1 / Dice Score: 2 x P x R / (P + R)` is the harmonic mean of precision and recall. It is also commonly used in image segmentation along with `Hausdorff Distance`. `Jaccard Index / Intersection over Union (IoU)` is used for object detection and categorical features.

Besides single value metrics, `Receiver Operating Characteristic (ROC) Curve` assess model performance at different threshold values and `Area Under the Curve (AUC)` quantifies the curve. `Calibration Plot` visualizes the performance with different labels to detect `Bucketed Bias`.

## Model

In supervised learning, models use `parameters` θ and the `examples` x to form the `hypothesis` and make `predictions` h<sub>θ</sub>(x) and compare with the `labels` y. Examples X are often represented in `design matrix` with rows being examples (m) and columns as features (n). Parameters θ represents the weight for each feature. Predictions are therfore h<sub>θ</sub>(x) = X θ. Equations below uses single example for clarity.

A ML model consists of `model`, `loss`, `optimizer`, `hyperparameters`.

-   `Models` can be categorized into `Discriminative` and `Generative` based on their assumptions. Additionally, `parametric` models refer to models with fixed set of parameters while `non-parametric` models have number of parameters that grows with data size.

-   The `Loss` function (L) quantifies the difference between h<sub>θ</sub>(x) and y to provide updates to θ to minimize the difference. The `Cost` function (J) is the loss function for the entire train set. Loss functions can usually be derived from `Maximimum Likelihood Estimation (MLE)` in frequentist statistics and / or `Maximum A Posteriori (MAP)` in bayesian statistics.

-   Optimal parameters θ* an be found by `iterative` methods such as `Gradient Descent` that utilizes the differences between y and h<sub>θ</sub>(x) to update θ. `Parameter Initialization` is key to convergence of the optimization problem. For example, non-convex problems in `Neural Networks` strongly depends on the initialization and often prefers `random sampling from a uniform distribution of small numbers`, where random breaks symmetry and small prevents exploding gradients. On the other hand, `MLE in exponential family is always concave` with respect to η (convex with negative log likelihood) and any initialization would converge.

-   `Hyperparameters` are settings defined by users and not learned. The most important hyperparameter is `learning rate` which directly controls the parameter update rate.

### Discriminative Models

Discriminative models find the conditional probability of y given x parameterized by θ, P(y | x; θ).

-   `Linear Regression` (Exponential family) follows the form h<sub>θ</sub>(x) = θ<sup>T</sup>x with x<sub>0</sub> being 1 and θ<sub>0</sub> being the bias term. It commonly uses `Minimum Squared Error` (MSE) as the Loss term, (y -h<sub>θ</sub>(x))<sup>2</sup> which can be derived from `Maximum Likelihood Estimation (MLE)` with `Gaussian` distribution. Besides iterative methods, Linear Regression can also be solved using analytical methods such as  Normal equation to find θ* in one step with θ* = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y, given that (X<sup>T</sup>X) is invertible. Note that matrix inverses are expensive operation and prone to numerical errors.

-   `Locally Weighted Linear Regression` (Non-Parametric) weighs the individual loss by non-negative function w(i) based on their distance to the point of prediction.

-   `Logistic Regression` (Exponential family) can be used to predict probabilities and is often used for `binary classification` with probability threshold. It follows the form h<sub>θ</sub>(x) = σ(z) with σ as the `sigmoid/logistic` function, σ (z) = 1/(1+e<sup>-z</sup>) and z = θ<sup>T</sup>x. The probability resembles the `Bernoulli` distribution, P(y = x; θ) = h<sub>θ</sub>(x) <sup>y</sup> (1 - h<sub>θ</sub>(x)) <sup> (1-y) </sup>. `Regularization` is `extremely important` for Logistic Regression as the sigmoid function reaches asymptotic when z approaches +/- infinity. Without regularization, the parameters may explode and/or vanish. The loss function is the product of all examples Π P(y = x; θ) and is usually applied by a logarithmic to become `log likelihood` which becomes sum of all examples, Σ P(y = x; θ) = 	Σ y log(h<sub>θ</sub>(x)) + (1-y) log((1 - h<sub>θ</sub>(x))). This loss is also know as `cross entropy loss (XEnt)`. `Maximum Likelihood Estimation` maximizes the log likelihood with gradient ascent, but is usually transformed to `minimizing negative log likelihood` with `gradient descent`.

-   `Softmax Regression` is logistic regression with `multi-class` classification. The logits (z, θ<sup>T</sup>x) are exponentialized and normalized so the probabilities sums to 1, (e<sup>θ<sub>i</sub><sup>T</sup>x</sup> / 	Σ<sub>j</sub> e<sup>θ<sub>j</sub><sup>T</sup>x</sup>). The loss function become `categorical cross entropy`. For problems with `multi-class single instance`, softmax is used and `multi-class multi-instance`, logistic regression is used for each instance.

-   `Exponential Family` is a family of models that can be factored into P(y; η) = b(y) exp(η<sup>T</sup>T(y)-α(η)), where α is log partition. Examples are `Gaussian`, `Bernoulli`, Poisson, Gamma, Exponential, Beta, Dirichlet. MLE of exponentation is always concave, so covergence is guaranteed with `random initialization`. `Generalized Linear Models (GLMs)` are a set of models that follows the properties:
    -   y | x; θ ~ Exponential Family
    -   η = θ<sup>T</sup>x  (linear)
    -   At train time, maximizes log P(y; η). At test time, outputs E[y; η] = h<sub>θ</sub>(x).
    -   Learning update rule: θ<sub>j</sub> = θ<sub>j</sub> + α (y<sup>i</sup> - h<sub>θ</sub>(x<sup>i</sup>)) x<sup>i</sup><sub>j</sub>, where α is learning rate.

Linear models are very efficient but sometimes lack the capacity to model difficult problems. `Non-linearity` can be introduced into linear models by `feature crosses` but may be prohibitively cumbersome. Support Vector Machines and Neural Networks add non-linearity by using kernels and activation functions.

-   `Support Vector Machines (SVMs)`: SVMs are `optimal margin classifiers` with `kernel tricks`.

    -   Optimal Margin Classifiers: For binary classification problems, the labels y become [-1, 1] and θ is represented by w and b. The hypothesis becomes h<sub>w,b</sub>(x) = g(w<sup>T</sup>x+b).

        -   The `Functional Margin` (FM) is defined as Γ<sup>i</sup><sub>FM</sub> = y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b). If y = 1, w<sup>T</sup>x<sup>i</sup>+b >> 0 and if y = -1, w<sup>T</sup>x<sup>i</sup>+b << 0. For the entire train set, Γ<sub>FM</sub> = min Γ<sup>i</sup><sub>FM</sub>. Note that FM is not a good metric of confidence since its magnitude is directly affected by w and b.

        -   The `Geometric Margin` (GM) is defined as Γ<sup>i</sup><sub>GM</sub> = y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) / ||w|| and Γ<sub>GM</sub> = min Γ<sup>i</sup><sub>GM</sub>.

        -   The optimal margin classifier seeks to max<sub>Γ, w, b</sub> Γ<sub>GM</sub> subject to y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) / ||w|| >= Γ<sub>GM</sub> and ||w|| = 1. It can be written as min<sub>w, b</sub> (1/2) *||w||<sup>2</sup> subject to y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) >= 1 when FM is scaled to be 1 or ||w|| = 1 / Γ<sub>GM</sub>. This optimization can be solved with `Quadratic Programming`.

    -   Kernel Tricks: The parameters w can been seen as adding some multiples of x, so the optimization problem can be written as min<sub>w, b</sub> (1/2) * (Σ<sub>i</sub>a<sub>i</sub>x<sup>i</sup>y<sup>i</sup>)<sup>T</sup>(Σ<sub>j</sub>a<sub>j</sub>x<sup>j</sup>y<sup>j</sup>) subject to y<sup>i</sup>(w<sup>T</sup>x<sup>i</sup>+b) >= 1. Expanding the matrix multiplication, min<sub>w, b</sub> (1/2) * (Σ<sub>i</sub>Σ<sub>j</sub>a<sub>i</sub>a<sub>j</sub>y<sup>i</sup>y<sup>j</sup>x<sup>i</sup><sup>T</sup>x<sup>j</sup>) where x<sup>i</sup><sup>T</sup>x<sup>j</sup> is the inner product <x<sup>i</sup>, x<sup>j</sup>>, or kernel K(x, z). Common kernels are,

        -   `Linear Kernel` K(x, z) = x<sup>T</sup>z
        -   `Gaussian Kernel` K(x, z) = exp(-||x-z||<sup>2</sup> / (2σ<sup>2</sup>))
        -   `Polynomial Kernel` K(x, z) = (x<sup>T</sup>z)<sup>d</sup>

    -   Regularization: For non-linear separable cases, `L1-norm soft margin` can be used with the C parameter controlling the functional margin error.

-   `Neural Networks (NN)`: NNs consists of layers which is a collection of units / neurons. Each unit represents a parameter, and the connections between units are the weights. NNs with multi-llayers are also known as `Multilayer Perceptron (MLP)`. NNs introduce non-linearity by applying `non-linear activation` to the units. If all layers have linear activation, then it is the same as one layer with linear activation. Common activation functions are,

    -   Rectified Linear Unit (ReLU): g(x) = max(0, x)
    -   Sigmoid: g(x) = 1 / (1 + e<sup>-x</sup>)
    -   tanh: g(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)

    NNs increase model capacity by increasing number of layers (depth) and number of units (width). Typically, the input layer is visualized as the bottom layer and output layer on top. In-between layers are called `hidden layers`. NNs with multiple hidden layers are referred to as Deep Neural Networks (DNN). Layers with all units connected are know as `fully connected / dense` layers.

    One key challenge of DNNs is the computation of parameter updates which is addressed by `backpropagation`. A `forward propagation` gives the prediction while the backpropagation computes the parameter updates using `chain rule`. The parameters are updated using iterative methods such as gradient descent. Due to the use of chain rule, activation functions are usually `differentiable` (sigmoid and tanh) or can be easily appoximated (ReLU).

    Another challenge of DNNS is the vanishing / exploding gradients due to the many layers. This can be mitigated with `skip connections`. `Gradient clipping` also helps with exploding gradients. `Batch Normalization` also provides some help with vanishing / exploding gradients as well as `internal covariate shift` due to random initialization. However, evidences show that Batch Normalization may induce severe gradient explosion at initialization.

    -   `Initialization` is extremely import for NNs. The initial parameters need to `break symmetry` to ensure each unit learns different functions. Typically biases are set to constants and weights are randomly sampled form a Gaussian or Uniform distribution. The `scale of the distribution` has a large impact as well. If the initial values are too big, gradients may explode. Likewise initial values too small, gradients may vanish. Below are some common initializations,

        -   (Xavier) Glorot and Bengio Uniform: U[-1/sqrt(n), 1/sqrt(n)]
        -   Xavier Uniform: U[-sqrt(6)/sqrt(m+n), sqrt(6)/sqrt(m+n)], designed for equal variance
        -   Xavier Normal: N(0, sqrt(2/(m + n)))
        -   He for ReLU:  N(0, sqrt(2/n))

    Below are some specialized networks,

    -   `Convolution Neural Networks (CNNs)` are used for many computer vision tasks, such as object detection and segmentation. Convolution is an very important operation in signal processing and is basically a sliding window approach. In practice, `autocorrelation` is used but still refered as `convolution`. Instead of fully connected layers, CNNs are built with convolution layers that the sliding window is defined by convolution `kernels`. Due to the kernels, the outputs becomes smaller than the input. Additionally, `stride` also makes the outputs smaller. `Padding` can be used to mitigate the downsizing of output size. The output size can be computed from `O = floor[(I + 2p - f) / s)] + 1`. Another special layer in CNNs is `pooling`, which `average pooling` takes the average in th window and `max pooling` takes the max in the window. Convolution and Pooling can be seen as `Infinitely Strong Priors` that some unit connections are forbidden and neighboring units should have equal weights. CNNs have three desirable properties, `sparse interactions`, `parameter sharing` and `equivariant representations`.

    -   `Recurrent Neural Networks (RNNs)` are used for sequential data, such as text and video. Besides connection between layers, units are also connected across time / sequence. Therefore, parameters are updated via `backpropagation through time`. Connections can be `causal` where only information from the past is used. It can also be `bidirectional` where the entire sequence is being used. `Long-Term Dependencies` is a big challenge in sequence modeling due to vanishing / exploding gradients across time and space. Besides `skip connections`, `Gated RNNs`, such as `Gated Recurrent Units (GRUs)` and `Long Short Term Memory (LSTM)` are designed to address long-term dependencies by introducing `gates`. In particular, LSTMs uses the `cell` state to carry the information across the sequence. It is governed by three gates, `input`, `forget` and `cell / memory`. The forget gate controls how much to forget from the previous cell state. The input and cell gate control how much to update the cell state to pass to the next sequence. An additional `output` gate combines the updated cell state and the input state feed into the next unit. The input and previous output is concatenated for the gates.

    -   Regularization: NNs have a special regularization technique `Dropout` which randomly drops units during training. It has the effect of evenly distributing the weights throughout the network and not rely on specific units. At test time, dropout is turned off.

    <!---  It is common to increase depth rather than width since it takes exponentially more units to represent the same function [REF]. --->

Besides adding non-linearity, methods that partition the feature space also may provide better performance. `Decision Trees` find the optimal cutoff for each feature to separate the samples for in-group homoneneity. `Gini Impurity` is often used as the loss function to update the cutoffs. Decision Trees can easily overfit by using the same number of leaves and samples. It is regularized by `pruning` and setting the `maximum tree depth and leaves`. Decision Trees suffer from only finding decision boundaries that align with the feature axes. A common way to boost the performance of weak performers is by `Ensembling`. Generally, there are two ways to ensemble,

-   `Boosting` improves the performance by `sequentially adjusting the importance` of misclassified samples. Common boosting techniques are
    -   Adaptive Boosting (AdaBoost) are used in decision trees to `upweight the decision stumps`
    -   Gradient Boosting Machines (GBM) takes the boosting idea and apply it to gradient descent.
    -   Extreme Gradient Boosting (XGBoost) used a more regularized model formalization to control over-fitting

-   `Bagging` improves the performance by `parallelization`. A common technique is `Random Forest (RF)` that random samples the features in decision trees to build multiple classifiers and combined their prediction for the final prediction.

The following `non-parametric` models are commonly used for classification, regression and clustering. Both models follow the general principle to compute the `similarity` and observe k nearest neighbors. Similarity directly affects the prediction, therefore `similarity metric` and `feature tranformations` should be carefully designed.

-   `K Nearest Neighbors (KNN)` (classification / regression): For classification the majority label within the k nearest neighbors will be the prediction. For regression, the average of k nearest neighbors will be the prediction.

-   `K Means Clustering`: The algorithm repeats two step,
    -   `Assign Cluster`: Compute the distance to all k centroids and assign a cluster id with the nearest centroid
    -   `Update Cluster Centroid`: Update all centroid coordinates as the average of all points with the same cluster id. Due to `randomly initialization` of cluster centroids, it is common to run K-Means multiple times and choose the best result

A common hyperparamter for both models is the `number of clusters, k`. In KNN, since the problem is supervised, prediction metrics can be used to inform the optimal k. In `clustering`, various techniques can be used to choose the optimal number of clusters,

-   `Visualization`: `Silhouette plots` provides the inter-cluster similarity for each cluster. `K v.s. Total Distance` provides when the infliction point (slope > -1, theta > 135 degree) occurs
-   `Cardinality` is the number of example per cluster
-   `Magnitude` is the sum of ditance pre cluster
-   `Performance of Downstream Analysis`

### Generative Models

Generative models find the joint probability of y and x P(y, x). From the definition of `conditional probability`, P(x | y) = P(x, y) P(y) and P(y | x) = P(x, y) P(x). Therefore, it can also be seen as modeling P(x | y) P(y) with P(y) as a prior.

-   `Gaussian Discriminant Analysis (GDA)`: Using binary classification as an example to compare with logistic regression, GDA assumes x | y ~ N(μ, Σ) with different μ for y = 0 and 1 but identical Σ (Different Σ results in nonlinear decision boundary) and y is Bernoulli φ. It optimizes for φ, μ<sub>0</sub>, μ<sub>1</sub> and Σ. Under MLE, φ is the ratio of positive samples to all, μ is the average of all postive / negative samples, and Σ is computed with the estimated means. Due to the `stronger assumption`, GDA requires less data but only works when the data is Gaussian. Whereas Logistic Regression has weaker assumption and works with any distribution.

-   `Naive Bayes` is another popular generative model, especially in sequence models. It asuumes Xs are `conditionally independent` given y, P(x<sub>1</sub>, ..., x<sub>n</sub> | y)  = P(x<sub>1</sub> | y) P(x<sub>2</sub> | x<sub>1</sub>, y) P(x<sub>3</sub> | x<sub>2</sub> x<sub>1</sub>, y)... = P(x<sub>1</sub> | y) P(x<sub>2</sub> | y) ... P(x<sub>n</sub> |y). `Laplace Smoothing` is used to avoid zero probabilities.

-   Generative Adversarial Networks (GANs)
-   Autoencoder (AE)

### Optimizers

Optimizers control how the parameters are updated and can be separated into `iterative` and `analytical` methods.
Analytical methods take the optimization problem and derives a fomula and θ* can be found by plugging the variables. An example of analytical methods is `Normal Equation` for linear regression.

Not all problems have an analytical solution, and `iterative methods` approaches θ* by incremental updates. `Gradient Descent` is a very common iterative method with many variations. The parameter update generically follows the rule: θ<sub>n</sub> = θ<sub>o</sub> - α dL/dθ, with α as learning rate and the parameter update follows the `negative` gradient direction. Depending on the batch size, gradient descent can be classified as,

-   `Batch Gradient Descent`: where batch size is the entire dataset (m)
-   `Stochastic Gradient Descent`: where batch size is 1
-   `Mini-Batch Gradient Descent`: where batch size is between 1 and m

Batch gradient descent suffers from heavy computation and longer updates but more smooth learning curve. Batch gradient descent is used when the entire dataset can be fitted into memory. Stochastic gradient descent updates frequently but very noisy and never reaches convergence. Stochastic gradient descent is used when one sample is very big, such as medical images. In practice, Mini-Batch gradient descent is often used.

 `Feature Normalization` is very important for accelerating the convergence of iterative methods. Variations of gradient descent also speeds up the convergence. See https://ruder.io/optimizing-gradient-descent/index.html for more details.

-   `Momentum`: looks at previous updates to best inform the next update, it can be seen as a moving average
-   `Adaptive Learning Rate (AdaGrad / AdaDelta / RMSprop)`: modifies the learning rate for each parameter based on previous updates
-   `Momentum + Adaptive Learning Rate (Adam)`: Combines mementum and adaptive learning rate.

Gradient descent and its variations are considered `first order` methods since the update rule is linear. `Second order` methods, such as `Newton-Raphson` uses the `Hessian Matrix` or second order derivative and benefits from `quadratic convergence`.

-   `Newton-Raphson`: θ<sub>n</sub> = θ<sub>o</sub> - (dJ/dθ) / (dJ/dθ<sup>2</sup>) or θ<sub>n</sub> = θ<sub>o</sub> - H<sup>-1</sup> (dJ/dθ)
