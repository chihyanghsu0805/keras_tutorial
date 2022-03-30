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

-   Supervised:
    Supervised ML have features and labels. Depending on the labels, it can be further categorized into
    -   Classification, which labels are discrete
    -   Regression, which labels are continuous

-   Unsupervised: Unsupervised ML are used to find patterns in the features. Some examples are,
    -   Clustering
    -   Network Analysis
    -   Market Segmentation

-   Reinforcement Learning (RL): RL providing an environment to learn value and policy functions.

## Data
To build a good solution to a ML problem, it is important to understand your data by exploratory data analysis (EDA).

-   Features

    Features can usually be separated into numerical and categorical features. For each of the feature, look at its distribution and some representative statistics such as minimum, mean, and maximum to identify potential tranformations.

    -   Numerical Features
        -   Scaling
        -   Z-score
        -   Log transform for power law
        -   Bucketing (uniform / quantile) can be used to group numerical values into categorical when the numerical value do not directly relate (e.g. longitude and latitude, zipcode)

        There can also be mistakes in the data, such as outliers, missing data, incorrect values.

        -   Outliers can be clipped
        -   Missing data can be inferred by mean or most frequent value
        -   If too many observations are missing the feature, the feature can be discarded
        -   Check for incorrect entries such as typos

    -   Categorial Features
        -   Categorical features can be converted into one-hot or multi-hot vectors
        -   Sparse representation can be used instead of dense representations
        -   They can also be represented as embeddings with continuous values

    Additionally, check the correlation among all the features to identify highly-correlated feature, which can be removed.

    Also, check the correlation between features and labels to gain a early understanding of the predictability of the features.

    Note that correlation only finds linear relationships, mutual information can be used to find more complex relationships.

    Nevertheless, correlation is only applicable to univariate analysis. For multi-variate, feature selection techniques can be used later in the pipeline.

-   Labels

    For labels, check for outliers, missing values, typos. Additionally check the distribution to identify class imbalances.

    -   Class Imbalance:

        To overcome class imbalance, downsampling and upsampling techniques can be used.

        -   Downsampling: downsample the majority class but upweight the samples. https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data
        -   Upsampling: upsample the minority class by bootstrapping
        -   Additionally, for image segmentation, biased random crop can be used. https://developer.nvidia.com/blog/accelerating-medical-image-processing-with-dali/

        Additionally, one can use weighted loss function and other metrics (precision and recall) to evaluate the model with class imbalanced data.

    Labels can also be grouped to change the ML problem.

- Data Centric

## Generalization

-   Data Splits:

    -   Random independent identically distributed (iid)
    -   Stationary distribution
    -   Same Distribution

    -   Traditional
    -   Big Data
    -   Random or Stratified
    -   Cross Validation

-   Bias / Underfitting

-   Variance / Overfitting

-   Loss:

    Empirical Risk Minimization
    Strucural Risk Minimization

-   Regularization

    -   L1 / LASSO pushes weights to zero to encourage sparsity
    -   L2 / ridge pushes weights near zero to encourage stability
    -   Early Stopping
    -   Feature Selection
    -   C + L1 (SVM)
    -   Pruning, Depth and Number of Trees (RF)
    -   Dropout (NN)

-   Metrics

    -   Accuracy
    -   Precision
    -   Recall
    -   Receiver Operating Characteristic (ROC) Curve
    -   Area Under the Curve (AUC)
    -   Calibration Plot
    -   Bucketed Bias

## Model
-   Discriminative
    -   Linear Regression (Exponential family)
    -   Locally Weighted Linear Regression (Non-Parametric)
    -   Logistic Regression (Exponential family)
    -   Softmax Regression
    -   Generalized Linear Models (Exponential family)
    -   Support Vector Machines (SVM)
    -   Neural Networks (NN)
    -   Decision Trees
    -   Ensembles
        -   Boosting: Adaptive Boosting (AdaBoost), Gradient Boosting Machines (GBM), Extreme Gradient Boosting (XGBoost)
        -   Bagging: Random Forest (RF)

    -   K Nearest Neighbors (Non-Parametric)
    -   K Means (Clustering, Non-Parametric)

-   Generative
    -   Gaussian Discriminant Analysis
    -   Naive Bayes

-   Estimators:
    -   Maximimum Likelihood Estimation (MLE, frequentist):
    -   Maximum A Posteriori (MAP, bayesian):

-   Optimizers:

    -   Iterative Methods
        -   Gradient Descent
            -   Batch Gradient Descent
            -   Mini-Batch Gradient Descent
            -   Stochastic Gradient Descent

        -   Momentum
        -   Adaptive Learning Rate
        -   Momentum + Adaptive Learning Rate
        -   Newton-Raphson

    -   Direct / Analytical Methods
        -   Normal Equation
