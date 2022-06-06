## Generalization

A successful ML model needs to perform well on new and unseen data. In other words, it needs to `generalize` well. In order to test the model generalization, the dataset is splitted into several partitions, `train`, `development (dev)` / `validation (val)`, `test` sets. Train set is used to train the model. Dev set is used to tune the hyperparameters to detect overfitting and quantifies generalizability. Test sets are used to evaluate performance. Dev sets are used so the model never sees the test set for unbiased evaluation. See https://developers.google.com/machine-learning/crash-course/validation/another-partition for more information. This is referred to as `hold-out` validation.

Ideally, distribution of train / dev / test should follow the following properties,

-   Random independent identically distributed (iid)
-   Stationary distribution
-   Same Distribution

Traditionally, the data is `randomly` splitted into approximately 70% for train 10% for dev and 20% for test. Note that for ML problems, the size of dev and test sets need to be big enough for an unbiased evalution of performance. When the data size is big enough, the majority of data can be used for train and leave smaller portions for dev and test sets. For example, a dataset with one million examples 98% can be used for training, and 1% for dev set, 1% for test set still has 10,000 examples.

Typically, the data is randomly shuffled before splitting. However, random splitting is not ideal in certain scenarios. For example, class imbalance, clustered, and time dependent data. See https://developers.google.com/machine-learning/crash-course/18th-century-literature for more details. For randomization, always `seed` the sampling for reproducible results.

When the dataset is small, `k-fold` cross validation is often used to give a better estimate. K-fold cross validation first splits the dataset into k partitions, and trains k models each with one of the partitions left out. The final performance is averaged across all models. `Leave-one-out` cross validation is a special case where k is the sample size. The objective of cross validation is to check whether the model is overfitting to the train set and does not generalizes well.

Generally in ML, the chains of assumption follows,

-   Fit train set well ~ human level performance (bigger network, different alogrithm)
-   Fit dev set well (regularization, bigger train set)
-   Fit test set well (bigger dev set)
-   Real World Data (change dev set or cost function)

Build the first system quickly then iterate.

### Bayes Error

`Bayes Error` is the lowest achievable error and is used to determine whether model is underfitting. `Human Performance` is often used as a surrogate for Bayes Error.

### Variance / Overfitting

`Overfitting` happens when the model capacity is big enough and memorizes the train set. When tested with the dev or test set, the perfomance drops significantly and is therfore not useful. Overfitting is referred to model having `high variance`, with variance being a property of an estimator. It can be described as how much the estimator varies as a function of a data sample.

There are many ways to address overfitting, the most common is `regularization`. Increasing the dataset size can also help the model reach statistical efficiency.

### Bias / Underfitting

Another property of an estimator is the `bias`, it can be described as how much the estimator differs from the truth. When a model has high bias, it means the model is fitting the traing set poorly, also known as `underfitting`.

The most common way to address high bias is to increase the hypothesis space or increase model capacity. This can be done by `adding more features`, or `build bigger models` in neural network. The downside is the risk of overfitting.

The relationship between bias and variance is often seen as a trade-off. But it is more of finding the optimal model capacity so that both the bias and generalization error is low.

Bayes (Human) Error <- `(Avoidable) Bias` -> Train Error <- `Variance` -> Dev Error

To detect overfitting, it is helpful to monitor the training loss and validation loss while training.

### Different Train / Dev Distribution

Due to difficulties in data collection, train and dev set may have different distributions. For example, many pictures maybe collected from the web (200k) but not as many from mobile apps (10k). Random sampling may not work well. A better options is 200k+5k, 2.5k, 2.5k for train / dev / test respectively. Another example is Speech Recognition inside Vehicles. In this scenario, an additional set, `train-dev set`, is useful for analyzing the effect of distribution mismatch. Train-dev set is not used in training.

Bayes (Human) Error <- `(Avoidable) Bias` -> Train Error <- `Variance` -> Dev Error <- `Data Mismatch` -> Dev Error

Sometimes, dev error may be lower than train error when the `dev set is easier` than the train set.
