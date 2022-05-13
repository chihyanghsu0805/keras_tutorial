# Feature Engineering

-   `Features`: Features can usually be separated into numerical and categorical features

    -   `Numerical` Features

        -   `Normalization`: Normalization transforms the distribution for each feature to a similar range. For the test set, the means and variance from train set is used.
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
