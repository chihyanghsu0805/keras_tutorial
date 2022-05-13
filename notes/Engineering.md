# Feature Engineering

## Features

### Numerical

-   `Normalization`:
    -   Transforms the distribution for each feature to a similar range
        -   `Scaling`
        -   `Z-score`
    -   For the test set, the means and variance from train set is used
    -   For distribution that follows power law, `log tranformations` can be used
    
-   `Bucketing`:
    -   Group numerical values into categorical when the numerical value do not directly relate (e.g. longitude and latitude, zipcode)
        -   `Uniform`
        -   `Quantile`

### Categorical
    -   Categorical features can be converted into `one-hot` or `multi-hot` vectors
    -   `Sparse` representation can be used instead of dense representations
    -   They can also be represented as `embeddings` with continuous values

### `Feature Crosses`
    -   Feature crosses are used to introduce `non-linear` relationships between features to engineer new features

## Labels

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
