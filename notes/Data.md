#   Data

## Data Types

-   `Structured` data refers to data that is represented with rows as subjects and columns as features, or `design matrix`
-   `Unstructured` data refers to data that do not follow this format, such as images and audios

## `Exploratory Data Analysis (EDA)`

-   EDA refers to collecting information to better understand the data
-   It is typical to collect representative statistics (such as mean, minimum, maximum, etc) for each feature / label and detect and correct outliers and errors
-   Histogram visualization helps with understanding the feature distribution to determine possible means to normalize the features and label distribution to identify `class imbalance`
-   Outliers
    -   For both features and labels, check for outliers, missing values, and other mistakes.
    -   Feature outliers can be `clipped`
    -   Missing data can be `imputed` by mean or most frequent value
    -   If too many observations are missing the feature, the feature can be `discarded`
    -   Check for incorrect entries such as typos
    -   For labels, check whether all the entries are in expected range
    -   Check every case for label correctness if possible, otherwise spot check
    -   For missing label, ask for annotation or discard the observation

-   `Correlations`

    Additionally, for supervised learning, check the correlation between features and labels. Identify feature highly correlated with labels and use them to establish an early baseline performance.

    -   Check the correlation among all the features to identify highly-correlated feature, which can be removed
    -   Check the correlation between features and labels to gain a early understanding of the predictability of the features
    -   Note that correlation only finds linear relationships, `mutual information` can be used to find more complex relationships
    -   Correlation is only applicable to univariate analysis. For multi-variate, `feature selection` techniques can be used later in the pipeline

After EDA, various approaches can be applied to the features / labels to help with model training. For example, feature normalization allows certain algorithms to converge faster. These approaches are referred to as `feature engineering`

## [Feature Engineering](./Engineering.md)
