#   Data

## Data Types

-   `Structured` data refers to data that is represented with rows as subjects and columns as features, or `design matrix`
-   `Unstructured` data refers to data that do not follow this format, such as images and audios

## `Exploratory Data Analysis (EDA)`

-   EDA refers to collecting information to better understand the data
-   It is typical to collect representative statistics (such as mean, minimum, maximum, etc) for each feature / label and detect and correct outliers and errors
-   Histogram visualization helps with understanding the feature distribution to determine possible means to normalize the features and label distribution to identify `class imbalance`
-   After EDA, various approaches can be applied to the features / labels to help with model training
-   For example, feature normalization allows certain algorithms to converge faster
-   These approaches are referred to as `feature engineering`
