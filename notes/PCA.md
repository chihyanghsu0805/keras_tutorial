#  Principal Components Analysis (PCA)

PCA is a very popular algorithm to find lower-dimension data representations, or `Principal Components`. It can be seen as `projection` of the data points with minimum projection error. To apply PCA,

-   Input x (n ,1), m samples
-   `Feature Normalization and Zero-Mean`
-   Compute Covariance Matrix, Σ = (1/m) xx<sup>T</sup>, (n, n)
-   Compute eigenvectors of Σ using eig or svd (eig = svd when the matrix is symmetric positive definite)
    -   U, S, V = svd(Σ), U (n, n), S (n, m), V (m, m)
-   Choose number of principal components K
    -   Increase K until % `variance explained` is reached
    -   The diagonal S matrix stores variance explained
