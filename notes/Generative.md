Generative models find the joint probability of y and x P(y, x). From the definition of `conditional probability`, P(x | y) = P(x, y) P(y) and P(y | x) = P(x, y) P(x). Therefore, it can also be seen as modeling P(x | y) P(y) with P(y) as a prior.

-   `Gaussian Discriminant Analysis (GDA)`: Using binary classification as an example to compare with logistic regression, GDA assumes x | y ~ N(μ, Σ) with different μ for y = 0 and 1 but identical Σ (Different Σ results in nonlinear decision boundary) and y is Bernoulli φ. It optimizes for φ, μ<sub>0</sub>, μ<sub>1</sub> and Σ. Under MLE, φ is the ratio of positive samples to all, μ is the average of all postive / negative samples, and Σ is computed with the estimated means. Due to the `stronger assumption`, GDA requires less data but only works when the data is Gaussian. Whereas Logistic Regression has weaker assumption and works with any distribution.

-   `Naive Bayes` is another popular generative model, especially in sequence models. It asuumes Xs are `conditionally independent` given y, P(x<sub>1</sub>, ..., x<sub>n</sub> | y)  = P(x<sub>1</sub> | y) P(x<sub>2</sub> | x<sub>1</sub>, y) P(x<sub>3</sub> | x<sub>2</sub> x<sub>1</sub>, y)... = P(x<sub>1</sub> | y) P(x<sub>2</sub> | y) ... P(x<sub>n</sub> |y). `Laplace Smoothing` is used to avoid zero probabilities.

-   [`Generative Adversarial Networks (GANs)`](./GANs.md)
-   [`Autoencoders (AEs)`](./AE.md)
