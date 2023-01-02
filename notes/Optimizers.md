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
