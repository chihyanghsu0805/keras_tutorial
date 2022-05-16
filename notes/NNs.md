#   Neural Networks (NN):

NNs consists of layers which is a collection of units / neurons. Each unit represents a parameter, and the connections between units are the weights. NNs are often represented by `computational graphs`. NNs with multi-layers are also known as `Multilayer Perceptron (MLP)` and the input layer is usually not counted. Conventionally, weights are represented using `w and b`. NNs increase model capacity by increasing number of layers (depth) and number of units (width). Typically, the input layer is visualized as the bottom layer and output layer on top. In-between layers are called `hidden layers`. NNs with multiple hidden layers are referred to as Deep Neural Networks (DNN). Layers with all units connected are known as `fully connected / dense` layers.

##  Initialization

Initialization is extremely import for NNs. The initial parameters need to `break symmetry` to ensure each unit learns different functions. Typically biases are set to constants and weights are randomly sampled form a Gaussian or Uniform distribution. The `scale of the distribution` has a large impact as well. If the initial values are too big, gradients may explode. Likewise initial values too small, gradients may vanish. Generally, the wights are initialized to have equal variance. Below are some common initializations,

-   (Xavier) Glorot and Bengio Uniform: U[-1/sqrt(n), 1/sqrt(n)]
-   Xavier Uniform: U[-sqrt(6)/sqrt(m+n), sqrt(6)/sqrt(m+n)]
-   Xavier Normal: N(0, sqrt(2/(m + n)))
-   He for ReLU:  N(0, sqrt(2/n))

##  [Activation Functions](./Activations.md)

##  Forward / Backward Propagation

One key challenge of DNNs is the computation of parameter updates which is addressed by `backpropagation`. A `forward propagation` gives the prediction while the backpropagation computes the parameter updates using `chain rule`. The parameters are updated using iterative methods such as gradient descent. Due to the use of chain rule, activation functions are usually `differentiable` (sigmoid and tanh) or can be easily appoximated (ReLU).

Derivatives of commong activation functions,

-   `Rectified Linear Unit (ReLU)`: 0 if z < 0 else 1 > 0
-   `Sigmoid`: a (1 - a)
-   `tanh`: 1 - a<sup>2</sup>

The shapes of the derivatives are the same with the variables.

##  Why Deep?

-   Deeper filters may be combinations of previous layers, (Edge -> Corner -> Shape)
-   It is common to increase depth rather than width since it takes exponentially more units to represent the same function (Circuit Theory).

## Vanishing / Exploding Gradients

Another challenge of DNNS is the vanishing / exploding gradients due to the many layers. This can be mitigated with `skip connections`. `Gradient clipping` also helps with exploding gradients. `ReLUs` also help with vanishing gradients.

## [Batch Normalization](./BatchNorm.md)

##  Regularization

NNs have a special regularization technique `Dropout` which randomly drops units during training. It has the effect of evenly distributing the weights throughout the network and `not rely on specific units`. With wider layers, the keep probability should be lower. At test time, dropout is turned off.

- Dropout at test time use all units but with the weights going out of unit multiplied by the probability of including unit, weight scaling inference rule.

`Stochastic Depth` [1] is another regularization technique for NNs. 

L2 regularization in NNs is the `Frobenius Norm` of the weight matrix W and is often referred to as `weight decay` due to the refactoring of terms in gradient descent.

[1] Huang, G., Sun, Y., Liu, Z., Sedra, D. and Weinberger, K.Q., 2016, October. Deep networks with stochastic depth. In European conference on computer vision (pp. 646-661). Springer, Cham.

##  [`Convolution Neural Networks (CNNs)`](./CNNs.md)

##  [`Recurrent Neural Networks (CNNs)`](./RNNs.md)

