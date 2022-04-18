#   Neural Networks (NN):

NNs consists of layers which is a collection of units / neurons. Each unit represents a parameter, and the connections between units are the weights. NNs are often represented by `computational graphs`. NNs with multi-layers are also known as `Multilayer Perceptron (MLP)` and the input layer is usually not counted. Conventionally, weights are represented using `w and b`. NNs increase model capacity by increasing number of layers (depth) and number of units (width). Typically, the input layer is visualized as the bottom layer and output layer on top. In-between layers are called `hidden layers`. NNs with multiple hidden layers are referred to as Deep Neural Networks (DNN). Layers with all units connected are known as `fully connected / dense` layers.

##  Initialization

Initialization is extremely import for NNs. The initial parameters need to `break symmetry` to ensure each unit learns different functions. Typically biases are set to constants and weights are randomly sampled form a Gaussian or Uniform distribution. The `scale of the distribution` has a large impact as well. If the initial values are too big, gradients may explode. Likewise initial values too small, gradients may vanish. Generally, the wights are initialized to have equal variance. Below are some common initializations,

-   (Xavier) Glorot and Bengio Uniform: U[-1/sqrt(n), 1/sqrt(n)]
-   Xavier Uniform: U[-sqrt(6)/sqrt(m+n), sqrt(6)/sqrt(m+n)]
-   Xavier Normal: N(0, sqrt(2/(m + n)))
-   He for ReLU:  N(0, sqrt(2/n))

##  Activation Functions

NNs introduce non-linearity by applying `non-linear activation` to the units. If all layers have linear activation, then it is the same as one layer with linear activation. Common activation functions are,

-   `Rectified Linear Unit (ReLU)`: g(x) = max(0, x), ReLU mitigates vanishing gradients and trains faster but may experience dead ReLU.
-   `Sigmoid`: g(x) = 1 / (1 + e<sup>-x</sup>)
-   `tanh`: g(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)

A single layer is represented as A = g(Z), Z = WX + b with the following shapes

-   n = number of features / filters
-   m = number of samples
-   X (n<sub>i</sub>, m), note this is different in conventional design matrix
-   W (n<sub>i+1</sub>, n<sub>i</sub>)
-   Z (n<sub>i+1</sub>, m)
-   A (n<sub>i+1</sub>, m)
-   b (n<sub>i</sub>, 1) but `broadcasted` to (n<sub>i</sub>, m)

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

Another challenge of DNNS is the vanishing / exploding gradients due to the many layers. This can be mitigated with `skip connections`. `Gradient clipping` also helps with exploding gradients.

## Batch Normalization

 `Batch Normalization` helps with vanishing / exploding gradients as well as `internal covariate shift` due to random initialization. Bach norm works by making the weights in later layers more robust to chnages in earlier layer. The normalization parameters can be learned so that `batchnorm is not always zero mean and unit variance`. During training, batchnorm is estimated with `exponentially weighted averages`. And at test time, the batchnorm `parameters from training` is used.
 
  However, evidences show that Batch Normalization may induce `severe gradient explosion` at initialization. Batch norm also incurs inter-device synchronization
cost and the need for running statistics limits transfer learning. [1]

 There are also variants of normalization, such as `instance normalization`, `layer normalization`, and `group normalization`.

[1] Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S. and Houlsby, N., 2020, August. Big transfer (bit): General visual representation learning. In European conference on computer vision (pp. 491-507). Springer, Cham.

##  Regularization

NNs have a special regularization technique `Dropout` which randomly drops units during training. It has the effect of evenly distributing the weights throughout the network and `not rely on specific units`. With wider layers, the keep probability should be lower. At test time, dropout is turned off.

L2 regularization in NNs is the `Frobenius Norm` of the weight matrix W and is often referred to as `weight decay` due to the refactoring of terms in gradient descent.

##  [`Convolution Neural Networks (CNNs)`](./CNNs.md)

##  [`Recurrent Neural Networks (CNNs)`](./RNNs.md)
