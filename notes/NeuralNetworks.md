#   Neural Networks (NN):

NNs consists of layers which is a collection of units / neurons. Each unit represents a parameter, and the connections between units are the weights. NNs are often represented by `computational graphs`. NNs with multi-layers are also known as `Multilayer Perceptron (MLP)` and the input layer is usually not counted. Conventioally, weights are represented using `w and b`. NNs increase model capacity by increasing number of layers (depth) and number of units (width). Typically, the input layer is visualized as the bottom layer and output layer on top. In-between layers are called `hidden layers`. NNs with multiple hidden layers are referred to as Deep Neural Networks (DNN). Layers with all units connected are know as `fully connected / dense` layers.

##  Initialization

Initialization is extremely import for NNs. The initial parameters need to `break symmetry` to ensure each unit learns different functions. Typically biases are set to constants and weights are randomly sampled form a Gaussian or Uniform distribution. The `scale of the distribution` has a large impact as well. If the initial values are too big, gradients may explode. Likewise initial values too small, gradients may vanish. Generally, the wights are initialized to have equal variance. the Below are some common initializations,

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
-   W (n</sub>i+1</sub>, n<sub>i</sub>)
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

 `Batch Normalization` helps with vanishing / exploding gradients as well as `internal covariate shift` due to random initialization. Barch norm works by making the weights in later layers more robust to chnages in earlier layer. The normalization parameters can be learned so that `batchnorm is not always zero mean and unit variance`. However, evidences show that Batch Normalization may induce `severe gradient explosion` at initialization. During training, batchnorm is estimated with `exponentially weighted averages`. And at test time, the batchnorm `parameters from training` is used.

##  Regularization

NNs have a special regularization technique `Dropout` which randomly drops units during training. It has the effect of evenly distributing the weights throughout the network and `not rely on specific units`. With wider layers, the keep probability should be lower. At test time, dropout is turned off.

L2 regularization in NNs is the `Frobenius Norm` of the weight matrix W and is often referred to as `weight decay` due to the refactoring of terms in gradient descent.

##  Convolution Neural Networks (CNNs)

CNNs are used for many computer vision tasks, such as object detection and segmentation. Convolution is an very important operation in signal processing and is basically a sliding window approach. In practice, `autocorrelation` is used but still refered as `convolution`. Instead of fully connected layers, CNNs are built with convolution layers that the sliding window is defined by convolution `kernels`. Due to the kernels, the outputs becomes smaller than the input. Additionally, `stride` also makes the outputs smaller. `Padding` can be used to mitigate the downsizing of output size. The output size can be computed from `O = floor[(I + 2p - f) / s)] + 1`. Another special layer in CNNs is `pooling`, which `average pooling` takes the average in th window and `max pooling` takes the max in the window. Convolution and Pooling can be seen as `Infinitely Strong Priors` that some unit connections are forbidden and neighboring units should have equal weights. CNNs have three desirable properties, `sparse interactions`, `parameter sharing` and `equivariant representations`.

## Recurrent Neural Networks (RNNs)

RNNs are used for sequential data, such as text and video. Besides connection between layers, units are also connected across time / sequence. Therefore, parameters are updated via `backpropagation through time`. Connections can be `causal` where only information from the past is used. It can also be `bidirectional` where the entire sequence is being used. `Long-Term Dependencies` is a big challenge in sequence modeling due to vanishing / exploding gradients across time and space. Besides `skip connections`, `Gated RNNs`, such as `Gated Recurrent Units (GRUs)` and `Long Short Term Memory (LSTM)` are designed to address long-term dependencies by introducing `gates`. In particular, LSTMs uses the `cell` state to carry the information across the sequence. It is governed by three gates, `input`, `forget` and `cell / memory`. The forget gate controls how much to forget from the previous cell state. The input and cell gate control how much to update the cell state to pass to the next sequence. An additional `output` gate combines the updated cell state and the input state feed into the next unit. The input and previous output is concatenated for the gates.
