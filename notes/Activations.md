# Activations

NNs introduce non-linearity by applying `non-linear activation` to the units. If all layers have linear activation, then it is the same as one layer with linear activation. Common activation functions are,

-   `Rectified Linear Unit (ReLU)`: g(x) = max(0, x), ReLU mitigates vanishing gradients and trains faster but may experience dead ReLU.
-   `Leaky ReLU`: 
-   `Parameterized PReLU`: 
-   `Gaussian Error Linear Units GeLU`: 

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

