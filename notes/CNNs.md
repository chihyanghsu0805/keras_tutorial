#  Convolution Neural Networks (CNNs)

CNNs are used for many `computer vision` tasks, such as object detection and segmentation. Convolution is an very important operation in signal processing and is basically a sliding window element wise multiplication and sum. In practice, `cross-correlation` is used but still refered as `convolution` as convolution requires `flipping`. Instead of fully connected layers, CNNs are built with convolution layers that the sliding window is defined by convolution `kernels`.

-   Due to the `kernels / filters`, the outputs becomes smaller than the input by f - 1, or `valid` convolution. Additionally, `stride` also makes the outputs smaller. `Padding` can be used to mitigate the downsizing of output size. `Same` convolution uses padding so the input/output size is the same. The output size can be computed from `O = floor[(I + 2p - f) / s)] + 1` with O as output, I as input, p as padding, f as filter size and s as stride.

-   Images usually have `channels` (RGB) as an additional dimension. It is important to match the kernel dimension and number with the channels.

-   `Weights` are stored within the kernels and each kernel stores one `bias`. Therefore, the number of parameters # #Filters x (kernel size) + #Filters.

-   Another special layer is `Pooling`, which `average pooling` takes the average in th window and `max pooling` takes the max in the window.

-   Convolution and Pooling can be seen as `Infinitely Strong Priors` that some unit connections are forbidden and neighboring units should have equal weights. CNNs have three desirable properties, `sparse interactions`, `parameter sharing` and `equivariant representations / translation invariance`.

-   Staple CNNs are,

    -   LeNet-5 established the decrement of input size with increment of filters and mixed conv with pool
    -   AlexNet used ReLU for activation
    -   VGG-16 used all 3x3 conv and pool with 2x2, stride = 2
    -   `ResNets` uses `skip connections` to form residual blocks and help with naishing gradients to allow stacking more layers.
    -   `Inception Network` uses `Network in Network / (1x1 Convolutions)` to reduce computational cost and number of parameters. For example, using (28x28x192) as input with 5x5 conv to get (28x28x32) output needs (28x28x32)x(5x5x192) = 120M parameters. Using 1x1 conv with 16 filters followed by 5x5 conv needs (28x28x16)x192 + (28x28x32)x(5x5x16) = 12.4M parameters.
    -   `MobileNet` uses depthwise separable convolutions. For example, with normal convolution (6x6x3) * (3x3x3)x5 = 2160. Depthwise (6x6x3) * (3x3)=(4x4x3) + Pointwise (4x4x3) * (1x1x3)=(4x4x5)=672.
    -   `EfficientNet`

Below is some applications,

-   `Object detection` is `Classification + Localization (Bounding Box)` with multiple objects. The y vector is labeld by [P<sub>c</sub> (is there a object), b<sub>x</sub> b<sub>y</sub> b<sub>h</sub>, b<sub>w</sub> (bounding box), C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>]. The loss function is MSE of the bounding box if there is an object, else MSE of the probability. An intuitive approach is the `sliding window` with high computation cost. Below is two algorithms for object detection,

    -   `YOLO` separates the image into grids and estimate the bounding boxes realtive to the grids. It uses IoU for evaluating the predictions. Since multiple bounding boxes may be predicted, `Non Max Suppression` is used to pick the best prediction. Additionally, `Anchor Boxes` can be used when the object's midpoint overlaps.

    -   `Region Proposal Networks (RPNs)` follows three steps, propose regions, classify regions, output label and bounding box. The region proposal evolved from using k-means to CNNs.


-   `Semantic Segmentation` is pixel level classification. A popular structure is UNet which uses transpose convolution for decoding and skip connections.

-   `Landmark Detection` detects where there is a face and the coordinates of the landmarks.

-   `Face Recognition` is to identify whether an input image is in a database. It is an extension of `Face Verification` where an input image is a given person. `One Shot Learning` is used in face recognition where one sample is learned to recognize the same person. A `Similarity Function` is defined and if the difference between two images is smaller tahn threshold, it is the same person. `Siamese Network` is a popular network used for Face Recognition. It uses `Triplet Loss` that compares an anchor (A), positive sample (P) and negative sample (N). The Loss function is L(A,P,N) = max(||f(A)-f(P)||<sup>2</sup> - ||f(A)-f(N)||<sup>2</sup> + α, 0) where α is to prevent the trivial solution f(x) = 0. It is im,portant to chose the triples that are hard, d(A, P) ~= d(A, N) soe the model learns efficiently.

-   `Neural Style Transfer` is used to create fusion of a `style` image and a `content` image. As the units in different layer learns different context, one can pick a unit in a layer and finds the patch that maximizes that unit's activation. Therefore, one can find corresponding units / layers and fuse the combine them. The cost function J(G) is α J<sub>content</sub>(C, G) + β J<sub>style</sub>(S, G).

    -   Content Cost: content is usually extracted from a middle layer and the cost minimizes the MSE between the content layer and the generated layer.

    -   Style Cost: style is usually extrqacted from multiple layers. Style is defined as `correlation` between activations across channels. In other words, how often does the textures co-occur. It can be quantified by the `Gram Matrix`.
