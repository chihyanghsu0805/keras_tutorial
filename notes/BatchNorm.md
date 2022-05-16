# Batch Normalization

 `Batch Normalization` helps with vanishing / exploding gradients as well as `internal covariate shift` due to random initialization. Bach norm works by making the weights in later layers more robust to chnges in earlier layer. The normalization parameters can be learned so that `batchnorm is not always zero mean and unit variance`. During training, batchnorm is estimated with `exponentially weighted averages`. 
 At test time, the batchnorm uses `parameters from training` often by running average.
 
  However, evidences show that Batch Normalization may induce `severe gradient explosion` at initialization. Batch norm also incurs inter-device synchronization
cost and the need for running statistics limits transfer learning. [1]

Batchnorm suffers when the batch size is small and the batch statistics is not representative, e.g. computer vision.
Variants of normalization includes,

- `instance normalization` [4]
- `layer normalization` [3]
- `weight normalization`
- `group normalization`[2]

[1] Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S. and Houlsby, N., 2020, August. Big transfer (bit): General visual representation learning. In European conference on computer vision (pp. 491-507). Springer, Cham.

[2] Wu, Y. and He, K., 2018. Group normalization. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19).

[3] Ba, J.L., Kiros, J.R. and Hinton, G.E., 2016. Layer normalization. arXiv preprint arXiv:1607.06450.

[4] Ulyanov, D., Vedaldi, A. and Lempitsky, V., 2016. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022.

[5] Salimans, T. and Kingma, D.P., 2016. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. Advances in neural information processing systems, 29.

