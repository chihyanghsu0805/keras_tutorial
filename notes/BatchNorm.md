# Batch Normalization

Feature normalization helps the training converge faster. Therefore, it makes sense to normalize the inputs to each hidden layer. Additionally, `internal covariate shift` or changes of the distribution of activations due to change of parameters are mitigated by `batch normalization`.

At train time,

- Batch norm uses mean and standard deviation across the mini batch and spatial dimensions to normalize the activations
- Additional parameters can be learned the mean is not always zero and the standard deviation is not always one
- Batch norm also helps with `vanishing / exploding` gradients
- In [1], batch norm is proposed to be before the activation

At test time,

- Means and standard deviations are estimated with `exponentially weighted averages` of mini batch stats from training
 
  
However, evidences show that Batch Normalization may induce `severe gradient explosion` at initialization. Batch norm also incurs inter-device synchronization cost and the need for running statistics limits transfer learning [2]. Batch normalization also suffers when the batch size is small and the batch statistics is not representative, e.g. computer vision. 

Other normalization techniques focused on normalizing across channelsm or instances,

- `Instance normalization` [3] computes the stats across spatial dimensions in a single channel and single instance, and is used in style transfer
- `Layer normalization` [4] computes the stats across spatial dimensions and channels in a single instance
- `Group normalization`[5] generalizes instance and layer normalization with parameter G, where G = 1 is layer normalization, and G = #C is instance normalization

`Weight normalization`

[1] He, K., Zhang, X., Ren, S. and Sun, J., 2016, October. Identity mappings in deep residual networks. In European conference on computer vision (pp. 630-645). Springer, Cham.

[2] Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S. and Houlsby, N., 2020, August. Big transfer (bit): General visual representation learning. In European conference on computer vision (pp. 491-507). Springer, Cham.

[3] Ulyanov, D., Vedaldi, A. and Lempitsky, V., 2016. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022.

[4] Ba, J.L., Kiros, J.R. and Hinton, G.E., 2016. Layer normalization. arXiv preprint arXiv:1607.06450.

[5] Wu, Y. and He, K., 2018. Group normalization. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19).

[6] Salimans, T. and Kingma, D.P., 2016. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. Advances in neural information processing systems, 29.

