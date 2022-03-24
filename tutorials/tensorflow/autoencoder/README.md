# Autoencdoers
This code follows the tutorial on https://www.tensorflow.org/tutorials/generative/autoencoder.

# Overview
Autoencoders consist of two components, encoder and decoder.

The encoder decreases in size but increases in filters.

The bottleneck layer is the lowest layer and can be used as embeddings/ latent dimensions.

The decoder increases in size and decrease in filters.

Autoencoders can be used in various applications.

In this notebook, three applications were demonstrated.

1. Data Reconstruction/Compression
   
   The data can be represented using the latent dimension trained with identical input/output.

2. Denoising
   
   The denoised data can be restored by training with noist input and cleaned output.

3. Anomaly detection
   
   Outliers can be detected both supervised and unsupervised using mostly normal data.
   
