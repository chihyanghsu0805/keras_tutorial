#  Autoencoder (AEs)

Autoencoders is a NN that is trained to attemp to copy its input to its output. It consist of two parts, `Encoder` and `Decoder`. The encoder compresses the input (x) to `latent variables (h)`, or p(h | x), and the decoder restores the input from h, p(x | h). h is usuallly constrained to have a smaller dimension than x, or `undercomplete`. An undercomplete representation forces the AE to learn the `most salient` features. The loss function is L(x, g(f(x)).When te decoder is linear, and L is MSE, AEs span the same subspace as PCA.

When the encoder and decoder has too much capacity, AE may fail to learn meaninful representations. The same when h > x, or `overcomplete`. Rather than limiting the model capacity, `regularized` AEs use a loss function to have other properties besides copying the input to output, such as `sparsity of the representations`, `smallness of the derivative of the representation`, and `robustness to noise or missing inputs`.

## Variational Autoencoders (VAEs)
