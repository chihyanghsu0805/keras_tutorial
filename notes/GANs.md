#  Generative Adversarial Networks (GANs)

GANs train two models,

-   `Generator G` generates plausible data that acts as negative training sample for discriminator. It samples from random noise to produce a generated output to be classified by the discriminator. The generator loss is computed and backpropagetd but `only the generator is updated`.

-   `Discriminator D` distinguish fake fom real and penalizes generator for implausible data. The discriminator classifies real and fake data and is penalizaed for misclassification. The parameters are updated thorugh backpropagation.

GAN training alternates between generator and discriminator, the discriminator needs to be good to provide informative updates to the generator. GAN convergence is hard to identify.

## Loss Function

The loss function is a `minimax loss` L = E<sub>x</sub>[log(D(x))] + E<sub>z</sub>[log(1-D(G(Z)))] where D maximizes the loss E<sub>x</sub> and G minimizes E<sub>z</sub>. The loss gets stuick when D is easy. Therefore a modified version is used, L = E<sub>x</sub>[log(D(x))] + E<sub>z</sub>[log(D(G(Z)))]

A variation of the loss is the `Wasserstein Loss` in `WGAN` where the `Earth Mover's Distance` is used. The discriminator no longer classifies but output a number that real is bigger than fake, or a `critic`. The critic maximizes D(x) - D(G(z)) `critic loss`, and the generator maximizes D(G(z)) `generator loss`. The gradients for WGAN need to be `clipped` throughout the GAN.

## Common Problems

-   `Vanishing Gradients` may happend when D is too good. Wasserstein Loss and the Modified Minimax Loss can help.
-   `Mode collapse` happens when the generator only output certain output. It can be alleviated by using `WGAN`.
-   `Failure to Converge` can be remedied by regularization such as `Adding Noise to D inputs` and `Penalizing D Weights`.

##  Applications

-   Progressive GANs
-   Conditional GANs
-   Image-to-Image Translation
-   CycleGAN
-   Text-to-Image Synthesis
-   Super-Resolution
-   Face Inpainting
-   Text-to-Seppech
