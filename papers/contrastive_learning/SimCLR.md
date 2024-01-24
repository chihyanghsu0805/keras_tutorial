# SimCLR

-   A Simple Framework for Contrastive Learning of Visual Representations
    -   https://arxiv.org/abs/2002.05709
    -   ICML 2020
    -   https://github.com/google-research/simclr
    -   https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html

-   Big Self-Supervised Models are Strong Semi-Supervised Learners
    -   https://arxiv.org/abs/2006.10029
    -   NIPS 2020
    -   https://github.com/google-research/simclr


## A Simple Framework for Contrastive Learning of Visual Representations
### Introduction
1. Data augmentation
2. Nonlinear transformation between representation and contrastive loss
3. Larger batch size and more training steps
4. normalized embeddings and adjusted temperature

### Method

- Framework
    -   Data augmentation
        - random cropping followed by resize
        - random color distortions
        - random Gaussian blur
    - Base encoder
        - ResNet fc 1000
    - Projection head
        - MLP with one hidden layer and ReLU -> 128-d
    - Contrastive loss (Normalized temperature scaled cross entropy loss)
        - sim(z_i, z_j) = cosine similarity
        - L(i, j) = -log(exp(sim(z_i, z_j / T)) / sum k != i exp(sim(z_i, z_k) / T))

    - Sample N and augment to get 2N samples, do not explicitely sample negative samples, but use 2(n - 1) samples as negative samples

    - Layer-wise Adaptive Rate Scaling (LARS) optimizer
    - Global BatchNorm
        - Local stats in multi device might leak information
        - Aggreagate stats
        - Shuffle samples
        - Layer norm



### Encoder head
- the hidden layer before the projection head is a better representation than the layer after.

## Big Self-Supervised Models are Strong Semi-Supervised Learners
### Introduction

- unsupervised pretraining
- supervised fine-tuning
- distillation with unlabeled examples

### Method

- ResNet-152, selective kernels
- 3 layer projection, using a deeper projection head during pretraining is better when fine-tuning from the optimal layer of projection head
- memory mechanism
