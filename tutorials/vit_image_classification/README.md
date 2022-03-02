# Image Classification with Vision Transformer

## Overview
This tutorial trains a Vision Transformer to classify the CIFAR dataset.

The original paper is https://arxiv.org/pdf/2010.11929.pdf.

Vision Transformers work as follows:
1. Extract patches from the original image and flatten them.
2. Linear project the flatten images.
3. Concatenate position embeddings.
4. Feed the projection and embeddings into a Transformer Encoder consisting of Multi Head Attention and Multi Layer Perceptron.
5. Feed the encodings to a MLP head for classification.

![Alt text](./images/vit.PNG)
! Image snapshot from original paper.

## Transformer Encoder

### Multi Head Attention
