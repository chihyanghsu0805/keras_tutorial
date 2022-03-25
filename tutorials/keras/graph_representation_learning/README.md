# Graph Representation Learning with node2vec
This code follows the tutorial on https://keras.io/examples/graph/node2vec_movielens/.

## Overview
This code takes movie ratngs and build a weighted graph between the movies.

Positive and negative samples are determined by biased random walk of the graph.

Embeddings are learnt from training a classifier with positive and negative examples.
