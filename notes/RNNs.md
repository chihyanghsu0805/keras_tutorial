#  Recurrent Neural Networks (RNNs)

RNNs are used for `sequential` data, such as text and video. Based on the inputs and outputs, there are several variations,

-   One to One
-   Many to One: Sentiment Classification
-   One to Many: Music Generation
-   Many to Many: Machine Translation

Sequence data may contain non numerical data such as text, the different ways of converting text to numbers are

-   `Vocabulary` with tokens with specialized tokens `EOS` for end of sentence and `OOV / UNK` for out of vocabulary
-   `Embeddings`

Comapred to a standard NN which units are connectly vertically, RNNs also connect units horizontally across time / sequence. Therefore, during forward progagation the input includes the t<sup>th</sup> input x<sup>t</sup> and activation from previous sequence a<sup>t-1</sup> with different weight matrices W<sub>aa</sub> and W<sub>ax</sub> which can be concatenated into one matrix. Parameters are updated via `backpropagation through time`.

-   a<sup>t</sup> = g(W<sub>a</sub>[a<sup>t-1</sup>, x<sup>t</sup>] + b<sub>a</sub>)
-   y_hat<sup>t</sup> = g(W<sub>ya</sub>a<sup>t</sup> + b<sub>y</sub>)


##  Sequence Generation

An example of sequence generating model is `Naive Bayes` which models P(sentence) as a conditionally independent distribution p(y<sup>1</sup> ... y<sup>T</sup>). Using a RNN model, the inital input can be a random number and generates y_hat<sup>1</sup> which is used as the input for the second unit. It can work at a `word-level` or `character-level`, in which longer vocabularies may suffer from `Long-Range Dependencies`.

##  Long-Range Dependencies

`Long-Range Dependencies` is a big challenge in sequence modeling due to vanishing / exploding gradients across time and space. Besides `skip connections`, `Gated RNNs`, such as `Gated Recurrent Units (GRUs)` and `Long Short Term Memory (LSTM)` are designed to address long-term dependencies by introducing `gates`.

### GRUs

GRUs have two gates, `update, u` and `relevant, r` with the following update rules.

-   C_hat<sup>t</sup> = tanh(W<sub>c</sub>[Γ<sub>r</sub> .* C<sup>t-1</sup>, x<sup>t</sup>] + b<sub>c</sub>)
-   Γ<sub>u</sub> = σ(W<sub>u</sub>[C<sup>t-1</sup>, x<sup>t</sup>] + b<sub>u</sub>)
-   Γ<sub>r</sub> = σ(W<sub>r</sub>[C<sup>t-1</sup>, x<sup>t</sup>] + b<sub>c</sub>)
-   C<sup>t</sup> = Γ<sub>u</sub> .* C_hat<sup>t</sup> + (1 - Γ<sub>u</sub>) .* C_hat<sup>t-1</sup>

### LSTMs
LSTMs uses the `cell` state to carry the information across the sequence. It is governed by three gates, `input`, `forget` and `cell / memory`. The forget gate controls how much to forget from the previous cell state. The input and cell gate control how much to update the cell state to pass to the next sequence. An additional `output` gate combines the updated cell state and the input state feed into the next unit. The update rules are,

-   C_hat<sup>t</sup> = tanh(W<sub>c</sub>[a<sup>t-1</sup>, x<sup>t</sup>] + b<sub>c</sub>)
-   Γ<sub>u</sub> = σ(W<sub>u</sub>[a<sup>t-1</sup>, x<sup>t</sup>] + b<sub>u</sub>)
-   Γ<sub>f</sub> = σ(W<sub>f</sub>[a<sup>t-1</sup>, x<sup>t</sup>] + b<sub>f</sub>)
-   Γ<sub>o</sub> = σ(W<sub>o</sub>[a<sup>t-1</sup>, x<sup>t</sup>] + b<sub>o</sub>)
-   C<sup>t</sup> = Γ<sub>u</sub> .* C_hat<sup>t</sup> + Γ<sub>f</sub> .* C_hat<sup>t-1</sup>
-   a<sup>t</sup> = Γ<sub>o</sub> .* tanh(C<sup>t</sup>)

## Bidirectional

Connections can be `causal` where only information from the past is used. It can also be `bidirectional` where the entire sequence is being used.

-   y_hat<sup>t</sup> = g(W<sub>y</sub>[a<sup>->t</sup>, a<sup><-t</sup>] + b<sub>y</sub>)

## Word Embeddings

Using vocabularies do not find the relationship between similar words such as apple and orange since the dot product of one hot is zero. `Embeddings` encode the text into a feature space that can be used as `similarity` between words. Hence, `transfer learning` is commonly used in word embeddings since the embeddings should be shared disregard of the task.

To learn word embeddings, it is common to observe in a `history window` and count the co-occurence of word pairs, or `skip-grams` with `context` and `target`.

### Word2Vec

Word2Vec utilizeds skip gram and softmax to model P(target | context). One challenge is the vocabulary size is too big and full softmax becomes expensive. `Hiearchical Softmax` and `Candidate / Negative Sampling` can be used instead. To sample the context C, one can use `Random (Negative) Sampling` or Heuristics.

### Glove

Glove stands for Global Vectors for Word Representation and utilizes the co-occurence statistics and minimizes the MSE of the product of θ e and log of co-occurence X<sub>ij</sub> .

 Since word embeddings are trained from collected materials, it may be `biased`. To debias the word embeddings,

 -  Identify bias direction
 -  Neutralize
 -  Equalize pairs

## Sequence to Sequence

There are many sequence to sequence applications,

-   `Image Captioning` where input is image and output is text
-   `Machine Translation` where input and output are both text

### Machine Translation

Machine translation models the P(y<sup>1</sup>...y<sup>t</sup> | x). By design, longer sentences have lower probability, hence `Length Normalization` is needed. `Greedy search` chooses the word with highest probability at each location for the entire senetence. `Beam Search` chooses N best sequences with `beam width`. Beam width trades off result and efficiency with beam width = 1 identical to greedy. Error Analysis in Beam search has two scenarios,

-   P(y* | x) > P (y_hat | x): beam search is at fault
-   P(y* | x) <=> P (y_hat | x): RNN is at fault, may need length normalization

`Bleu Score` is commonly used for evaluating translation. Bleu stands for Bilingual Evaluation Understudy.
The `precision` is defined as (#occurence in truth) / (#occurence in prediction). `Brevity Penalty` is used to penalize short sentences.

`Attention Model` us used to overcome problems with long sequences. It multiplies the activation by `attention α` where the sum of attention is 1.
