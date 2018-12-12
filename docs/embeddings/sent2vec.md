# Sent2Vec
```python
seqlearner.Sent2Vec(sequences, word_length, emb_dim, epochs, lr, wordNgrams, loss, neg, thread, t, dropoutK, bucket)
```
Sent2Vec Embedding Method. This class is wrapper for Sent2Vec Embedding method to apply on a set of sequences. This is child class of WordEmbedder.
You can train Embedding layer on vocabulary in order to get embedding weights for each word in vocabulary. compress each in `emb_dim` vectors with `sent2vec_maker` method.
The `sent2vec_maker` method returns the embedding weights of the vocabulary. You can access to the vocabulary via `vocab` attribute.

### Arguments
- __sequences__: Numpy ndarray, list, or DataFrame, sequences of data like protein sequences
- __word_length__: Positive integer, the length of each word in sequences to be separated from each other.
- __window_size__: Positive integer, size of window for counting the number of neighbors.
- __emb_dim__: Positive integer, number of embedding vector dimensions.
- __epochs__: Positive integer, number of epochs for training the embedding.
- __loss__: String, the loss function is going to be used on training phase.
- __wordNgrams__: Positive integer, max length of word n-gram
- __loss__: String, loss function, possible values are {"ns", "hs", "softmax"}
- __neg__: Positive integer, number of negatives sampled
- __thread__: Positive integer: number of threads
- __t__: Float, sampling threshold
- __dropoutK__: Positive integer, number of n-grams dropped when training a sent2vec model
- __bucket__: Positive integer, number of hash buckets for vocabulary


### Example: make the embedding of protein sequences

```python
import pandas as pd
from seqlearner import Sent2Vec
sequences = pd.read_csv("./protein_sequences.csv", header=None)
sent2vec = Sent2Vec(sequences, word_length=3, emb_dim=25, epoch=100, lr=0.2, wordNgrams=5, loss="hs", neg=20, thread=10, t=0.0000005, dropoutK=2, bucket=4000000)
encoding = sent2vec.sent2vec_maker()
```

### See Also
- [Sent2Vec code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/Sent2Vec.py)

