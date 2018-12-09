# Freq2Vec
```python
import seqlearner
seqlearner.Freq2Vec(sequences, word_length, window_size, emb_dim, loss, epochs)
```

Freq2Vec is an Embedding Method. This class is wrapper for Freq2Vec Embedding method to apply on a set of sequences. Child class of WordEmbedder.
You can train Embedding layer on vocabulary in order to get embedding weights for each word in vocabulary. compress each in `emb_dim` vectors with `freq2vec_maker` method.
You can make an instance of it with the following parameters:

### Arguments
- __sequences__: numpy ndarray, list, or DataFrame
   sequences of data like protein sequences
- __word_length__: integer
    The length of each word in sequences to be separated from each other.
- __window_size__: integer
    Size of window for counting the number of neighbors.
- __emb_dim__:: integer
    Number of embedding vector dimensions.
- __loss__: basestring
    The loss function is going to be used on training phase.
- __epochs__: integer
    Number of epochs for training the embedding.

### Example: make the embedding of protein sequences

```python
import pandas as pd
from seqlearner import Freq2Vec
sequences = pd.read_csv("./protein_sequences.csv", header=None)
freq2vec = Freq2Vec(sequences, word_length=3, window_size=5, emb_dim=25, loss="mean_squared_error", epochs=250)
freq2vec.freq2vec_maker()
```




