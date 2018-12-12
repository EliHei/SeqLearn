# GensimWord2Vec
```python
seqlearner.GensimWord2Vec(sequences, word_length, window_size, emb_dim, loss, epochs)
```

Word2Vec is an Embedding Method. This class contains the implementation for Word2Vec Embedding method to apply on a set of sequences. Child class of WordEmbedder.
You can train Embedding layer on vocabulary in order to get embedding weights for each word in vocabulary. compress each in `emb_dim` vectors with `freq2vec_maker` method.
The `freq2vec_maker` method returns the embedding weights of the vocabulary. You can access to the vocabulary via `vocab` attribute.

### Arguments
- __sequences__: Numpy ndarray, list, or DataFrame, sequences of data like protein sequences
- __word_length__: Positive integer, the length of each word in sequences to be separated from each other.
- __window_size__: Positive integer, size of window for counting the number of neighbors.
- __emb_dim__: Positive Integer, number of embedding vector dimensions.
- __loss__: String, the loss function is going to be used on training phase.
- __epochs__: Positive integer, number of epochs for training the embedding.

### freq2vec_maker
 
This is a function of Word2Vec class which you can use to embed your vocabulary.
you can train Embedding layer on vocabulary in order to get embedding weights for each word in vocabulary. compress each in `emb_dim` vectors.
This function accepts no arguments.

### Example: make the embedding of protein sequences

```python
import pandas as pd
from seqlearner import GensimWord2Vec
sequences = pd.read_csv("./protein_sequences.csv", header=None)
freq2vec = GensimWord2Vec(sequences, word_length=3, window_size=5, emb_dim=25, loss="mean_squared_error", epochs=250)
freq2vec.freq2vec_maker()
```

### See Also
- [Word2Vec code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/GensimWord2Vec.py)

