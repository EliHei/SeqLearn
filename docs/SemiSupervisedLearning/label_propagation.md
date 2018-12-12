# LabelPropagation
```python
seqlearner.SemiSupervisedLearner.label_propagation(kernel, gamma, n_neighbors, alpha, max_iter, tol, n_jobs)
```

LabelPropagation classifier for semi-supervised learning. It's one of the basic semi-supervised learning algorithms that assigns labels to previously unlabeled data points. 
At the start of the algorithm, a (generally small) subset of the data points have labels (or classifications). 
These labels are propagated to the unlabeled points throughout the course of the algorithm.

### Arguments
- __kernel__: String, String identifier for kernel function to use or the kernel function itself. Only 'rbf' and 'knn' strings are valid inputs. The function passed should take two inputs, each of shape [n_samples, n_features], and return a [n_samples, n_samples] shaped weight matrix.
- __gamma__: Float, Parameter for rbf kernel
- __n_neighbors__: Positive integer, Parameter for knn kernel
- __alpha__: Float, Clamping factor
- __max_iter__: Positive integer, Change maximum number of iterations allowed
- __tol__: Float, Convergence tolerance: threshold to consider the system at steady state
- __n_jobs__: Positive integer, The number of parallel jobs to run


### Example: make the embedding of protein sequences

```python
import pandas as pd
from seqlearner import Freq2Vec
sequences = pd.read_csv("./protein_sequences.csv", header=None)
freq2vec = Freq2Vec(sequences, word_length=3, window_size=5, emb_dim=25, loss="mean_squared_error", epochs=250)
freq2vec.freq2vec_maker()
```

### See Also
- [Freq2Vec code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/Freq2Vec.py)

