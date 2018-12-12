# LabelSpreading
```python
seqlearner.SemiSupervisedLearner.label_spreading(kernel, gamma, n_neighbors, alpha, max_iter, tol, n_jobs)
```

LabelSpreading model for semi-supervised learning This model is similar to the basic Label Propagation algorithm, but uses affinity matrix based on the normalized graph Laplacian and soft clamping across the labels.

### Arguments
- __kernel__: String, String identifier for kernel function to use or the kernel function itself. Only 'rbf' and 'knn' strings are valid inputs. The function passed should take two inputs, each of shape [n_samples, n_features], and return a [n_samples, n_samples] shaped weight matrix.
- __gamma__: Float, Parameter for rbf kernel
- __n_neighbors__: Positive integer, Parameter for knn kernel
- __alpha__: Float, Clamping factor
- __max_iter__: Positive integer, Change maximum number of iterations allowed
- __tol__: Float, Convergence tolerance: threshold to consider the system at steady state
- __n_jobs__: Positive integer, The number of parallel jobs to run


### Example: predict the unlabeled sequences

```python
from sklearn.model_selection import train_test_split
from seqlearner import MultiTaskLearner
labeled_path = "../data/labeled.csv"
unlabeled_path = "../data/unlabeled.csv"
mtl = MultiTaskLearner(labeled_path, unlabeled_path)
encoding = mtl.embed(word_length=5)
X, y, X_t, y_t = train_test_split(mtl.sequences, mtl.labels, test_size=0.33)
score = mtl.semi_supervised_learner(X, y, X_t, y_t, ssl="label_spreading")
```

### See Also
- [LabelSpreading code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/SemiSupervisedLearner.py)

