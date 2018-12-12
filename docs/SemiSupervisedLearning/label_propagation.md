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


### Example: predict the unlabeled sequences

```python
from sklearn.model_selection import train_test_split
from seqlearner import MultiTaskLearner
labeled_path = "../data/labeled.csv"
unlabeled_path = "../data/unlabeled.csv"
mtl = MultiTaskLearner(labeled_path, unlabeled_path)
encoding = mtl.embed(word_length=5)
X, y, X_t, y_t = train_test_split(mtl.sequences, mtl.labels, test_size=0.33)
score = mtl.semi_supervised_learner(X, y, X_t, y_t, ssl="label_propagation")
```

### See Also
- [LabelPropagation code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/SemiSupervisedLearner.py)

