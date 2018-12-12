# General Information

```python
seqlearner.SemiSupervisedLearner(X, y, X_t, y_t)
```

SemiSupervisedLearner is a Wrapper class for semi-supervised learning methods. This class will apply a specific semi-supervised learning method on the sequences and return score for validation sequences.
This markdown contains some information about general and helper functions in SemiSupervisedLearner class.

### Arguments
- __X__: list, numpy ndarray, pandas DataFrame, list of training embedding vectors for learning
- __y__: list, numpy ndarray, pandas DataFrame, list of training labels
- __X_t__: list, numpy ndarray, pandas DataFrame, list of test embedding vectors for learning
- __y_t__: list, numpy ndarray, pandas DataFrame, list of test labels


### Example: Apply a SemiSupervised algorithm on unlabeled sequences using PseudoLabeling

```python
from sklearn.model_selection import train_test_split
from seqlearner import MultiTaskLearner
labeled_path = "../data/labeled.csv"
unlabeled_path = "../data/unlabeled.csv"
mtl = MultiTaskLearner(labeled_path, unlabeled_path)
encoding = mtl.embed(word_length=5)
X, y, X_t, y_t = train_test_split(mtl.sequences, mtl.labels, test_size=0.33)
score = mtl.semi_supervised_learner(X, y, X_t, y_t, ssl="pseudo_labeling", sample_rate=0.3)
```

### See Also
- [SemiSupervisedLearner code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/SemiSupervisedLearner.py)

