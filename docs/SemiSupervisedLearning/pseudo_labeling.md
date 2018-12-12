# PesudoLabeling

```python
seqlearner.SemiSupervisedLearner.naive_bayes(alg, sample_rate)
```

Pseudo Labeling method for the semi-supervised learning. This method will train a classifier algorithm for labeled sequences. Then, it will predict the labels of unlabeled sequences.

### Arguments
- __alg__: Scikit-learn Object, Sklearn classifier object to be used in training and prediction phase
- __sample_rate__: Float, The proportion of unlabeled sequences from X_t


### Example: predict the unlabeled sequences

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
- [PseudoLabeling code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/SemiSupervisedLearner.py)

