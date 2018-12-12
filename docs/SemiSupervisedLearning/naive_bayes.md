# NaiveBayes
```python
seqlearner.SemiSupervisedLearner.naive_bayes(distributions, verbose, max_iter, stop_threshold, pseudocount, weights)
```

Naive Bayesian algorithm for semi-supervised learning. 
Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

### Arguments
- __distributions__: Pomegranate Object, Distribution object from pomegranate package
- __verbose__: Boolean, For showing debug messages
- __max_iter__: Positive integer, The number of maximum iterations
- __stop_threshold__: Float, threshold for stop
- __pseudocount__: Float, A pseudocount to add to the emission of each distribution. This effectively smoothes the states to prevent 0. probability symbols if they don't happen to occur in the data. Only effects mixture models defined over discrete distributions
- __weights__: Array-like, The initial weights of each sample in the matrix. If nothing is passed in then each sample is assumed to be the same weight. Default is None.


### Example: predict the unlabeled sequences

```python
from sklearn.model_selection import train_test_split
from seqlearner import MultiTaskLearner
labeled_path = "../data/labeled.csv"
unlabeled_path = "../data/unlabeled.csv"
mtl = MultiTaskLearner(labeled_path, unlabeled_path)
encoding = mtl.embed(word_length=5)
X, y, X_t, y_t = train_test_split(mtl.sequences, mtl.labels, test_size=0.33)
score = mtl.semi_supervised_learner(X, y, X_t, y_t, ssl="naive_bayes", max_iter=1e8)
```

### See Also
- [NaiveBayes code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/SemiSupervisedLearner.py)

