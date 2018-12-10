# SeqLearner
[![PyPI version](https://badge.fury.io/py/seqlearner.svg)](https://badge.fury.io/py/seqlearner) [![Build Status](https://travis-ci.org/EliHei/SeqLearn.svg?branch=master)](https://travis-ci.org/EliHei/SeqLearn) [![Documentation Status](https://readthedocs.org/projects/seqlearner/badge/?version=latest)](https://seqlearner.readthedocs.io/en/latest/?badge=latest)

![](logo_small.png)

## SeqLearner is the Sequence Learner!
SeqLearner is a multitask learning package for semi-supervised learning on biological sequences
SeqLearner is a high-level API, written in Python and capable of running on different embedding methods such as Freq2Vec, Word2Vec, Sent2Vec and etc. It also provides some visualizations to analyze the embedding.
It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

SeqLearner is compatible with: Python 3.6-3.7.

## Main Principles
SeqLearner has some main principles:
  
- __User Friendly__: SeqLearner is an API designed for human beings, not machines. SeqLearner offers consistent & simple APIs, it minimizes the number of user actions required for a common use case, and it provides clear feedback upon user error.

- __Modularity__: A model is understood as a sequence or a graph of standalone modules that can be plugged together with as few restrictions as possible. In particular, embedding methods, semi-supervised algorithms schemes are all standalone modules that you can combine to create your own new model.

- __extensibility__: It's very simple to add new modules, and existing modules provide examples. To be able to easily create new modules allows SeqLearner suitable for advanced research.

- __Python Implementation__: All models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

## Getting Started: A Simple Example
Here is a simple example to make a Freq2Vec embedding and use it for labeling the unlabeled data using semi-supervised task.

```python
from seqlearner import MultiTaskLearner
from sklearn import svm
mtl = MultiTaskLearner("labeled.csv", "unlabeled.csv")
results = mtl.learner(3, 5, "freq2vec", "pseudo_labeling", emb_dim=50, epochs=250,
                alg=svm.SVC(C=1.0, kernel='rbf', max_iter=50))                            
```

## Support
Please feel free to ask questions:

- [Elyas Heidari](mailto:almasmadani@gmail.com)

- [Mohsen Naghipourfar](mailto:mn7697np@gmail.com)

You can also post bug reports and feature requests in [GitHub issues](https://github.com/EliHei/SeqLearn/issues). Please Make sure to read our guidelines first.

