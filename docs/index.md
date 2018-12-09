# SeqLearner
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

## Getting Started
