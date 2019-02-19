## Installation
- __Install SeqLearner from PyPI (recommended)__:

The easiest way to get SeqLearner is through pip using the following command:
```python
sudo pip install seqlearner
```
If you are using a virtualenv, you may want to avoid using sudo:
```python
pip install seqlearner
```
This should install all the dependencies in addition to the package.

- __Alternatively: install SeqLearner from the GitHub source:__

You can also get SeqLearner from Github using the following steps:
First, clone SeqLearner using `git`:

```python
git clone https://github.com/EliHei/SeqLearn
```

Then, `cd` to the SeqLearner folder and run the install command:
```python
cd SeqLearn
python setup.py install
```

On Windows machines you may need to download a C++ compiler if you wish to build from source yourself. 

## Dependencies
The requirements for SeqLearner can be found in the requirements.txt file in the repository, and include numpy, pandas, tensorflow, keras, gensim, pomegranate, and matplotlib.

- [__numpy__](http://numpy.org): The fundamental package for scientific computing.

- [__pandas__](https://pandas.pydata.org): The library which provides high-performance, easy-to-use data structures and data analysis tools for the Python.


- [__tensorflow__](https://www.tensorflow.org): The library for high performance numerical computation.

- [__keras__](https://keras.io): Keras is a high-level neural networks API.

- [__gensim__](https://radimrehurek.com/gensim/index.html): Tools for Scalable statistical semantics.

- [__pomegranate__](https://pomegranate.readthedocs.io/en/latest/): A Python package that implements fast and flexible probabilistic models.

- [__matplotlib__](https://matplotlib.org): a Python 2D plotting library which produces publication quality figures

