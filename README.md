# SeqLearner [![PyPI version](https://badge.fury.io/py/seqlearner.svg)](https://badge.fury.io/py/seqlearner) [![Build Status](https://travis-ci.org/EliHei/SeqLearn.svg?branch=master)](https://travis-ci.org/EliHei/SeqLearn) [![Documentation Status](https://readthedocs.org/projects/seqlearner/badge/?version=latest)](https://seqlearner.readthedocs.io/en/latest/?badge=latest)


The multitask learning package for semi-supervised learning on biological sequences

<div float="left">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png" height="120" >
  <img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" height="120">
</div>
<div float="right">
</div>

## Introduction
A tensorflow implementation of multitask learning package for semi-supervised learning on biological sequences


## Getting Started

## Installation
```python
pip install seqlearner
```


## File Illustration
This repo is divided into 3 directories.
 1. The `seqlearner` directory contains all codes and jupyter notebooks.
 2. The `seqlearner/data/` directory is place where data is in.
 3. The `seqlearner/results/` directory contains all results plots, Logs and etc.


## Examples
After Embedding the protein sequences with embedding methods, we provide some visualization for it. TSNE and UMAP have been used for visualizing embedding of 2 protein families to gather some evaluation about the embedding.
With this evaluation we want to give some intuition about how well protein families are seperated via this embedding and the corresponding function.

Here is a simple example for calculating the embedding using `Freq2Vec` and visualize it via `TSNE` method.
First, you have to calculate and save the embedding via `learner` method.
```python
freq2vec_embedding = mtl.embed(word_length=3, embedding="freq2vec", func="sum", emb_dim=25, gamma=0.1, epochs=100)
```
after calculating the freq2vec embedding with 25 dimensions, we would like to visualize it via `TSNE` method.
```python
visualize(method="TSNE", proportion=2.0)
```
This will save a plot for you in `seqlearner/results/` folder which the points are samples from 2 protein families which has the most samples in the dataset. Here is a sample plot.

<div align="center">
	<img src="./seqlearner/results/visualization/Freq2Vec/weighted_average/DEFL family/UMAP_weighted_average.png" width="80%" />
</div>
