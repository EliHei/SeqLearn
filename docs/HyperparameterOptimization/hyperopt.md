# HyperParameter Optimization
```python
seqlearner.EmbeddingHyperOptimization.optmize(embedding, sequence_datapath)
```

Hyper-parameter optimization for an embedding method implementation method. 
You can specify the embedding method to function `optimize` and the best choice for parameters.
The `optimize` function takes the following arguments:
  
### Arguments
- __embedding__: String, Embedding method which its hyper-parameters are going to be optimized
- __sequence_datapath__: String, sequences file path

The `optmize` function returns a dictionary of hyperparamters and their best value for the corresponding hyperparameters.

### Example: Hyperparameter optimization for Freq2Vec

```python
from seqlearner import EmbeddingHyperOptimization as eho
labeled_path = "../data/labeled.csv"
unlabeled_path = "../data/unlabeled.csv"
best_parameters = eho.optimize(embedding="freq2vec")
print(best_parameters)
```

This will print the dictionary of best values for each hyper-parameter attending in Freq2Vec embedding method.

### See Also
- [EmbeddingHyperparameterOptmization code implementation on Github](https://github.com/EliHei/seqlearn/blob/master/seqlearner/EmbeddingHyperOptimization.py)

