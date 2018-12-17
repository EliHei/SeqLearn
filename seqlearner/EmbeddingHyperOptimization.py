from __future__ import print_function

import pandas as pd
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, tpe, STATUS_OK

from seqlearner.MultiTaskLearner import MultiTaskLearner

"""
    Created by Mohsen Naghipourfar on 11/13/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""
embedding_method = None
sequence_data = None

def load_data():
    dataset_name = "uniprot/"
    data_path = "./data/"
    labeled_data = pd.read_csv(data_path + dataset_name + "labeled.csv", index_col="Unnamed: 0")
    unlabeled_data = pd.read_csv(data_path + dataset_name + "unlabeled.csv", index_col="Unnamed: 0")
    return labeled_data, unlabeled_data


def embed():
    global embedding_method, sequence_data
    embedding = embedding_method
    if embedding is None:
        raise Exception("Embedding method must be specified.")
    labeled_data = sequence_data
    unlabeled_data = sequence_data
    emb_dim = {{choice([5 * i for i in range(1, 21)])}}
    word_length = {{choice([i for i in range(3, 11)])}}
    k = {{choice([2, 3, 4, 5, 6, 7, 8, 9, 10])}}
    epoch = {{choice([50])}}
    func = {{choice(["sum", "weighted_sum", "average", "weighted_average"])}}
    window_size = {{choice([5, 10, 15])}}
    lr = {{choice([.1, .2, 1e-2, 1e-3])}}
    wordNgrams = {{choice([i for i in range(5, 16)])}}
    options = {}
    if embedding is "freq2vec":
        options = {"func": func, "emb_dim": emb_dim,
                   "gamma": 0.1, "epochs": epoch}
    elif embedding is "sent2vec":
        options = {"func": func, "emb_dim": emb_dim, "lr": lr, "wordNgrams": wordNgrams}
    elif embedding is "skipgram":
        options = {"func": func, "emb_dim": emb_dim, "window_size": window_size}
    elif options is "word2vec":
        options = {"func": func, "emb_dim": emb_dim, "window_size": window_size}

    mtl = MultiTaskLearner(labeled_data, unlabeled_data)
    mtl.learner(word_length=word_length, k=k, embedding=embedding, ssl=None, **options)

    return {'status': STATUS_OK}


def optimize(embedding=None, sequence_datapath=None):
    global embedding_method, sequence_data
    sequence_data = sequence_datapath
    embedding_method = embedding
    best_run, best_model = optim.minimize(model=embed,
                                          data=load_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
