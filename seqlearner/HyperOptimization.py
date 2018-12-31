from __future__ import print_function

from hyperopt import Trials, tpe, fmin
from hyperopt.hp import choice
from sklearn import svm

from seqlearner.MultiTaskLearner import MultiTaskLearner

"""
    Created by Mohsen Naghipourfar on 11/13/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def hyper_optimize(space):
    embedding_method = space["embedding_method"]
    labeled_sequence_path = space["labeled_sequence_path"]
    unlabeled_sequence_path = space["unlabeled_sequence_path"]

    emb_dim = space["emb_dim"]
    word_length = space["word_length"]
    k = space["k"]
    epochs = space["n_epochs"]
    func = space["func"]
    window_size = space["window_size"]
    lr = space["lr"]
    wordNgrams = space["wordNgrams"]
    gamma = space["gamma"]
    ssl = space["ssl"]

    options = {}
    if embedding_method.lower() == "freq2vec":
        options = {"func": func,
                   "emb_dim": emb_dim,
                   "gamma": gamma,
                   "epochs": epochs}
    elif embedding_method.lower() == "sent2vec":
        options = {"func": func,
                   "emb_dim": emb_dim,
                   "lr": lr,
                   "wordNgrams": wordNgrams}
    elif embedding_method.lower() == "skipgram":
        options = {"func": func,
                   "emb_dim": emb_dim,
                   "window_size": window_size}
    elif embedding_method.lower() == "word2vec":
        options = {"func": func,
                   "emb_dim": emb_dim,
                   "window_size": window_size}
    options["alg"] = svm.SVC(C=1.0, cache_size=200, class_weight=None,
                             coef0=0.0,
                             decision_function_shape='ovr', degree=3,
                             gamma='auto', kernel='rbf',
                             max_iter=-1, probability=False,
                             random_state=None, shrinking=True,
                             tol=0.001, verbose=False)

    mtl = MultiTaskLearner(labeled_sequence_path, unlabeled_sequence_path)
    class_scores, overall_score, class_freq = mtl.learner(word_length=word_length, k=k, embedding=embedding_method,
                                                          ssl=ssl,
                                                          **options)
    return overall_score


def optimize(embedding_method=None, labeled_sequence=None, unlabeled_sequence=None):
    if embedding_method is None:
        raise Exception("Embedding method must be specified.")

    hyper_parameter_space = {
        "embedding_method": embedding_method,
        "labeled_sequence_path": labeled_sequence,
        "unlabeled_sequence_path": unlabeled_sequence,
        "ssl": choice("ssl",
                      ["label_spreading", "label_propagation", "naive_bayes", "bayes_classifier", "pseudo_labeling"]),
        "emb_dim": choice("emb_dim", [5 * i for i in range(1, 21)]),
        "word_length": choice("word_length", [i for i in range(3, 4)]),
        "k": choice("k", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "n_epochs": choice("n_epochs", [50, 100, 150, 200]),
        "func": choice("func", ["sum", "weighted_sum", "average", "weighted_average"]),
        "window_size": choice("window_size", [5, 10, 15]),
        "lr": choice("lr", [.1, .2, 1e-2, 1e-3]),
        "wordNgrams": choice("wordNgrams", [i for i in range(5, 16)]),
        "gamma": choice("gamma", [0.1, 0.2, 0.3, 0.4, 0.5])
    }
    best_run, best_model = fmin(fn=hyper_optimize,
                                space=hyper_parameter_space,
                                algo=tpe.suggest,
                                max_evals=5,
                                trials=Trials())
    print("Evalutation of best performing model:")
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


if __name__ == '__main__':
    data_path = "./seqlearner/data/mohimani/"
    labeled_sequences_path = data_path + "labeled.xlsx"
    unlabeled_sequences_path = data_path + "unlabeled.xlsx"

    optimize(embedding_method="Freq2Vec", labeled_sequence=labeled_sequences_path,
             unlabeled_sequence=unlabeled_sequences_path)
