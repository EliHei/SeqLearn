import numpy as np
import pandas as pd
import pomegranate as pm
from numpy import mean

from code.Embedding import Embedding
from code.SemiSupervisedLearner import SemiSupervisedLearner


class MultiTaskLearner:
    def __init__(self, file_labeled, file_unlabeled):
        self.data_labeled = pd.read_excel(file_labeled)
        self.data_unlabeled = pd.read_excel(file_unlabeled)
        self.data_labeled = self.data_labeled.sample(frac=1)
        self.data_unlabeled = self.data_unlabeled.sample(frac=1)
        self.seq_labeled = self.data_labeled['sequence'].tolist()
        self.seq_unlabeled = self.data_unlabeled['sequence'].tolist()
        self.labels = self.data_labeled['label'].tolist()
        self.labels = self.labels + [-1] * len(self.seq_unlabeled)
        self.classes = list(set(self.labels))
        self.classes.remove(-1)
        self.sequences = self.seq_labeled + self.seq_unlabeled

    def learner(self, word_length, k, embedding, ssl, **options):
        self.embedding = embedding
        self.ssl = ssl
        print(embedding, "*******", ssl)
        emb = self.__embedding(word_length, options)
        labelings = list(map(lambda x: self.__one_versus_all_maker(x), self.classes))
        idx = self.__cv(k)
        scores = list(map(lambda labels: mean(list(map(
            lambda i: self.__semi_supervised_learner(np.delete(np.stack(emb, axis=0), i, 0),
                                                     np.delete(np.array(labels), i, 0),
                                                     np.array(emb)[i], np.array(labels)[i], options), idx))),
                          labelings))
        return dict(zip(self.classes, scores))

    def __embedding(self, word_length, options):
        embed = Embedding(self.sequences, word_length)
        if self.embedding is "skipgram":
            embedding = embed.skipgram(
                func=options.get("func", "sum"),
                window_size=options.get("window_size", 10),
                emb_dim=options.get("emb_dim", 10),
                loss=options.get("loss", "mean_squared_error"),
                optimizer=options.get("optimizer", "adam")
            )
            return embedding

        elif self.embedding is "sent2vec":
            embedding = embed.sent2vec(
                emb_dim=options.get("emb_dim", 10),
                epoch=options.get("epoch", 1000),
                lr=options.get("lr", .2),
                wordNgrams=options.get("wordNgrams", 10),
                loss=options.get("loss", "ns"),
                neg=options.get("neg", 10),
                thread=options.get("thread", 20),
                t=options.get("t", 0.000005),
                dropoutK=options.get("dropoutK", 4),
                bucket=options.get("bucket", 400)
            )
            return embedding

        elif self.embedding is "load_embedding":
            embedding = embed.load_embedding(
                func=options.get("func", "sum"),
                file=options.get("file")
            )
            return embedding

    def __semi_supervised_learner(self, X, y, X_t, y_t, options):
        SSL = SemiSupervisedLearner(X, y, X_t, y_t)
        if self.ssl is "label_spreading":
            score = SSL.label_spreading(
                kernel=options.get("kernel", "rbf"),
                gamma=options.get("gamma", 20),
                n_neighbors=options.get("n_neighbors", 7),
                alpha=options.get("alpha", .2),
                max_iter=options.get("max_iter", 30),
                tol=options.get("tol", 0.001),
                n_jobs=1
            )
            return score
        elif self.ssl is "label_propagation":
            score = SSL.label_propagation(
                kernel=options.get("kernel", "rbf"),
                gamma=options.get("gamma", 20),
                n_neighbors=options.get("n_neighbors", 7),
                max_iter=options.get("max_iter", 30),
                tol=options.get("tol", 0.001),
                n_jobs=1
            )
            return score
        elif self.ssl is "naive_bayes":
            score = SSL.naive_bayes(
                distributions=options.get("distributions", pm.NormalDistribution),
                verbose=options.get("verbose", True),
                max_iter=options.get("max_iter", 1e8),
                stop_threshold=options.get("stop_threshold", .1),
                pseudocount=options.get("pseudocount", 0),
                weights=options.get("weights", None)
            )
            return score
        elif self.ssl is "bayes_classifier":
            score = SSL.bayes_classifier(
                distributions=options.get("distributions", pm.NormalDistribution)
            )
            return score
        elif self.ssl is "pseudo_labeling":
            score = SSL.pseudo_labeling(
                alg=options.get("alg"),
                sample_rate=options.get("sample_rate", .2)
            )
            return score
        # elif self.ssl is "CPLE":
        #     score = SSL.CPLE(
        #         model=options.get("model"),
        #         sample_rate=options.get("sample_rate", .2)
        #     )
        #     return score
        elif self.ssl is "TSVM":
            score = SSL.TSVM(
                kernel=options.get("kernel", 'RBF'),
                C=options.get("C", 1e-4),
                gamma=options.get("gamma", 0.5),
                lamU=options.get("lamU", 1.),
                probability=options.get("probability", True)
            )
            return score

    def __one_versus_all_maker(self, label):
        label_dict = {label: 1, -1: -1}
        return list(map(lambda l: label_dict.get(l, 0), self.labels))

    def __cv(self, k):
        def chunkify(lst, n):
            return [lst[i::n] for i in range(n)]

        return chunkify(range(len(self.seq_labeled)), k)
