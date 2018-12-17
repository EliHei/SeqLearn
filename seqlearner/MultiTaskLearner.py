import os
import sys
from functools import reduce

import numpy as np
import pandas as pd
import pomegranate as pm
import umap
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from seqlearner.Embedding import Embedding
from seqlearner.SemiSupervisedLearner import SemiSupervisedLearner

sys.setrecursionlimit(35000)


class MultiTaskLearner:
    """
        MultiTaskLearner is class for do both embedding and semi-supervised task.
        It is a wrapper for Embedding and SemiSupervisedLearner.

        Parameters
        ----------
        file_labeled : basestring, numpy ndarray, pandas DataFrame
            labeled sequences' path or matrix of labeled sequences with labels
        file_unlabeled : basestring, numpy ndarray, pandas DataFrame
            unlabeled sequences' path or matrix of unlabeled sequences

        See also
        --------
        MultiTaskLearner.learner : computes embedding and label the sequences respectively
        MultiTaskLearner.embed : computes embedding with the specified embedding method
    """
    def __init__(self, file_labeled, file_unlabeled):
        self.file_labeled = file_labeled
        self.file_unlabeled = file_unlabeled
        if isinstance(file_labeled, str):
            if file_labeled.endswith(".csv"):
                self.data_labeled = pd.read_csv(file_labeled)
                self.data_unlabeled = pd.read_csv(file_unlabeled)
            elif file_labeled.endswith(".xlsx"):
                self.data_labeled = pd.read_excel(file_labeled)
                self.data_unlabeled = pd.read_excel(file_unlabeled)
        else:
            self.data_labeled = file_labeled
            self.data_unlabeled = file_unlabeled
        if self.data_labeled is None:
            raise Exception("Labeled data is None")
        else:
            self.data_labeled = self.data_labeled.sample(frac=1)
            self.seq_labeled = self.data_labeled['sequence'].tolist()
        if self.data_unlabeled is not None:
            self.data_unlabeled = self.data_unlabeled.sample(frac=1)
            self.seq_unlabeled = self.data_unlabeled['sequence'].tolist()
        else:
            self.data_unlabeled = None
            self.seq_unlabeled = []
        self.labels = self.data_labeled['label'].tolist()
        self.labels = self.labels + [-1] * len(self.seq_unlabeled)
        self.classes = list(set(self.labels))
        self.classes.remove(-1)
        self.sequences = self.seq_labeled + self.seq_unlabeled
        self.embedding_method = None
        self.ssl = None
        self.class_freq = None
        self.class_scores = None
        self.overall_score = 0

    def learner(self, word_length, k, embedding, ssl, **options):
        """
            Applies the specific embedding method with its hyper-parameters. Next, will train
            a semi-supervised learning method and label the unlabeled sequences. Finally will
            compute the scores of embedding predictions.

            Parameters
            ----------
            word_length : integer
                The length of words to be separated from each other in sequences
            k : integer
                Number of chunks in cross-validation
            embedding : basestring
                The embedding method for computing embedding the sequences
            ssl : basestring
                The semi-supervised learning method to be applied
            options : dict
                dict of hyper-parameters with values for embedding method and semi-supervised learning method

            Returns
            -------
            scores, overall_scores, frequencies: matrix of class scores, overall scores, each class sample frequency

            Example
            --------
            >>> labeled_path = "../data/labeled.csv"
            >>> unlabeled_path = "../data/unlabeled.csv"
            >>> mtl = MultiTaskLearner(labeled_path, unlabeled_path)
            >>> scores, overall_scores, class_freqs = mtl.learner(word_length=5, k=10, embedding="freq2vec", ssl="label_spreading")
        """
        self.embedding_method = embedding
        self.ssl = ssl
        self.func = options.get("func", None)
        print("Start embedding with " + embedding + " ...")
        self.embedding = self.embed(word_length, **options)
        print("Embedding has been finished!")
        if ssl is None:
            return
        print("Start Running Semi-Supervised task...")
        labelings = list(map(lambda x: self.__one_versus_all_maker(x), self.classes))
        idx = self.__cv(k)
        # print(np.array(idx))
        # print(np.stack(emb, axis=0).shape)
        # print(np.delete(np.stack(emb, axis=0), idx[0], 0).shape)
        # print(np.delete(np.array(labelings[0]), idx[0], 0).shape)
        # print(np.array(emb)[idx].shape, np.array(labelings[0])[idx].shape)
        scores = list(map(lambda labels: np.mean(list(map(
            lambda i: self.semi_supervised_learner(np.delete(np.stack(self.embedding, axis=0), i, 0),
                                                   np.delete(np.array(labels), i, 0),
                                                   np.array(self.embedding)[i],
                                                   np.array(labels)[i], options), idx))),
                          labelings))
        self.class_scores = dict(zip(self.classes, scores))
        self.__calc_overal_score()
        return self.class_scores, self.overall_score, self.class_freq

    def embed(self, word_length, **options):
        """
            Applies the embedding method on sequences with specified embedding method.

            Parameters
            ----------
            word_length : integer
                The length of words to be separated from each other in sequences
            options : dict
                dict of hyper-parameters with values for embedding method

            Returns
            -------
            encoding : list of embedding vectors for each sentence

            Example
            --------
            >>> labeled_path = "../data/labeled.csv"
            >>> unlabeled_path = "../data/unlabeled.csv"
            >>> mtl = MultiTaskLearner(labeled_path, unlabeled_path)
            >>> encoding = mtl.embed(word_length=5)
        """
        self.func = options.get("func", None)
        if self.embedding_method is None:
            if options.get("embedding", None) is None:
                raise Exception("The embedding must have specified")
            self.embedding_method = options.get("embedding", None)
        embed = Embedding(self.sequences, word_length)
        if self.embedding_method is "skipgram":
            self.embedding = embed.skipgram(
                func=options.get("func", "sum"),
                window_size=options.get("window_size", 10),
                emb_dim=options.get("emb_dim", 10),
                loss=options.get("loss", "mean_squared_error"),
                epochs=options.get("epochs", 100)
            )
            return self.embedding

        elif self.embedding_method is "freq2vec":
            self.embedding = embed.freq2vec(
                func=options.get("func", "sum"),
                window_size=options.get("window_size", 10),
                emb_dim=options.get("emb_dim", 10),
                loss=options.get("loss", "mean_squared_error"),
                epochs=options.get("epochs", 100)
            )
            return self.embedding

        elif self.embedding_method is "sent2vec":
            self.embedding = embed.sent2vec(
                emb_dim=options.get("emb_dim", 10),
                epochs=options.get("epoch", 1000),
                lr=options.get("lr", .2),
                wordNgrams=options.get("wordNgrams", 10),
                loss=options.get("loss", "ns"),
                neg=options.get("neg", 10),
                thread=options.get("thread", 20),
                t=options.get("t", 0.000005),
                dropoutK=options.get("dropoutK", 4),
                bucket=options.get("bucket", 400)
            )
            return self.embedding

        elif self.embedding_method is "load_embedding":
            self.embedding = embed.load_embedding(
                func=options.get("func", "sum"),
                file=options.get("file")
            )
            return self.embedding

        elif self.embedding_method is "word2vec":
            self.embedding = embed.word2vec(
                func=options.get("func", "sum"),
                window_size=options.get("window_size", 10),
                emb_dim=options.get("emb_dim", 10),
                workers=options.get("workers", 2),
                epochs=options.get("epochs", 100)
            )
            return self.embedding
        else:
            raise Exception("Embedding method is not valid.")

    def semi_supervised_learner(self, X, y, X_t, y_t, options):
        """
            Applies the semi-supervised learning method on sequences.

            Parameters
            ----------
            X : list, numpy ndarray, pandas DataFrame
                list of training embedding vectors for learning
            y : list, numpy ndarray, pandas DataFrame
                list of training labels.
            X_t : list, numpy ndarray, pandas DataFrame
                list of test embedding vectors for learning
            y_t : list, numpy ndarray, pandas DataFrame
                list of test labels.
            options : dict
                dict of hyper-parameters with values for semi-supervised method

            Returns
            -------
            score : the score of learning model on test data

            Example
            --------
            >>> labeled_path = "../data/labeled.csv"
            >>> unlabeled_path = "../data/unlabeled.csv"
            >>> mtl = MultiTaskLearner(labeled_path, unlabeled_path)
            >>> encoding = mtl.embed(word_length=5)
            >>> X, y, X_t, y_t = train_test_split(mtl.sequences, mtl.labels, test_size=0.33)
            >>> score = mtl.semi_supervised_learner(X, y, X_t, y_t, ssl="label_spreading")
        """
        SSL = SemiSupervisedLearner(X, y, X_t, y_t)
        if self.ssl is None:
            if options.get("ssl", None) is None:
                raise Exception("The semi-supervised learning method must specified")
            self.ssl = options.get("ssl", None)
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
            print(score)
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
            print(score)
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
            print(score)
            return score
        elif self.ssl is "bayes_classifier":
            score = SSL.bayes_classifier(
                distributions=options.get("distributions", pm.MultivariateGaussianDistribution),
                max_iter=options.get("max_iter", 1e8),
                stop_threshold=options.get("stop_threshold", .1)
            )
            print(score)
            return score
        elif self.ssl is "pseudo_labeling":
            score = SSL.pseudo_labeling(
                alg=options.get("alg"),
                sample_rate=options.get("sample_rate", .2)
            )
            print(score)
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

    def __one_versus_all_maker_v2(self, label):
        df = pd.DataFrame({"label": self.labels})
        df[df["label"] != label] = -1
        df[df["label"] == label] = 1
        if df[df["label"] == 1].shape[0] > 10:
            print('and'.join(label.split('&')), end=" & ")
        return df

    def __train_test_split_stratified(self, x_data, y_data):
        n_negative_samples = y_data[y_data["label"] == -1].shape[0]
        n_positive_samples = y_data[y_data["label"] == 1].shape[0]
        select_idx = np.random.choice(n_negative_samples, n_positive_samples).tolist()
        negative_samples = y_data[y_data["label"] == -1].iloc[select_idx, :]
        positive_samples = y_data[y_data["label"] == 1]
        x_data = x_data[positive_samples.index.tolist() + select_idx]
        y_data = pd.concat([negative_samples, positive_samples])
        x_data = self.__umap(x_data)
        return train_test_split(x_data, y_data, stratify=y_data, shuffle=True, test_size=0.3)

    def __cv(self, k):
        def chunkify(lst, n):
            return [lst[i::n] for i in range(n)]

        return chunkify(range(len(self.seq_labeled)), k)

    def __label_freq(self):
        def adder(label):
            try:
                self.class_freq[label] += 1
            except KeyError:
                return 0

        self.class_freq = dict.fromkeys(self.classes, 0)
        list(map(lambda label: adder(label), self.labels))
        # self.class_freq = {k: v / len(self.seq_labeled) for k, v in self.class_freq.items()}

    def __calc_overal_score(self):
        self.__label_freq()
        print(self.class_scores)
        print(self.class_freq)
        print("*******")
        # self.overal_score = 0
        # for i in self.classes:
        #     self.overal_score = self.class_freq[i] * self.class_scores[i]
        normalized_scores = [self.class_freq[c] * self.class_scores[c] for c in self.classes]
        self.overall_score = reduce(lambda x, y: x + y, normalized_scores)

    def classify(self, embedding_path="../data/uniprot/", embedding="Freq2Vec", method="SVM",
                 func="weighted_average"):
        def learn(x_data, label_name, classifier):
            y_data = self.__one_versus_all_maker_v2(label_name)
            x_train, x_test, y_train, y_test = self.__train_test_split_stratified(x_data, y_data)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            score = classifier.score(x_test, y_test)
            # print("%.2f\\%% & %.4f & %.4f\\\\" % (score * 100, sensitivity, specificity))
            return score, sensitivity, specificity

        embedding_features = pd.read_csv(embedding_path + embedding + "_" + func + "_Encoding.csv", header=None,
                                         delimiter=',')
        labelings = list(map(lambda x: self.__one_versus_all_maker(x), self.classes))
        x_data = embedding_features.values[:len(self.sequences)]
        if method is "SVM":
            classifier = SVC()
        elif method is "QDA":
            classifier = QuadraticDiscriminantAnalysis()
        elif method is "RandomForest":
            classifier = RandomForestClassifier()
        elif method is "GradientBoosting":
            classifier = GradientBoostingClassifier()
        else:
            classifier = KNeighborsClassifier(5)
        # classes = pd.DataFrame({"label": self.labels})["label"].value_counts().index.tolist()
        # random_classes = ['Frataxin family',
        #                   'Cytochrome b family, PetB subfamily',
        #                   'MscL family',
        #                   'Snaclec family',
        #                   'Transcriptional regulatory CopG/NikR family',
        #                   'Endonuclease V family',
        #                   'SfsA family',
        #                   'SpoVG family',
        #                   'Oligoribonuclease family',
        #                   'Sodium:solute symporter (SSF) (TC 2.A.21) family',
        #                   'Hfq family',
        #                   'UreE family',
        #                   'TRAFAC class YlqF/YawG GTPase family, RsgA subfamily',
        #                   'V-ATPase E subunit family',
        #                   'PsbB/PsbC family, PsbB subfamily',
        #                   'Dethiobiotin synthetase family',
        #                   'Acyl-CoA dehydrogenase family',
        #                   'PPR family, P subfamily',
        #                   'Protein-tyrosine phosphatase family',
        #                   'ScpA family',
        #                   'Group II decarboxylase family',
        #                   'SNF2/RAD54 helicase family',
        #                   'HSP33 family',
        #                   'Somatotropin/prolactin family',
        #                   'DEFL family',
        #                   'YciB family',
        #                   'Peptidase C15 family']
        scores = np.array(list(map(lambda label: learn(x_data, label, classifier), labelings)))
        return pd.DataFrame(
            {"Accuracy": scores[:, 0], "Sensitivity": scores[:, 1], "Specificity": scores[:, 2]}), self.classes

    def __umap(self, embedding_weights, **options):
        return umap.UMAP(n_components=options.get("n_components", 2), metric="correlation").fit_transform(
            embedding_weights)

    def __tsne(self, embedding_weights, **options):
        return TSNE(n_components=options.get("n_components", 2)).fit_transform(embedding_weights)

    def __get_components_from_visualization_method(self, method="TSNE.md"):
        if method is "TSNE.md":
            return self.__tsne()
        elif method is "UMAP":
            return self.__umap()

    def visualize(self, method="TSNE", family=None, proportion=1.5):
        if family is None:
            protein_family = str(pd.Series(self.labels).value_counts().index[0])
            protein_family_freq = pd.Series(self.labels).value_counts().iloc[0]
        else:
            protein_family = family
            protein_family_freq = pd.Series(self.labels).value_counts().loc[family]
        random_samples = np.random.choice(len(self.labels), int(proportion * protein_family_freq), replace=False)
        embedding_weights = pd.DataFrame(self.embedding)
        embedding_weights = pd.concat(
            [embedding_weights.iloc[random_samples], embedding_weights.iloc[random_samples]],
            axis=0)
        protein_families = pd.DataFrame(self.labels, columns=["label"])
        p_samples = protein_families[protein_families["label"] == protein_family].index.tolist()
        protein_families = pd.concat([protein_families.iloc[random_samples], protein_families.iloc[p_samples]],
                                     axis=0)
        protein_families = pd.DataFrame(protein_families).reset_index()
        if method == "TSNE":
            tsne = TSNE(n_components=2)
            embedding = tsne.fit_transform(embedding_weights)
        else:
            embedding = umap.UMAP(n_components=2, metric="correlation").fit_transform(embedding_weights)
        embedding = pd.DataFrame(embedding)
        embedding["label"] = protein_families["label"]
        pfam_emb = embedding.loc[embedding["label"] == protein_family, [0, 1]]
        pfam_emb_others = embedding.loc[embedding["label"] != protein_family, [0, 1]]
        if protein_family.__contains__("/"):
            protein_family = '-'.join(protein_family.split('/'))
        self.__plot([pfam_emb[0], pfam_emb_others[0]], [pfam_emb[1], pfam_emb_others[1]],
                    [protein_family, "Others"], method)

    def __plot(self, xs, ys, labels, method="tsne"):
        path = "../results/visualization/%s/%s/%s/" % (self.embedding_method, self.func, labels[0])
        os.makedirs(path, exist_ok=True)
        plt.close("all")
        plt.figure(figsize=(30, 30))
        plt.plot(xs[0], ys[0], 'ro', label=labels[0], markersize=14)
        plt.legend()
        plt.savefig(path + "%s_%s_%s.pdf" % (method, labels[0], self.func))
        plt.close("all")
        plt.figure(figsize=(30, 30))
        plt.plot(xs[1], ys[1], 'bo', label=labels[1], markersize=14)
        plt.legend()
        plt.savefig(path + "%s_%s_%s.pdf" % (method, labels[1], self.func))
        plt.close("all")
        plt.figure(figsize=(30, 30))
        plt.plot(xs[0], ys[0], 'o', label=labels[0], markersize=14)
        plt.plot(xs[1], ys[1], 'o', label=labels[1], markersize=14)
        plt.legend()
        plt.savefig(path + "%s_%s.pdf" % (method, self.func))
        print("The plot has been saved in " + path)
