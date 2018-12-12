import numpy as np
import pomegranate as pm
# from semisup.frameworks.CPLELearning import CPLELearningModel
from pomegranate import BayesClassifier, NaiveBayes
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from scipy.sparse import  csgraph
from .PseudoLabeler import PseudoLabeler


# from semisup.methods import scikitTSVM as tsvm


class SemiSupervisedLearner:
    """
        Wrapper class for semi-supervised learning methods. This class
        will apply a specific semi-supervised learning method on the
        sequences and return score for validation sequences.

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

        Returns
        -------
        score : the score of learning model on test data

        See Also
        -------
        SemiSupervisedLearner.label_spreading : Apply Label Spreading Algorithm
        SemiSupervisedLearner.label_propagation : Apply Label Propagation Algorithm
        SemiSupervisedLearner.naive_bayes : Apply Naive Bayes Algorithm
        SemiSupervisedLearner.pseudo_labeling : Apply Pseudo Labeling Algorithm

    """
    def __init__(self, X, y, X_t, y_t):
        self.X = X
        self.y = y
        self.X_t = X_t
        self.y_t = y_t

    def label_spreading(self, kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2,
                        max_iter=30, tol=0.001, n_jobs=1):
        """
            LabelSpreading model for semi-supervised learning
            This model is similar to the basic Label Propagation algorithm,
            but uses affinity matrix based on the normalized graph Laplacian
            and soft clamping across the labels.

            Parameters
            ----------
            kernel : {'knn', 'rbf', callable}
                String identifier for kernel function to use or the kernel function
                itself. Only 'rbf' and 'knn' strings are valid inputs. The function
                passed should take two inputs, each of shape [n_samples, n_features],
                and return a [n_samples, n_samples] shaped weight matrix

            gamma : float
              parameter for rbf kernel

            n_neighbors : integer > 0
              parameter for knn kernel

            alpha : float
              Clamping factor. A value in [0, 1] that specifies the relative amount
              that an instance should adopt the information from its neighbors as
              opposed to its initial label.
              alpha=0 means keeping the initial label information; alpha=1 means
              replacing all initial information.

            max_iter : integer
              maximum number of iterations allowed

            tol : float
              Convergence tolerance: threshold to consider the system at steady
              state

            n_jobs : int or None, optional (default=None)
                The number of parallel jobs to run.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.

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
        model = LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha,
                               max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        model.fit(self.X, self.y)
        return model.score(self.X_t, self.y_t)

    def label_propagation(self, kernel='rbf', gamma=20, n_neighbors=7,
                          max_iter=30, tol=1e-3, n_jobs=1):
        """
            Label Propagation classifier for semi-supervised learning

            Parameters
            ----------
            kernel : {'knn', 'rbf'}
                String identifier for kernel function to use or the kernel function
                itself. Only 'rbf' and 'knn' strings are valid inputs. The function
                passed should take two inputs, each of shape [n_samples, n_features],
                and return a [n_samples, n_samples] shaped weight matrix.

            gamma : float
                Parameter for rbf kernel

            n_neighbors : integer > 0
                Parameter for knn kernel

            alpha : float
                Clamping factor.


            max_iter : integer
                Change maximum number of iterations allowed

            tol : float
                Convergence tolerance: threshold to consider the system at steady
                state

            n_jobs : int or None, optional (default=None)
                The number of parallel jobs to run.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.

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
            >>> score = mtl.semi_supervised_learner(X, y, X_t, y_t, ssl="label_propagation")
        """
        model = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors,
                                 max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        model.fit(self.X, self.y)
        return model.score(self.X_t, self.y_t)

    def naive_bayes(self, distributions=pm.NormalDistribution, verbose=True, max_iter=1e8,
                    stop_threshold=0.1, pseudocount=0, weights=None):
        """
            Naive Bayesian algorithm for semi-supervised learning

            Parameters
            ----------
            distributions : object, pomegranate object
                Distribution object from pomegranate package
            verbose: boolean, default True
                for showing debug messages
            max_iter: integer, default 1e8
                The number of maximum iterations
            stop_threshold: float, 0.1
                threshold for stop.
            pseudocount: integer, default 0
                A pseudocount to add to the emission of each distribution. This
                effectively smoothes the states to prevent 0. probability symbols
                if they don't happen to occur in the data. Only effects mixture
                models defined over discrete distributions. Default is 0.
            weights: Array-Like
                The initial weights of each sample in the matrix. If nothing is
                passed in then each sample is assumed to be the same weight.
                Default is None.
        """
        model = NaiveBayes.from_samples(distributions=distributions, X=self.X, y=self.y, verbose=verbose,
                                        max_iterations=max_iter, stop_threshold=stop_threshold, pseudocount=pseudocount,
                                        weights=weights)
        return model.score(self.X_t, self.y_t)

    def bayes_classifier(self, distributions=pm.NormalDistribution,
                         max_iter=1e8,
                         stop_threshold=0.1):
        """
            Bayesian Classifier for semi-supervised learning

            Parameters
            ----------
            distributions : object, pomegranate object
                Distribution object from pomegranate package
            max_iter: integer, default 1e8
                The number of maximum iterations
            stop_threshold: float, 0.1
                threshold for stop.

        """
        model = BayesClassifier.from_samples(distributions=distributions, max_iterations=max_iter,
                                             stop_threshold=stop_threshold, X=self.X, y=self.y)
        return model.score(self.X_t, self.y_t)

    def GAN(self):
        pass

    def pseudo_labeling(self, alg, sample_rate=0.2):
        """
            Pseudo Labeling method for the semi-supervised learning.
            This method will train a classifier for labeled sequences.

            Parameters
            ----------
            alg : object, Sklearn object
               Sklearn classifier object like svm.SVC classifier.
            sample_rate: float, default 0.2
                The proportion of unlabeled sequences from X_t.

        """
        idx = np.where(self.y == -1)[0].tolist()
        X_u = self.X[idx]
        model = PseudoLabeler(model=alg, X_t=X_u, sample_rate=sample_rate)
        model.fit(self.X, self.y)
        return model.score(self.X_t, self.y_t)

    # def CPLE(self, model, sample_rate=0.5):
    #     num_of_samples = int(self.X_t.shape[0] * sample_rate)
    #     model.fit(self.X.sample(num_of_samples), self.y.sample(num_of_samples))
    #     ssmodel = CPLELearningModel(model)
    #     ssmodel.fit(self.X, self.y)
    #     return ssmodel.score(self.X_t, self.y_t)

    # def TSVM(self, kernel='RBF', C=1e-4, gamma=0.5, lamU=1.0, probability=True):
    #     model = tsvm.SKTSVM(kernel=kernel, C=C, gamma=gamma, lamU=lamU, probability=probability)
    #     model.fit(self.X, self.y)
    #     return model.score(self.X_t, self.y_t)
