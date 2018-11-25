import numpy as np
import pomegranate as pm
# from semisup.frameworks.CPLELearning import CPLELearningModel
from pomegranate import BayesClassifier, NaiveBayes
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from scipy.sparse import  csgraph
from .PseudoLabeler import PseudoLabeler


# from semisup.methods import scikitTSVM as tsvm


class SemiSupervisedLearner:
    def __init__(self, X, y, X_t, y_t):
        self.X = X
        self.y = y
        self.X_t = X_t
        self.y_t = y_t

    def label_spreading(self, kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2,
                        max_iter=30, tol=0.001, n_jobs=1):
        model = LabelSpreading(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha,
                               max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        model.fit(self.X, self.y)
        return model.score(self.X_t, self.y_t)

    def label_propagation(self, kernel='rbf', gamma=20, n_neighbors=7,
                          max_iter=30, tol=1e-3, n_jobs=1):
        model = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors,
                                 max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        model.fit(self.X, self.y)
        return model.score(self.X_t, self.y_t)

    def naive_bayes(self, distributions=pm.NormalDistribution, verbose=True, max_iter=1e8,
                    stop_threshold=0.1, pseudocount=0, weights=None):
        model = NaiveBayes.from_samples(distributions=distributions, X=self.X, y=self.y, verbose=verbose,
                                        max_iterations=max_iter, stop_threshold=stop_threshold, pseudocount=pseudocount,
                                        weights=weights)
        return model.score(self.X_t, self.y_t)

    def bayes_classifier(self, distributions=pm.NormalDistribution,
                         max_iter=1e8,
                         stop_threshold=.1):
        model = BayesClassifier.from_samples(distributions=distributions, max_iterations=max_iter,
                                             stop_threshold=stop_threshold, X=self.X, y=self.y)
        return model.score(self.X_t, self.y_t)

    def GAN(self):
        pass

    def pseudo_labeling(self, alg, sample_rate=0.2):
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
