import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import shuffle


class PseudoLabeler(BaseEstimator, RegressorMixin):
    """
        PseudoLabeler class for the semi-supervised task.
        This class will train a classifier for labeled sequences. PseudoLabeler
        will evaluate the classification accuracy with labeled sequences.
        After training to labeled dataset, It will label the unlabeled sequences.

        Parameters
        ----------
        model : object, Sklearn object
           Sklearn classifier object like svm.SVC classifier.
        X_t : list, numpy ndarray
            The whole sequences. labeled and unlabeled sequences are in this list.
        sample_rate: float, default 0.2
            The proportion of unlabeled sequences from X_t.

        See also
        --------
        PseudoLabeler.fit : trains the classifier
        PseudoLabeler.predict : predicts the unlabeled validation data class
        PseudoLabeler.score : get the score of prediction

    """
    def __init__(self, model, X_t, sample_rate=0.2):
        self.sample_rate = sample_rate
        self.model = model
        self.X_t = X_t

    def fit(self, X, y):
        """
            trains the classifier `model` to the training data.

            Parameters
            ----------
            X : list, numpy ndarray, pd.DataFrame
                Sequences or features to be labeled
            y : list, numpy ndarray, pd.DataFrame
                Labels(Classes) of X.

            Returns
            -------
            self : The current object of PseudoLabeler

            Example
            --------
            >>> from sklearn.svm import SVC
            >>> X = pd.read_csv("./sequences.csv", header=None)
            >>> y = pd.read_csv("./labels.csv", header=None)
            >>> alg = SVC(C=1.0, cache_size=200, class_weight=None)
            >>> model = PseudoLabeler(model=alg, X_t=X, sample_rate=0.2)
            >>> model.fit(X, y)
        """
        if self.sample_rate > 0.0:
            augmented_train = self.__create_augmented_train(X, y)
            self.model.fit(
                pd.DataFrame.as_matrix(augmented_train.iloc[:, :(augmented_train.shape[1] - 1)]),
                pd.DataFrame.as_matrix(augmented_train.iloc[:, (augmented_train.shape[1] - 1)])
            )
        else:
            self.model.fit(X, y)
        return self

    def __create_augmented_train(self, X, y):
        num_of_samples = int(self.X_t.shape[0] * self.sample_rate)
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.X_t)
        augmented_test = pd.concat([pd.DataFrame(self.X_t), pd.DataFrame(pseudo_labels)], axis=1)
        sampled_test = augmented_test.sample(n=num_of_samples)
        temp_train = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
        augmented_train = pd.concat([sampled_test, temp_train])
        return shuffle(augmented_train)

    def predict(self, X):
        """
            Predicts the label(class) of sequences in list `X`.

            Parameters
            ----------
            X : list, numpy ndarray, pd.DataFrame
                Sequences or features to be labeled

            Returns
            -------
            prediction : list of predicted labels(classes) for the sequences in `X`

            Example
            --------
            >>> from sklearn.svm import SVC
            >>> X = pd.read_csv("./sequences.csv", header=None)
            >>> y = pd.read_csv("./labels.csv", header=None)
            >>> X_t = pd.read_csv("./unlabeled_sequences.csv", header=None)
            >>> alg = SVC(C=1.0, cache_size=200, class_weight=None)
            >>> model = PseudoLabeler(model=alg, X_t=X, sample_rate=0.2)
            >>> model.fit(X, y)
            >>> predicted_labels = model.predict(X_t)
        """
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        """
            Predicts the label(class) of sequences in list `X`.

            Parameters
            ----------
            X : list, numpy ndarray, pd.DataFrame
                Sequences or features to be labeled
            y : list, numpy ndarray, pd.DataFrame
                Labels(Classes) of X.
            sample_weight: list, numpy ndarray, pd.DataFrame
                List of weights for each sequence.

            Returns
            -------
            score : float, The classifier accuracy in prediction for sequences in `X` which is evaluated by `y`.
                This is a float number between [0, 1].

            Example
            --------
            >>> from sklearn.svm import SVC
            >>> X = pd.read_csv("./sequences.csv", header=None)
            >>> y = pd.read_csv("./labels.csv", header=None)
            >>> alg = SVC(C=1.0, cache_size=200, class_weight=None)
            >>> model = PseudoLabeler(model=alg, X_t=X, sample_rate=0.2)
            >>> model.fit(X, y)
            >>> score = model.score(X, y)
        """
        return self.model.score(X, y, sample_weight)
