import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import shuffle


class PseudoLabeler(BaseEstimator, RegressorMixin):
    def __init__(self, model, X_t, sample_rate=0.2):
        self.sample_rate = sample_rate
        self.model = model
        self.X_t = X_t

    def fit(self, X, y):
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
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        return self.model.score(X, y, sample_weight)
