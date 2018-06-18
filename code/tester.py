from sklearn.linear_model import SGDClassifier
from code.MultiTaskLearner import MultiTaskLearner
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
mtl = MultiTaskLearner("../data/labeled.xlsx", "../data/unlabeled.xlsx")
print(mtl.learner(3, 5, "load_embedding", "label_spreading", file="../data/protVec_100d_3grams.csv"))
