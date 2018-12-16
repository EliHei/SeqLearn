import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.svm as svm
import umap
from sklearn.manifold import TSNE

from seqlearner.MultiTaskLearner import MultiTaskLearner

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data_path = "../data/uniprot/"


def preprocess():
    data_filename = "uniprot.tab"

    data = pd.read_csv(data_path + data_filename, delimiter="\t")

    data.drop(["Entry", "Protein names", "Cross-reference (Pfam)"], axis=1, inplace=True)

    data.dropna(axis=0, how='any', inplace=True)

    n_samples = data.shape[0]
    print(data["Protein families"].value_counts().shape[0])
    labeled_data = data.iloc[:n_samples // 100, :]
    unlabeled_data = data.iloc[n_samples // 100: n_samples // 100 + 1, :]

    pd.DataFrame.to_csv(labeled_data, data_path + "labeled.csv", header=["sequence", "label"])
    pd.DataFrame.to_csv(pd.DataFrame(unlabeled_data["Sequence"]), data_path + "unlabeled.csv",
                        header=["sequence"])

    print(labeled_data.shape)
    print(unlabeled_data.shape)


def train():
    np.seterr(divide='ignore', invalid='ignore')
    mtl = MultiTaskLearner(data_path + "labeled.csv", data_path + "unlabeled.csv")

    # freq2vec_embedding = mtl.embed(word_length=3, embedding="freq2vec", func="weighted_average", emb_dim=25, gamma=0.1,
    #                                epochs=1)
    # mtl.visualize(method="TNSE", family="ATCase/OTCase family", proportion=2.0)
    # mtl.visualize(method="UMAP", family="ATCase/OTCase family", proportion=2.0)
    # class_scores, overall_score, class_freqs = mtl.learner(3, 5, "freq2vec", "label_spreading", func="sum", emb_dim=2,
    #                                                        gamma=0.1, epochs=1)

    # mtl.learner(3, 5, "load_embedding", "pseudo_labeling",
    #             file="../data/uniprot/freq2vec_embedding_20_10_3.txt",
    #             func="average", sample_rate=0.0,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))
    #
    # mtl.learner(3, 5, "load_embedding", "pseudo_labeling",
    #             file="../data/uniprot/skipgram_embedding_50_10_3.txt",
    #             func="sum", sample_rate=0.0,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))
    # mtl.learner(3, 5, "load_embedding", "pseudo_labeling",
    #             file="../data/skipgram_embedding_50_10_3.txt",
    #             func="weighted_sum", sample_rate=0.0,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))

    mtl.learner(3, 5, "sent2vec", "pseudo_labeling", emb_dim=50,
                file="../data/skipgram_embedding_50_10_3.txt",
                func="weighted_average", sample_rate=0.0,
                alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
                            coef0=0.0,
                            decision_function_shape='ovr', degree=3,
                            gamma='auto', kernel='rbf',
                            max_iter=-1, probability=False,
                            random_state=None, shrinking=True,
                            tol=0.001, verbose=False))
    #
    # mtl.learner(3, 5, "load_embedding", "pseudo_labeling",
    #             file="../data/skipgram_embedding_50_10_3.txt",
    #             func="average", sample_rate=0.0,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))

    # mtl.learner(3, 5, "sent2vec", "pseudo_labeling",
    #             func="sum", sample_rate=0.0, emb_dim=50,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))

    # mtl.learner(3, 5, "sent2vec", "pseudo_labeling",
    #             file="../data/uniprot/freq2vec_embedding_20_10_3.txt",
    #             func="sum", sample_rate=0.0, emb_dim=50,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))
    #
    # mtl.learner(3, 5, "word2vec", "pseudo_labeling",
    #             func="sum", sample_rate=0.0, emb_dim=50,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))
    #
    # mtl.learner(3, 5, "skipgram", "pseudo_labeling",
    #             func="sum", sample_rate=0.0, emb_dim=50,
    #             alg=svm.SVC(C=1.0, cache_size=200, class_weight=None,
    #                         coef0=0.0,
    #                         decision_function_shape='ovr', degree=3,
    #                         gamma='auto', kernel='rbf',
    #                         max_iter=-1, probability=False,
    #                         random_state=None, shrinking=True,
    #                         tol=0.001, verbose=False))

    # print(overall_score)
    # print(class_scores)
    # class_scores = pd.DataFrame(class_scores, index=['accuracy'])
    # class_scores.loc['accuracy', :] *= 100.0
    # class_freqs = pd.DataFrame(class_freqs, index=['frequency'], dtype=np.int64)
    # results = pd.concat([class_scores, class_freqs], axis=0)
    # results = results.transpose()
    # print(results.shape)
    # make_table(results)
    # results.to_csv("../uniprot_scores_3_5_freq2Vec_label_spreading_20_sum_0.1_.csv")


def make_table(table):
    print("\\begin{table}[h!]")
    print("\\resizebox{\columnwidth}{!}{")
    print("\\centering")
    print("\\begin{tabular}{||c c c||}")
    print("\\hline")
    print("Protein Family & Frequency & Accuracy \\\\ [0.5ex]")
    print("\\hline\\hline")

    for protein_family in table.transpose().columns:
        print("%s & %d & %.4f " % (
            protein_family, table.loc[protein_family, 'frequency'], table.loc[protein_family, 'accuracy']) + "\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("}")
    print("\caption{%s}" % "Results for Uniprot Dataset")
    print("\\end{table}")
    print(end="\n\n\n\n")


def visualization(embedding_path="../data/uniprot/LoadEmbedding_Encoding.csv", func="sum", method="TSNE.md"):
    data = pd.read_csv(data_path + "uniprot.tab", delimiter="\t")
    data.drop(["Entry", "Protein names", "Cross-reference (Pfam)"], axis=1, inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data.reset_index(inplace=True)
    data_aux = data
    protein_families_unique = data["Protein families"].value_counts()
    for idx in range(300, protein_families_unique.shape[0], 25):
        protein_families = data_aux["Protein families"]
        protein_family = str(protein_families_unique.index[idx])
        print(protein_family)
        p_samples = data_aux[data_aux["Protein families"] == protein_family].index.tolist()
        data_aux.drop(p_samples, inplace=True)
        data_aux.reset_index(inplace=True, drop=True)
        random_samples = np.random.choice(data_aux.shape[0], int(protein_families_unique[idx] * 1.5), replace=False)
        embedding_weights = pd.read_csv(embedding_path, header=None, delimiter=',')
        embedding_weights = pd.concat([embedding_weights.iloc[random_samples], embedding_weights.iloc[p_samples]],
                                      axis=0)
        protein_families = pd.concat([protein_families.iloc[random_samples], protein_families.iloc[p_samples]], axis=0)
        protein_families = pd.DataFrame(protein_families).reset_index()
        if method == "TSNE.md":
            tsne = TSNE(n_components=2)
            embedding = tsne.fit_transform(embedding_weights)
        else:
            embedding = umap.UMAP(n_components=2, metric="correlation").fit_transform(embedding_weights)
        embedding = pd.DataFrame(embedding)
        embedding["Pfam"] = protein_families["Protein families"]
        pfam_emb = embedding.loc[embedding["Pfam"] == protein_family, [0, 1]]
        pfam_emb_others = embedding.loc[embedding["Pfam"] != protein_family, [0, 1]]
        if protein_family.__contains__("/"):
            protein_family = '-'.join(protein_family.split('/'))
        plot([pfam_emb[0], pfam_emb_others[0]], [pfam_emb[1], pfam_emb_others[1]], [protein_family, "Others"], method,
             func)
        data_aux = data


def plot(xs, ys, labels, method="tsne", func="sum"):
    path = "../results/uniprot/Skipgram/%s/%s/" % (func, labels[0])
    if not os.path.exists(path):
        os.mkdir(path)
    plt.close("all")
    plt.figure(figsize=(30, 30))
    plt.plot(xs[0], ys[0], 'ro', label=labels[0], markersize=14)
    plt.savefig(path + "%s_%s_%s.pdf" % (method, labels[0], func))
    plt.close("all")
    plt.figure(figsize=(30, 30))
    plt.plot(xs[1], ys[1], 'bo', label=labels[1], markersize=14)
    plt.savefig(path + "%s_%s_%s.pdf" % (method, labels[1], func))
    plt.close("all")
    plt.figure(figsize=(30, 30))
    plt.plot(xs[0], ys[0], 'o', label=labels[0], markersize=14)
    plt.plot(xs[1], ys[1], 'o', label=labels[1], markersize=14)
    plt.legend()
    # plt.savefig(path + "%s_%s.pdf" % (method, func))
    plt.show()


def classify():
    np.seterr(divide='ignore', invalid='ignore')
    mtl = MultiTaskLearner(data_path + "labeled_batch.csv", data_path + "unlabeled_batch.csv")
    scores, families = mtl.classify(method="RandomForest", embedding="Freq2Vec", func="weighted_average")
    # scores = mtl.classify(method="SVM", embedding="Freq2Vec", func="weighted_average")
    # scores = mtl.classify(method="KNN", embedding="Freq2Vec", func="weighted_average")
    # scores = mtl.classify(method="KNN", embedding="Freq2Vec", func="weighted_average")
    scores_2, families = mtl.classify(method="RandomForest", embedding="Skipgram", func="weighted_average")
    scores = pd.DataFrame(pd.concat([scores, scores_2], axis=1))
    print(scores)
    for idx, family in enumerate(families):
        print(family + " & " + "%.2f%% & %.4f & %.4f & " % (
            100 * scores.iloc[idx, 0], scores.iloc[idx, 1], scores.iloc[idx, 2]), end="")
        print("%.2f%% & %.4f & %.4f \\\\" % (100 * scores.iloc[idx, 3], scores.iloc[idx, 4], scores.iloc[idx, 5]))
    scores["family"] = families
    # scores = mtl.classify(method="GradientBoosting", embedding="Freq2Vec", func="weighted_average")
    # print(scores)
    # class_scores = pd.DataFrame(scores, index=['accuracy'])
    # class_scores.loc['accuracy', :] *= 100.0
    # class_scores.to_csv("../results/uniprot/Freq2Vec_weighted_avg.csv")
    scores.to_csv("../results/uniprot/embedding_comparison.csv")


if __name__ == '__main__':
    # preprocess()
    # classify()
    train()
    # visualization(embedding_path="../data/uniprot/Skipgram_weighted_average_Encoding.csv", method="UMAP", func="weighted_average")
    # visualization(embedding_path="../data/uniprot/Freq2Vec_weighted_average_Encoding.csv", method="UMAP",
    #               func="weighted_average")
    # visualization(embedding_path="../data/uniprot/Freq2Vec_weighted_sum_Encoding.csv", method="UMAP",
    #               func="weighted_sum")
    # print(pd.read_csv("../data/uniprot/LoadEmbedding_Encoding.csv").shape)
    # classify(method="SVM", embedding="Freq2Vec", func="weighted_average")
    # classify(method="SVM", embedding="Skipgram", func="weighted_average")
