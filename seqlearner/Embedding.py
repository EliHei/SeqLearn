import os

import numpy as np
import pandas as pd

from seqlearner.Freq2Vec import Freq2Vec
from seqlearner.GensimWord2Vec import GensimWord2Vec
from seqlearner.Sent2Vec import Sent2Vec
from seqlearner.SkipGram import SkipGram


class Embedding:
    def __init__(self, sequences, word_length):
        self.sequences = sequences
        self.word_length = word_length
        self.sentences = None
        self.embedding_layer = None
        self.encoding = None
        self.frequency = None

    def skipgram(self, func="sum", window_size=10, emb_dim=20, loss="mean_squared_error", epochs=100):
        skipgram = SkipGram(self.sequences, self.word_length, window_size, emb_dim, loss, epochs)
        skipgram.skipgram_maker()
        skipgram.__name__ = "Skipgram"
        self.sentences = skipgram.sentences
        self.embedding_layer = skipgram.embedding_layer
        if func is "sum":
            self.__sum()
        elif func is "average":
            self.__average()
        elif func is "weighted_sum":
            self.frequency = skipgram.frequency
            self.__weighted_sum()
        elif func is "weighted_average":
            self.frequency = skipgram.frequency
            self.__weighted_average()
        self.__save_embedding(skipgram, file_path="../results/embeddings/")
        return self.encoding

    def freq2vec(self, func="sum", window_size=10, emb_dim=20, loss="mean_squared_error", epochs=100):
        freq2vec = Freq2Vec(self.sequences, self.word_length, window_size, emb_dim, loss, epochs)
        freq2vec.freq2vec_maker()
        freq2vec.__name__ = "Freq2Vec"
        self.sentences = freq2vec.sentences
        self.embedding_layer = freq2vec.embedding_layer
        if func is "sum":
            self.__sum()
        elif func is "average":
            self.__average()
        elif func is "weighted_sum":
            self.frequency = freq2vec.frequency
            self.__weighted_sum()
        elif func is "weighted_average":
            self.frequency = freq2vec.frequency
            self.__weighted_average()
        self.__save_embedding(freq2vec, file_path="../results/embeddings/")
        return self.encoding

    def word2vec(self, func="sum", window_size=10, emb_dim=20, workers=2, epochs=1000):
        gensim_wor2vec = GensimWord2Vec(self.sequences, self.word_length, window_size, emb_dim, workers, epochs)
        gensim_wor2vec.word2vec_maker()
        gensim_wor2vec.__name__ = "Word2Vec"
        self.sentences = gensim_wor2vec.sentences
        self.embedding_layer = gensim_wor2vec.embedding_layer
        if func is "sum":
            self.__sum()
        elif func is "average":
            self.__average()
        elif func is "weighted_sum":
            self.frequency = gensim_wor2vec.frequency
            self.__weighted_sum()
        elif func is "weighted_average":
            self.frequency = gensim_wor2vec.frequency
            self.__weighted_average()
        self.__save_embedding(gensim_wor2vec, file_path="../results/embeddings/")
        return self.encoding

    def sent2vec(self, emb_dim=100, epoch=1000, lr=1, wordNgrams=10, loss="ns", neg=10, thread=10,
                 t=0.000005, dropoutK=4, bucket=4000000):
        s2v = Sent2Vec(self.sequences, self.word_length, emb_dim, epoch, lr, wordNgrams, loss, neg, thread, t, dropoutK,
                       bucket)
        self.encoding = s2v.sent2vec()
        s2v.__name__ = "Sent2Vec"
        self.__save_embedding(s2v, file_path="../results/embeddings/")
        return self.encoding

    def load_embedding(self, func, file):
        embed = Embedding(self.sequences, self.word_length, file)
        embed.embed()
        embed.__name__ = "LoadEmbedding"
        self.sentences = embed.sentences
        self.embedding_layer = embed.embedding_layer
        if func is "sum":
            self.__sum()
        elif func is "average":
            self.__average()
        elif func is "weighted_sum":
            self.frequency = embed.frequency
            self.__weighted_sum()
        elif func is "weighted_average":
            self.frequency = embed.frequency
            self.__weighted_average()
        self.__save_embedding(embed, file_path="../results/embeddings/")
        return self.encoding

    def ELMo(self):
        pass

    def __sum(self):
        self.encoding = list(map(lambda sent: np.add.reduce(self.embedding_layer[sent, :]), self.sentences))

    def __average(self):
        self.__sum()
        self.encoding = [np.divide(self.encoding[i], len(self.sentences[i])) for i in range(len(self.encoding))]

    def __weighted_sum(self):
        self.frequency = np.array(list(self.frequency))
        self.frequency = np.reciprocal(self.frequency)
        self.sentences = list(map(lambda sent: (self.embedding_layer[sent].T * self.frequency[sent]).T, self.sentences))
        self.encoding = list(map(lambda sent: np.add.reduce(sent), self.sentences))

    def __weighted_average(self):
        self.__weighted_sum()
        self.encoding = [np.divide(self.encoding[i], len(self.sentences[i])) for i in range(len(self.encoding))]

    def __save_encoding(self, embedding_algorithm="Freq2Vec", file_path="../data/uniprot/"):
        encoding = np.array(self.encoding)
        np.savetxt(fname=file_path + embedding_algorithm + "_Encoding.csv", X=encoding, delimiter=',')

    def __save_embedding(self, embedding=None, file_path="../results/embeddings/"):
        if embedding is None:
            raise Exception("embedding has to be a WordEmbedder child class like Freq2Vec, ... .")
        file_path += embedding.__name__ + "/"
        os.makedirs(file_path, exist_ok=True)
        embedding_weights = pd.concat([pd.Series(embedding.vocab_indices), pd.DataFrame(embedding.embedding_layer)],
                                      axis=1)
        embedding_weights.columns = ["words"] + ["dim_%d" % i for i in range(embedding.emb_dim)]
        save_path = file_path + embedding.__name__ + "_" + "_".join(
            [str(embedding.emb_dim), str(embedding.window_size), str(embedding.word_length)]) + ".csv"
        embedding_weights.to_csv(save_path)
