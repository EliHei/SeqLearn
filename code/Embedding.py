import numpy as np
from code.SkipGram import SkipGram
from code.EmbeddingLoader import EmbeddingLoader
from code.Sent2Vec import Sent2Vec


class Embedding:
    def __init__(self, sequences, word_length):
        self.sequences = sequences
        self.word_length = word_length
        self.sentences = None
        self.embedding_layer = None
        self.encoding = None
        self.frequency = None

    def skipgram(self, func="sum", window_size=10, emb_dim=20, loss="mean_squared_error", optimizer="adam"):
        skipgram = SkipGram(self.sequences, self.word_length, window_size, emb_dim, loss, optimizer)
        skipgram.skipgram_maker()
        self.sentences = skipgram.sentences
        self.embedding_layer = skipgram.embedding_layer
        if func is "sum":
            self.__sum()
        elif func is "average":
            self.__average()
        elif func is "weighted_sum":
            self.__weighted_sum()
        elif func is "weighted_average":
            self.__weighted_average()
        return self.encoding

    def sent2vec(self, emb_dim=10, epoch=1000, lr=0.2, wordNgrams=10, loss="ns", neg=10, thread=20,
                 t=0.000005, dropoutK=4, bucket=400):
        s2v = Sent2Vec(self.sequences, self.word_length, emb_dim, epoch, lr, wordNgrams, loss, neg, thread, t, dropoutK,
                       bucket)
        self.encoding = s2v.sent2vec()
        return self.encoding

    def load_embedding(self, func, file):
        embed = EmbeddingLoader(self.sequences, self.word_length, file)
        embed.embed()
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
