import os
import numpy as np
import pandas as pd
from keras.optimizers import Adam


class WordEmbedder:
    """
        WordEmbedder is a helper class for every embedding algorithms. It
        does extract all possible words, adjacency matrix, corpus from
        the given sequences. It is parent class of SkipGram, Freq2Vec, GensimWord2Vec.

        Parameters
        ----------
        sequences : numpy ndarray, list, or DataFrame
           sequences of data like protein sequences
        word_length : integer
            The length of each word in sequences to be separated from each other.
        window_size: integer
            Size of window for counting the number of neighbors.
        emb_dim: integer
            Number of embedding vector dimensions.
        loss: basestring
            The loss function is going to be used on training phase.
        epochs: integer
            Number of epochs for training the embedding.

        See also
        --------
        SkipGram : Skipgram Embedding
        Freq2Vec : Freq2Vec Embedding
        GensimWord2Vec : Word2Vec Embedding
        Sent2Vec : Sent2Vec Embedding

    """

    def __init__(self, sequences, word_length, window_size, emb_dim, loss, epochs):
        self.sequences = sequences
        self.word_length = word_length
        self.window_size = window_size
        self.emb_dim = emb_dim
        self.loss = loss
        self.optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.epochs = epochs
        self.adj_matrix = None
        self.corpus = []
        self.vocab = set()
        self.vocabulary = pd.Series()
        self.sentences = []
        self.embedding_layer = None
        self.__corpus_maker()
        self.__adj_matrix_maker()

    def __seq_splitter(self, seq):
        words = list(map(lambda x: seq[x:(x + self.word_length)], range((len(seq) - self.word_length + 1))))
        self.vocab |= set(words)
        list(map(lambda s: self.corpus.append(words[s::self.word_length]), range(self.word_length)))
        self.sentences.append(words)

    def __freq_calc(self):
        def adder(idx):
            self.frequency[idx] += 1

        self.frequency = dict.fromkeys(range(len(self.vocab)), 0)
        list(map(lambda sent: list(map(lambda word: adder(word), sent)), self.sentences))
        os.makedirs("../aux/", exist_ok=True)
        with open('../aux/' + self.embedding + '_vocab.txt', 'w') as out:
            out.write(",".join(self.vocab))
        self.frequency = {k: v / total for total in (sum(self.frequency.values()),) for k, v in self.frequency.items()}
        self.frequency = self.frequency.values()

    def __corpus_maker(self):
        list(map(lambda seq: self.__seq_splitter(seq), self.sequences))
        self.input = self.sentences
        self.vocab = dict(list(enumerate(self.vocab)))
        self.vocab_aux = self.vocab
        self.vocab_indices = list(k for k, v in self.vocab.items())
        self.vocab = dict((v, k) for k, v in self.vocab.items())
        self.corpus = list(map(lambda x: list(map(lambda y: self.vocab.get(y, -1), x)), self.corpus))
        self.sentences = list(map(lambda x: list(map(lambda y: self.vocab.get(y, -1), x)), self.sentences))
        self.__freq_calc()

    def __neighbor_counter(self, idx, word_list):
        def __adder(idx1, idx2):
            self.adj_matrix[idx1, idx2] += 1

        s = idx - self.window_size
        e = idx + self.window_size + 1
        rng = range(max(s, 0), min(e, (len(word_list) - 1)))
        word = word_list[idx]
        list(map(lambda i: __adder(word, word_list[i]), rng))

    def __adj_matrix_maker(self):
        self.adj_matrix = np.zeros(((len(self.vocab)), (len(self.vocab))))
        list(map(lambda words: list(map(lambda idx: self.__neighbor_counter(idx, words), range(len(words)))),
                 self.corpus))
        np.fill_diagonal(self.adj_matrix, 0)
        self.adj_matrix = (self.adj_matrix.T / self.adj_matrix.sum(axis=1)).T
        self.adj_matrix = np.nan_to_num(self.adj_matrix)
