import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Embedding, Reshape, Dense
from keras.models import Sequential
from keras.utils import np_utils


class SkipGram:
    def __init__(self, sequences, word_length, window_size, emb_dim, loss, optimizer):
        self.sequences = sequences
        self.word_length = word_length
        self.window_size = window_size
        self.emb_dim = emb_dim
        self.loss = loss
        self.optimizer = optimizer
        self.adj_matrix = None
        self.corpus = []
        self.vocab = set()
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
        self.frequency = {k: v / total for total in (sum(self.frequency.values()),) for k, v in self.frequency.items()}
        self.frequency = self.frequency.values()

    def __corpus_maker(self):
        list(map(lambda seq: self.__seq_splitter(seq), self.sequences))
        self.vocab = dict(list(enumerate(self.vocab)))
        self.vocab = dict((v, k) for k, v in self.vocab.items())
        self.corpus = list(map(lambda x: list(map(lambda y: self.vocab.get(y, -1), x)), self.corpus))
        self.sentences = list(map(lambda x: list(map(lambda y: self.vocab.get(y, -1), x)), self.sentences))
        self.__freq_calc()

    def __word_indexer(self, idx, word_list):
        s = idx - self.window_size
        e = idx + self.window_size + 1
        rng = range(max(s, 0), min(e, (len(word_list) - 1)))
        in_words = [word_list[idx]] * len(rng)
        labels = list(map(lambda i: word_list[i], rng))
        return np.array(in_words, dtype=np.int32), np_utils.to_categorical(labels, len(self.vocab))

    def __generate_data(self):
        return list(map(lambda words: [(yield self.__word_indexer(idx, words)) for idx in range(len(words))],
                        self.corpus))

    def __neighbor_counter(self, idx, word_list):
        def __adder(idx1, idx2):
            self.adj_matrix[idx1, idx2] += 1

        s = idx - self.window_size
        e = idx + self.window_size + 1
        rng = range(max(s, 0), min(e, (len(word_list) - 1)))
        word = word_list[idx]
        list(map(lambda i: __adder(word, word_list[i]), rng))

    def __adj_matrix_maker(self):
        self.adj_matrix = np.zeros((len(self.vocab), len(self.vocab)))
        list(map(lambda words: list(map(lambda idx: self.__neighbor_counter(idx, words), range(len(words)))),
                 self.corpus))
        np.fill_diagonal(self.adj_matrix, 0)
        self.adj_matrix = (self.adj_matrix.T / self.adj_matrix.sum(axis=1)).T
        self.adj_matrix = np.nan_to_num(self.adj_matrix)

    def skipgram_maker(self):
        skipgram = Sequential()
        skipgram.add(Embedding(input_dim=len(self.vocab),
                               output_dim=self.emb_dim,
                               init='glorot_uniform',
                               input_length=1))
        skipgram.add(Reshape((self.emb_dim,)))
        skipgram.add(Dense(input_dim=self.emb_dim,
                           output_dim=len(self.vocab),
                           activation='softmax'))
        skipgram.compile(loss=self.loss,
                         optimizer=self.optimizer)
        csv_logger = CSVLogger(
            ''.join(['../logs/skipgram', '_',
                     str(self.emb_dim), '_',
                     str(self.window_size), '_',
                     str(self.word_length), '.log']))
        generator = self.__generate_data()
        for gen in generator:
            skipgram.fit_generator(gen,
                                   callbacks=[csv_logger],
                                   epochs=1000,
                                   verbose=0,
                                   steps_per_epoch=10)
        print("*****************evaluate*****************")
        skip_eval = skipgram.evaluate(np.array([[x] for x in range(len(self.vocab))]),
                                      self.adj_matrix,
                                      verbose=2,
                                      batch_size=100)
        np.savetxt(''.join(['../data/skipgram_embedding', '_',
                            str(self.emb_dim), '_',
                            str(self.window_size), '_',
                            str(self.word_length), '.txt']),
                   skipgram.layers[0].get_weights()[0])
        self.embedding_layer = skipgram.layers[0].get_weights()[0]
        skipgram.save(''.join(['../models/skipgram', '_',
                               str(self.emb_dim), '_',
                               str(self.window_size), '_',
                               str(self.word_length), '.h5']))
