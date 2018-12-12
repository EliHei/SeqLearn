import os
from subprocess import Popen, PIPE

import numpy as np
import pandas as pd


class Sent2Vec:
    """
        Sent2Vec Embedding Method. This class is wrapper for Sent2Vec Embedding
        method to apply on a set of sequences. Child class of WordEmbedder.

        Parameters
        ----------
        sequences : numpy ndarray, list, or DataFrame
           sequences of data like protein sequences
        word_length : integer
            The length of each word in sequences to be separated from each other.
        emb_dim: integer
            Number of embedding vector dimensions.
        epochs: integer
            Number of epochs for training the embedding.
        lr: float
                learning rate
        wordNgrams: integer
            max length of word n-gram
        loss: {"ns", "hs", "softmax"}, default "ns"
            loss function
        neg: integer
            number of negatives sampled
        thread: integer
            number of threads
        t: float
            sampling threshold
        dropoutK: integer
            number of n-grams dropped when training a sent2vec model
        bucket: integer
            number of hash buckets for vocabulary

        See also
        --------
        Sent2Vec.sent2vec : compute Sent2Vec embedding using fasttext.

    """

    def __init__(self, sequences, word_length, emb_dim, epochs, lr, wordNgrams, loss, neg, thread, t,
                 dropoutK, bucket):
        self.sequences = sequences
        self.word_length = word_length
        self.emb_dim = emb_dim
        self.epoch = epochs
        self.lr = lr
        self.wordNgrams = wordNgrams
        self.loss = loss
        self.neg = neg
        self.thread = thread
        self.t = t
        self.dropoutK = dropoutK
        self.bucket = bucket
        self.corpus = []
        self.vec_df = []

    def __seq_splitter(self, seq):
        words = list(map(lambda x: seq[x:(x + self.word_length)], range((len(seq) - self.word_length + 1))))
        list(map(lambda s: self.corpus.append(words[s::self.word_length]), range(self.word_length)))

    def __corpus_maker(self):
        list(map(lambda seq: self.__seq_splitter(seq), self.sequences))
        self.corpus = list(map(lambda seq: " ".join(seq), self.corpus))
        print(self.corpus[1])
        with open('../aux/Sent2Vec_sentences_aux.txt', 'w') as out:
            out.write("\n".join(self.corpus))

    def sent2vec_maker(self):
        """
            Train Embedding layer on vocabulary in order to get embedding weights
            for each word in vocabulary. compress each in `emb_dim` vectors.

            Parameters
            ----------
            No parameters are needed.

            Returns
            -------
            encoding: list of embedding vectors for sentences

            Example
            --------
            >>> import pandas as pd
            >>> sequences = pd.read_csv("./sequences.csv", header=None)
            >>> s2v = Sent2Vec(sequences, word_length=3, emb_dim=25, epoch=100, lr=0.2, wordNgrams=5, loss="hs", neg=20, thread=10, t=0.0000005, dropoutK=2, bucket=4000000)
            >>> encoding = s2v.sent2vec_maker()
        """
        if not os.path.exists("../models/sent2vec.bin"):

            self.__corpus_maker()
            embed = ('./fastText/./fasttext sent2vec'
                     ' -input ../aux/Sent2Vec_sentences_aux.txt'
                     ' -output ../models/sent2vec'
                     ' -dim {}'
                     ' -epoch {}'
                     ' -lr {}'
                     ' -wordNgrams {}'
                     ' -loss {}'
                     ' -neg {}'
                     ' -thread {}'
                     ' -t {}'
                     ' -dropoutK {}'
                     ' -bucket {}').format(self.emb_dim,
                                           self.epoch, self.lr,
                                           self.wordNgrams,
                                           self.loss, self.neg,
                                           self.thread, self.t,
                                           self.dropoutK,
                                           self.bucket)
            print(embed)
            self.__sh(embed)
        if not os.path.exists("../aux/sent2vec_embedding.txt"):
            ret = ('cat ../aux/Sent2Vec_sentences_aux.txt |'
                   ' ../fastText/./fasttext print-sentence-vectors ../models/sent2vec.bin'
                   ' > ../data/sent2vec_embedding.txt')
            print(ret)
            self.__sh(ret)
        self.vec_df = pd.read_csv('../data/sent2vec_embedding.txt', sep=" ", header=None)
        self.vec_df = self.vec_df.values
        return list(map(lambda idx: self.__add_rows(idx), range((self.vec_df.shape[0] - self.word_length + 1))))

    def __sh(self, script):
        p = Popen(script.split(" "), stdout=PIPE)
        p.wait()

    def __add_rows(self, idx):
        return np.sum(self.vec_df[idx:(idx + self.word_length), :], axis=0)
