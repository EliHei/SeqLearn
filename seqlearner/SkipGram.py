import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Embedding, Reshape, Dense
from keras.models import Sequential
from keras.utils import np_utils

from .WordEmbedder import WordEmbedder
import pandas as pd


class SkipGram(WordEmbedder):
    """
        SkipGram Embedding Method. This class is wrapper for Freq2Vec Embedding
        method to apply on a set of sequences. Child class of WordEmbedder.

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
        SkipGram.skipgram_maker : build a model and train it!

    """

    def __init__(self, sequences, word_length, window_size, emb_dim, loss, epochs):
        self.embedding = "SkipGram"
        super().__init__(sequences, word_length, window_size, emb_dim, loss, epochs)

    # def __word_indexer(self, idx, word_list):
    #     s = idx - self.window_size
    #     e = idx + self.window_size + 1
    #     rng = range(max(s, 0), min(e, (len(word_list) - 1)))
    #     in_words = [word_list[idx]] * len(rng)
    #     labels = list(map(lambda i: word_list[i], rng))
    #     return (np.array(in_words, dtype=np.int32), np_utils.to_categorical(labels, (len(self.vocab)+1)))

    def __generate_data(self):
        while True:
            for words in self.corpus:
                for idx in range(len(words)):
                    s = idx - self.window_size
                    e = idx + self.window_size + 1
                    rng = range(max(s, 0), min(e, (len(words) - 1)))
                    in_words = [words[idx]] * len(rng)
                    labels = list(map(lambda i: words[i], rng))
                    yield (np.array(in_words, dtype=np.int32), np_utils.to_categorical(labels, (len(self.vocab))))

    def skipgram_maker(self):
        """
            Train Embedding layer on vocabulary in order to get embedding weights
            for each word in vocabulary. compress each in `emb_dim` vectors.

            Parameters
            ----------
            No parameters are needed.

            Returns
            -------
            Nothing will be returned.

            Example
            --------
            >>> import pandas as pd
            >>> sequences = pd.read_csv("./sequences.csv", header=None)
            >>> freq2vec = SkipGram(sequences, word_length=3, window_size=5, emb_dim=25, loss="mean_squared_error", epochs=250)
            >>> freq2vec.skipgram_maker()
        """
        skipgram = Sequential()
        skipgram.add(Embedding(input_dim=(len(self.vocab)),
                               output_dim=self.emb_dim,
                               init='glorot_uniform',
                               input_length=1))
        skipgram.add(Reshape((self.emb_dim,)))
        skipgram.add(Dense(input_dim=self.emb_dim,
                           output_dim=(len(self.vocab)),
                           activation='softmax'))
        skipgram.compile(loss=self.loss,
                         optimizer=self.optimizer)
        csv_logger = CSVLogger(
            ''.join(['../logs/skipgram', '_',
                     str(self.emb_dim), '_',
                     str(self.window_size), '_',
                     str(self.word_length), '.log']))
        
        skipgram.fit_generator(self.__generate_data(),
                               callbacks=[csv_logger],
                               epochs=self.epochs,
                               verbose=2,
                               steps_per_epoch=100)

        print("*****************evaluate*****************")
        print(skipgram.evaluate(np.array([[x] for x in range(len(self.vocab))]),
                                self.adj_matrix,
                                verbose=2,
                                batch_size=100))
        filename = ''.join(['../data/skipgram_embedding', '_',
                            str(self.emb_dim), '_',
                            str(self.window_size), '_',
                            str(self.word_length), '.csv'])

        self.embedding_layer = skipgram.layers[0].get_weights()[0]
        pd.DataFrame(self.embedding_layer, index=pd.DataFrame(list(self.vocab))).to_csv(filename)
        # skipgram.save(''.join(['../models/skipgram', '_',
        #                        str(self.emb_dim), '_',
        #                        str(self.window_size), '_',
        #                        str(self.word_length), '.h5']))
