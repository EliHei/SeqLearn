import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger
from keras.layers import Embedding, Reshape, Dense
from keras.models import Sequential

from .WordEmbedder import WordEmbedder


class Freq2Vec(WordEmbedder):
    """
        Freq2Vec Embedding Method. This class contains the implementation for Freq2Vec Embedding
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
        Freq2Vec.freq2vec_maker : build a model and train it!

    """

    def __init__(self, sequences, word_length, window_size, emb_dim, loss, epochs):
        self.embedding = "Freq2Vec"
        super().__init__(sequences, word_length, window_size, emb_dim, loss, epochs)

    def freq2vec_maker(self):
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
            >>> freq2vec = Freq2Vec(sequences, word_length=3, window_size=5, emb_dim=25, loss="mean_squared_error", epochs=250)
            >>> freq2vec.freq2vec_maker()
        """
        freq2vec = Sequential()
        freq2vec.add(Embedding(input_dim=(len(self.vocab)),
                               output_dim=self.emb_dim,
                               init='glorot_uniform',
                               input_length=1))
        freq2vec.add(Reshape((self.emb_dim,)))
        freq2vec.add(Dense(input_dim=self.emb_dim,
                           output_dim=(len(self.vocab)),
                           activation='softmax'))
        freq2vec.compile(loss=self.loss,
                         optimizer=self.optimizer)
        csv_logger = CSVLogger(
            ''.join(['../logs/freq2vec', '_',
                     str(self.emb_dim), '_',
                     str(self.window_size), '_',
                     str(self.word_length), '.log']))

        freq2vec.summary()

        freq2vec.fit(x=np.array([[x] for x in range(len(self.vocab))]),
                     y=self.adj_matrix,
                     batch_size=100,
                     epochs=self.epochs,
                     verbose=2,
                     callbacks=[csv_logger],
                     validation_split=0.1)

        # np.savetxt(''.join(['../data/freq2vec_embedding', '_',
        #                     str(self.emb_dim), '_',
        #                     str(self.window_size), '_',
        #                     str(self.word_length), '.txt']),
        #            freq2vec.layers[0].get_weights()[0])
        # print(np.array(freq2vec.layers[0].get_weights()).shape)
        self.embedding_layer = freq2vec.layers[0].get_weights()[0]
        filename = ''.join(['../data/freq2vec_embedding', '_',
                            str(self.emb_dim), '_',
                            str(self.window_size), '_',
                            str(self.word_length), '.csv'])
        pd.DataFrame(self.embedding_layer, index=pd.DataFrame(list(self.vocab))).to_csv(filename)

        # freq2vec.save(''.join(['../models/freq2vec', '_',
        #                        str(self.emb_dim), '_',
        #                        str(self.window_size), '_',
        #                        str(self.word_length), '.h5']))
