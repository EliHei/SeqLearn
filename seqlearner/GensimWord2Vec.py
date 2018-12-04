import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from .WordEmbedder import WordEmbedder


class GensimWord2Vec(WordEmbedder):
    """
        Word2Vec Embedding Method. This class is wrapper for Word2Vec Embedding
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
        Word2Vec.word2vec_maker : build a model and train it!

    """

    def __init__(self, sequences, word_length, window_size, emb_dim, loss, epochs):
        self.embedding = "Word2Vec"
        super().__init__(sequences, word_length, window_size, emb_dim, loss, epochs)

    def word2vec_maker(self):
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
            >>> freq2vec = GensimWord2Vec(sequences, word_length=3, window_size=5, emb_dim=25, loss="mean_squared_error", epochs=250)
            >>> freq2vec.word2vec_maker()
        """
        word2vec = Word2Vec(self.input, size=self.emb_dim, window=self.window_size, min_count=1,
                            workers=2)  # TODO: workers=self.loss ?
        word2vec.save(''.join(['../models/word2vec', '_',
                               str(self.emb_dim), '_',
                               str(self.window_size), '_',
                               str(self.word_length)]))
        word2vec.train(self.input, epochs=self.epochs, total_words=len(self.vocab_aux))
        # self.embedding_layer = skipgram.layers[0].get_weights()[0]
        word_vectors = word2vec.wv
        # word_vectors.save('../aux/Word2Vec_words_aux.txt')
        arrays = list(map(lambda i: word2vec.wv[self.vocab_aux[i]], range(len(self.vocab_aux))))
        self.embedding_layer = np.stack(arrays, axis=0)
        filename = ''.join(['../data/word2vec_embedding', '_',
                            str(self.emb_dim), '_',
                            str(self.window_size), '_',
                            str(self.word_length), '.csv'])
        pd.DataFrame(self.embedding_layer, index=pd.DataFrame(list(self.vocab))).to_csv(filename)
