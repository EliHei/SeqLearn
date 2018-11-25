import numpy as np
from gensim.models import Word2Vec

from .WordEmbedder import WordEmbedder


class GensimWord2Vec(WordEmbedder):
    def word2vec_maker(self):
        word2vec = Word2Vec(self.input, size=self.emb_dim, window=self.window_size, min_count=1, workers=2)
        word2vec.save(''.join(['../models/word2vec', '_',
                               str(self.emb_dim), '_',
                               str(self.window_size), '_',
                               str(self.word_length)]))
        word2vec.train(self.input, epochs=self.epochs, total_words=len(self.vocab_aux))
        # self.embedding_layer = skipgram.layers[0].get_weights()[0]
        word_vectors = word2vec.wv
        word_vectors.save('../aux/Word2Vec_words_aux.txt')
        arrays = list(map(lambda i: word2vec.wv[self.vocab_aux[i]], range(len(self.vocab_aux))))
        self.embedding_layer = np.stack(arrays, axis=0)
        print(self.embedding_layer.shape)
