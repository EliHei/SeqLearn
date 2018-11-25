import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Embedding, Reshape, Dense
from keras.models import Sequential

from .WordEmbedder import WordEmbedder


class Freq2Vec(WordEmbedder):
    def freq2vec_maker(self):
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

        print(np.array([np.array(x) for x in range(len(self.vocab))]).shape)

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

        # freq2vec.save(''.join(['../models/freq2vec', '_',
        #                        str(self.emb_dim), '_',
        #                        str(self.window_size), '_',
        #                        str(self.word_length), '.h5']))
