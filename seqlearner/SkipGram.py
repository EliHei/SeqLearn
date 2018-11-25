import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Embedding, Reshape, Dense
from keras.models import Sequential
from keras.utils import np_utils

from .WordEmbedder import WordEmbedder


class SkipGram(WordEmbedder):
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

        np.savetxt(''.join(['../data/skipgram_embedding', '_',
                            str(self.emb_dim), '_',
                            str(self.window_size), '_',
                            str(self.word_length), '.txt']),
                   skipgram.layers[0].get_weights()[0])
        self.embedding_layer = skipgram.layers[0].get_weights()[0]
        # skipgram.save(''.join(['../models/skipgram', '_',
        #                        str(self.emb_dim), '_',
        #                        str(self.window_size), '_',
        #                        str(self.word_length), '.h5']))
