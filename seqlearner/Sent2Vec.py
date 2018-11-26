from subprocess import Popen, PIPE
import numpy as np
import pandas as pd


class Sent2Vec:
    def __init__(self, sequences, word_length, emb_dim, epoch, lr, wordNgrams, loss, neg, thread, t,
                 dropoutK, bucket):
        self.sequences = sequences
        self.word_length = word_length
        self.emb_dim = emb_dim
        self.epoch = epoch
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

    def sent2vec(self):
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
        self.__sh(embed)
        print(embed)
        print("dkjadksfjadlkfjasd")
        ret = ('cat ../aux/Sent2Vec_sentences_aux.txt |'
               ' ../fastText/./fasttext print-sentence-vectors ../models/sent2vec.bin'
               ' > ../data/sent2vec_embedding.txt')
        self.__sh(ret)
        self.vec_df = pd.read_csv('../data/sent2vec_embedding.txt', sep=" ", header=None)
        self.vec_df = self.vec_df.values
        return list(map(lambda idx: self.__add_rows(idx), range((self.vec_df.shape[0] - self.word_length + 1))))

    def __sh(self, script):
        p = Popen(script.split(" "), stdout=PIPE)
        p.wait()

    def __add_rows(self, idx):
        return np.sum(self.vec_df[idx:(idx + self.word_length), :], axis=0)
