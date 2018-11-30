import pandas as pd


class EmbeddingLoader:
    """
        class
        Embedding Algorithm to be instantiated. Algorithms are Skipgram, Freq2Vec,
        Sent2Vec and etc. Can be thought as a blackbox which applies a specific
        embedding algorithm on the sequences. It also can save the embedding layer
        and encoding.

        Parameters
        ----------
        sequences : numpy ndarray, list, or DataFrame
           sequences of data like protein sequences
        word_length : integer
            The length of each word in sequences to be separated from each other.
        file : basestring
            The path which the embedding is saved.

        See also
        --------
        Embedding.load_embedding : loads an existing embedding to apply

    """
    def __init__(self, sequences, word_length, file):
        self.sequences = sequences
        self.word_length = word_length
        self.sentences = []
        emb_df = pd.read_csv(file, delimiter=' ', header=None)
        self.vocab = emb_df.index
        self.embedding_layer = emb_df.values
        self.frequency = None

    def __seq_splitter(self, seq):
        words = list(map(lambda x: seq[x:(x + self.word_length)], range((len(seq) - self.word_length + 1))))
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
        self.sentences = list(map(lambda x: list(map(lambda y: self.vocab.get(y, 0), x)), self.sentences))
        self.__freq_calc()

    def embed(self):
        """
            do the embedding based on the corresponding embedding layer which has just been loaded.
        """
        self.__corpus_maker()
