# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #

import numpy as np

# =================================================================== #

class Word2vec():
    def __init__(self, w2v, nmax=100000):
        # Define self.word2vec and self.n
        self.word2vec = w2v
        self.n = len(self.word2vec)

        # Define self.word2id and self.id2word (two dictionaries)
        self.word2id = {w: i for i, w in enumerate(self.word2vec.keys())}
        self.id2word = {v: k for k, v in self.word2id.items()}

        # Define self.embeddings ([n,d] numpy array with d the dimension of the embeddings)
        self.embeddings = np.vstack(self.word2vec.values())

    def most_similar(self, w, K=5, verbose=True):
        # Initiate "all_score" variable (score for each words)
        all_score = np.zeros(self.n)

        # Save score/similarity for each words with "w"
        for idx, word in self.id2word.items() :
            all_score[idx] = self.score(w,word,False)

        # Extract the K most similar words of "w"
        res = np.argsort(all_score)[-K:][::-1]

        # Return result
        if verbose :
            print("Nearest neighbors of \"%s\":" % w)
            for i, idx in enumerate(res):
                print('%.4f - %s' % (all_score[idx], self.id2word[idx]))
        else :
            return res

    def score(self, w1, w2, verbose=True):
        # cosine similarity: np.dot  -  np.linalg.norm

        # Compute embeddings for each word "w1" and "w2"
        e1 = self.word2vec[w1]
        e2 = self.word2vec[w2]

        # Compute cosine similarity between "e1" and "e2" ("w1" and "w2")
        score = np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2))

        # Return result
        if verbose :
            print("Score between \"%s\" and \"%s\": %s" % (w1,w2,score))
        return score

# =================================================================== #

class BoV():
    def __init__(self, w2v):
        # Define Word2vec object for monolingual word embeddings
        self.w2v = w2v

    def encode(self, sentences, idf=False):
        # takes a list of sentences, outputs a numpy array of sentence embeddings
        # see TP1 for help

        # Initiate "sentemb" variable (sentence embeddings)
        sentemb = []

        # By default, if idf is False, we compute average words vectors for embedding sentence
        # If a idf dictionary is given, we compute idf-weighted average words vectors for embedding sentence

        # Define the list of all words which we know the embedding (and which we know the idf weighted)
        words = self.w2v.word2vec.keys()
        if not(idf is False) :
            words_idf = idf.keys()

        # Compute the embedding for each sentence in "sentences"
        for sent in sentences:
            # Extract sentence words in "words"
            if idf is False :
                sent = set([word for word in sent if word in words])
            else :
                sent = set([word for word in sent if word in words and word in words_idf])
            # Return the zero embedding if no word of the sentence is known
            if len(sent) == 0 :
                sentemb.append(0*self.w2v.word2vec[','])
            # Return the mean of word vectors
            elif idf is False:
                sentemb.append(np.mean([self.w2v.word2vec[word] for word in sent], axis=0))
            # Return the idf-weighted mean of word vectors
            else:
                sentemb.append(np.sum([self.w2v.word2vec[word]*idf[word] for word in sent], axis=0))

        # Return result
        return np.vstack(sentemb)

    def most_similar(self, s, sentences, idf=False, K=5, verbose=True):
        # get most similar sentences and **print** them

        # Initiate "all_score" variable (score for each sentence)
        all_score = np.zeros(len(sentences))

        # Save score/similarity for each sentence with "s"
        for idx, q in enumerate(sentences) :
            all_score[idx] = self.score(s, q, idf, False)

        # Extract the K most similar words of "w"
        res = np.argsort(all_score)[-K:][::-1]

        # Return result
        if verbose :
            print("Nearest neighbors of \"%s\":" % s)
            for i, idx in enumerate(res):
                print('%.4f - %s' % (all_score[idx], sentences[idx]))
        else :
            return [sentences[idx] for idx in res]

    def score(self, s1, s2, idf=False, verbose=True):
        # cosine similarity: use   np.dot  and  np.linalg.norm

        # Compute embeddings (or encoding) for each sentence "s1" and "s2"
        q1 = self.encode([s1], idf).flatten()
        q2 = self.encode([s2], idf).flatten()

        # Compute cosine similarity between "q1" and "q2" ("s1" and "s2")
        score = np.dot(q1,q2)/(np.linalg.norm(q1)*np.linalg.norm(q2))

        # Return result
        if verbose :
            print("Score between \n\"%s\" \nand \n\"%s\": \n%s" % (s1,s2,score))
        else :
            return score

    def build_idf(self, sentences):
        # build the idf dictionary: associate each word to its idf value

        # Initiate "idf" dictionary
        idf = {}

        # Extract the frequency for each word in sentence of "sentences"
        for sent in sentences :
            for word in set(sent) :
                idf[word] = idf.get(word, 0) + 1

        # Extract the number of sentences in "sentences"
        sentences_len = len(sentences)

        # Compute the idf weights for each word we meet
        for word in idf.keys() :
            idf[word] = max(1, np.log10(sentences_len / (idf[word])))

        # Return result
        return idf
