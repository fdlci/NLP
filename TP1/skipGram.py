from __future__ import division
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# useful stuff
import numpy as np
import random as rd
from scipy.special import expit
from sklearn.preprocessing import normalize
import string

__authors__ = ['Ines Florez De La Colina','Mohamed Chenene','Aymen Qabel', 'Elias Aouad']
__emails__  = ['ines.florezdelacolina@student-cs.fr','mohamed.chenene@student-cs.fr','aymen.qabel@student-cs.fr', 'elias.aouad@student.ecp.fr']


def text2sentences(path):
    """Transforms the input data into a training dataset. Removes the punctuation
    and the digits"""
    sentences = []
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
    with open(path, encoding='utf8') as f:
        for l in f:
            # get rid of punctuation
            s = l.translate(str.maketrans("","", punctuation))
            sentences.append(s.lower().split())
    return sentences

def loadPairs(path):
    """Loads the pairs of words from the test set"""
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

def appearances_of_words(sentences, minCount = 5):
    """Computes three dictionnaries
    - total_words: contains all the words from the vocab with their frequences
    - frequent_words: contains only the words that appear more than minCount times
    and their frequences
    - indexes: contains every word of the vocab and assigns an index to each word"""
    total_words = {}; frequent_words = {}; indexes = {}
    counter = 0
    # count number of times each word appears in the vocabulary
    for sentence in sentences:
        for word in sentence:
            if word in total_words:
                total_words[word] += 1
                if total_words[word] >= minCount:
                    frequent_words[word] = total_words[word]
            else:
                total_words[word] = 1
                indexes[word] = counter
                counter += 1
    return total_words, frequent_words, indexes

def prepare_sampling(num_words, frequent_words, indexes):
    """Computes the index table that makes the index of a word appear a given number 
    of times. This number of times is proportionnal to the probability"""
    index_table = []
    table_size = len(frequent_words)
    prob = []
    words_list = []
    total_frequences = sum([i**(3/4) for i in list(frequent_words.values())])
    for word in frequent_words:
        proba = frequent_words[word]**(3/4)/total_frequences
        prob.append(proba*table_size)
        index_table += [indexes[word] for i in range(int(proba)+2)]
    return index_table

def subsampling(num_words, t =1e-5, threshold = 0.9):
    """Computes subsampling (find the stop words of the vocabulary)"""
    words_distribution = np.array(list(num_words.values()))
    words_distribution = words_distribution/np.sum(words_distribution)
    distribution = (words_distribution-t)/words_distribution - np.sqrt(t/words_distribution)

    idx_subsample = np.where(distribution > threshold)

    idx2words = np.array(list(num_words.keys()))

    return idx2words[idx_subsample]

def sigmoid(x,y):
    """Computes the sigmoid function"""
    grad = 1/(1+np.exp(-x@y.T))
    return grad.reshape((-1,1))

class SkipGram:
    """Implementation of Skip Gram from scratch"""
    def __init__(self, sentences, sub_sampling=False, vector_norm = False, nEmbed=100, negativeRate=5, winSize = 3, minCount = 5, lr = 1e-1, epochs = 30, display_rate = 10):

        total_words, frequent_words, indexes = appearances_of_words(sentences)
        if sub_sampling:
          stop_words = subsampling(total_words)
          sentences = [list(filter(lambda word: not word in stop_words, sentence)) for sentence in sentences]
          total_words, frequent_words, indexes = appearances_of_words(sentences)

        self.w2id = indexes # word to ID mapping
        self.id2w = {self.w2id[w]:w for w in indexes.keys()}

        self.trainset = sentences # set of sentences
        self.vocab = total_words # list of valid words
        self.index_table = prepare_sampling(total_words, frequent_words, indexes)
        self.nEmbed = nEmbed
        self.W = np.random.uniform(-0.5,0.5,size=(len(self.vocab), nEmbed))
        self.C = np.random.uniform(-0.5,0.5,size=(len(self.vocab), nEmbed))
        self.loss = []
        self.positive_pairs = []
        self.trainWords = 0
        self.accLoss = 0
        self.negativeRate = negativeRate
        self.lr = lr
        self.winSize = winSize
        self.epochs = epochs
        self.display_rate = display_rate
        self.vector_norm = vector_norm


    def sample(self, omit):
        """samples negative words, ommitting those in set omit"""
        negative_words = []
        N = len(self.index_table)

        while len(negative_words) < self.negativeRate:
            neg_word_ind = self.index_table[rd.randint(0,N-1)]
            # neg_word = list(self.w2id.keys())[list(self.w2id.values()).index(neg_word_ind)]
            if neg_word_ind not in omit:
                negative_words.append(neg_word_ind)
        return np.array(negative_words)

    def train(self):
        """Trains the model"""
        best_loss = - np.inf
        count_loss = 0

        for epoch in range(self.epochs):
            np.random.shuffle(self.trainset)

            for sentence in self.trainset:
                for wpos, word in enumerate(sentence):
                    wIdx = self.w2id[word]
                    winsize = np.random.randint(self.winSize) + 1
                    start = max(0, wpos - winsize)
                    end = min(wpos + winsize + 1, len(sentence))


                    for context_word in sentence[start:end]:
                        ctxtId = self.w2id[context_word]
                        if ctxtId == wIdx: 
                            continue
                        else:
                            negativeIds = self.sample({wIdx,ctxtId})
                            self.trainWord(wIdx, ctxtId, negativeIds)
                            self.accLoss += self.compute_objective_function(wIdx, ctxtId, negativeIds)
                            self.trainWords += 1

            
            
            loss_epoch = self.accLoss / self.trainWords
            self.loss.append(self.accLoss / self.trainWords)

            if loss_epoch > best_loss:
                best_loss = loss_epoch
                count_loss = 0
            else:
                count_loss +=1
            if count_loss > 5:
                self.lr /=10
                count_loss = 0
            if self.lr < 1e-6:
                print('Last epoch {}/{}'.format(epoch,self.epochs))
                print('Learning rate too small')
                break

            self.trainWords = 0
            self.accLoss = 0
            if epoch%self.display_rate  == 0:
                print('Epoch {}/{}'.format(epoch,self.epochs))
                print('> Training loss : {:4f}'.format(loss_epoch))
                print('-'*30)

        if self.vector_norm:
            self.W /= np.linalg.norm(self.W,axis=0)

    def compute_objective_function(self, wordId, contextId, negativeIds):
        """Computes the value of the objective function to check that it is increasing"""
        x_word = self.W[wordId]
        y_context = self.C[contextId]
        Z = self.C[negativeIds]

        obj = np.sum(np.log(sigmoid(x_word,y_context))) + np.sum(np.log(sigmoid(-x_word,Z)))

        return obj

    def trainWord(self, wordId, contextId, negativeIds):
        """Computes the SGD to update the embeddings"""
        
        x_word = self.W[wordId]
        y_context = self.C[contextId]
        Z = self.C[negativeIds]

        grad_z_w = np.sum(sigmoid(x_word,Z)*Z,axis=0)
        grad_z_c = - sigmoid(x_word,Z)*x_word

        self.C[negativeIds] += self.lr*grad_z_c

        gradient_x = sigmoid(-x_word, y_context)*y_context - grad_z_w
        gradient_y = sigmoid(-x_word, y_context)*x_word


        self.W[wordId] = self.W[wordId] + self.lr*gradient_x
        self.C[contextId] = self.C[contextId] + self.lr*gradient_y

    def save(self,path):
        """Saves the model"""
        with open(path, 'wb') as f:
            pickle.dump(self.W, f)

    def similarity(self,word1,word2,W):
        """
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
        mean_vector = W.mean(0)

        if word1 not in self.vocab and word2 not in self.vocab:
            return 0
        elif word1 not in self.vocab:
            v_1 = mean_vector
            v_2 = W[self.w2id[word2]]
            return v_1.dot(v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
        elif word2 not in self.vocab:
            v_1 = W[self.w2id[word1]]
            v_2 = mean_vector
            return v_1.dot(v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))
        else:
			      #get the indexes
            v_1 = W[self.w2id[word1]]
            v_2 = W[self.w2id[word2]]

			      #cosine similarity
            cosine_sim = v_1.dot(v_2) / (np.linalg.norm(v_1) * np.linalg.norm(v_2))

            return cosine_sim

    def predict_similar_words(self, word, W, w2id, vocab, n_similar=5):
        """Predicts the n_similar most similar words to word"""
        key = lambda w: self.similarity(word,w,W,w2id,vocab) if w != word else - np.inf 
        similar_words = sorted(list(self.vocab.keys()), key=key, reverse=True)
        return similar_words[:n_similar]


    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            W = pickle.load(f)
        return W

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--text', help='path containing training data', required=True)
  parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
  parser.add_argument('--test', help='enters test mode', action='store_true')

  opts = parser.parse_args()
  sentences = text2sentences(opts.text)

  if not opts.test:
    sg = SkipGram(sentences, sub_sampling=True)
    sg.train()
    sg.save(opts.model)

  else:
    pairs = loadPairs(opts.text)
    sg = SkipGram(sentences)
    W = SkipGram.load(opts.model)
    # sg = SkipGram.load(opts.model)
    for a,b,_ in pairs:
      # make sure this does not raise any exception, even if a or b are not in sg.vocab
      print(sg.similarity(a,b,W))