from PageScraper import PageScraper
import TextTool

import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import pprint
import matplotlib.pyplot as plt



ps = PageScraper()

text = ps.get_page_text('https://www.cayic.com/yesil-cayin-faydalari')

sentences = TextTool.get_sentences(text)


corpus_test = TextTool.get_words(sentences)


def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    corpus_words = [y for x in corpus for y in x]
    corpus_words = sorted(list(set(corpus_words)))
    num_corpus_words = len(corpus_words)    


    # ------------------

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}
    
    # ------------------
    # Write your implementation here.
    word2Ind = dict(zip(words, range(num_words)))
    M = np.zeros((num_words, num_words))
    
    for _ , sentences in enumerate(corpus):
        for i in range(len(sentences) - window_size +1):
            curr_word = sentences[i]
            if i == 0:
                for j in range(min(window_size, len(sentences))):
                    neighbor_word = sentences[i+j+1]
                    M[word2Ind[curr_word], word2Ind[neighbor_word]] = 1
            
            elif i == len(sentences) - window_size:
                for j in range(min(window_size, len(sentences))):
                    neighbor_word = sentences[i-j-1]
                    M[word2Ind[curr_word], word2Ind[neighbor_word]] = 1   
                
            else:
                for j in range(min(window_size, len(sentences))):
                    neighbor_word1 = sentences[i-j-1]
                    neighbor_word2 = sentences[i+j+1]
                    M[word2Ind[curr_word], word2Ind[neighbor_word1]] = 1
                    M[word2Ind[curr_word], word2Ind[neighbor_word2]] = 1
    # ------------------

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
        # ------------------
        # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=10, random_state=42)
    M_reduced = svd.fit_transform(M)

    
        # ------------------

    print("Done.")
    return M_reduced



def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    X,Y = [], []
    for word in words:
        X.append(M_reduced[word2Ind[word], 0])
        Y.append(M_reduced[word2Ind[word], 1])
    X = np.array(X)
    
    fig, ax = plt.subplots()
    ax.scatter(X, Y, marker='.', color='blue')
    for i, name in enumerate(words):
        ax.annotate(name, (X[i], Y[i]))

    plt.show()

    # ------------------

M_test, word2Ind_test = compute_co_occurrence_matrix(corpus_test, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)
words = list(word2Ind_test.keys())
plot_embeddings(M_test_reduced, word2Ind_test, words)