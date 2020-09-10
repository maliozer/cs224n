import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)



def get_sentences(text):
    text_list = text.split('.')
    text_words_list = [d.split(' ') for d in text_list]
    return text_words_list

def get_words(documents):
    return [[START_TOKEN] + [w.lower() for w in d] + [END_TOKEN] for d in documents]


