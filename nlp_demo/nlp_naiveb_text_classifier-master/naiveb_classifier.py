# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 2019

@author: maliozer
"""

from tqdm import tqdm

import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer

#TRAINING

#file process
file_path = "files/tweetset.csv"
tweetsFrame = pd.read_csv(file_path)
#fix column_names
tweetsFrame.columns = ['polarity', 'id', 'date', 'query', 'username', 'tweet']
#drop irrelevant columns
tweetsFrame = tweetsFrame.drop(columns=['id','date','query','username'])

totalPositive = 0
totalNegative = 0

word_dict = dict()

stemmer = SnowballStemmer("english")

#reading Tweetdataframe to make df
for row in tqdm(tweetsFrame.iterrows(), total=tweetsFrame.shape[0]):
    df_record = row[1] #df row as series
    
    isPositive = df_record[0]
    sentence = df_record[1]
    
    if isPositive == 4:
        totalPositive += 1
    elif isPositive == 0:
        totalNegative += 1
    
    wordInSentence = nltk.word_tokenize(sentence)
    wordInSentence=[word.lower() for word in wordInSentence if word.isalpha()]    
    for word in wordInSentence:
        stemmedWord = stemmer.stem(word)
        
        if stemmedWord in word_dict:
            word_dict[stemmedWord][0] += 1
            
            if isPositive == 4:
                 word_dict[stemmedWord][1] += 1
            elif isPositive == 0:
                 word_dict[stemmedWord][2] += 1
            
            
        else:
            w_started = 3
            p_num = 1
            n_num = 1
            
            if isPositive == 4:
                p_num += 1
            elif isPositive == 0:
                n_num += 1
            
            word_dict[stemmedWord] = [w_started,p_num,n_num]


df_export = pd.DataFrame.from_dict(word_dict, orient='index')
df_export.columns = ['occurence', 'pos', 'neg']

#likelihood
df_export["likelihoodPos"] = df_export["pos"] / totalPositiveWords
df_export["likelihoodNeg"] = df_export["neg"] / totalNegativeWords


#negative log likelihood NLL
df_export['log_likePos'] = -1 * np.log(df_export.likelihoodPos)
df_export['log_likeNeg'] = -1 * np.log(df_export.likelihoodNeg)

logpriorPos = -1 * np.log(df_export.pos.sum() / df_export.occurence.sum())
logpriorNeg = -1 * np.log(df_export.neg.sum() / df_export.occurence.sum())

"""           
df_export = pd.DataFrame.from_dict(word_dict, orient='index')
df_export.columns = ['occurence', '+', '-']

#df_export["Ppositive"] = Series(np.random.randn(sLength), index=df1.index)


totalNegativeWords = df_export["-"].sum()
totalPositiveWords = df_export["+"].sum()
totalOccurence = df_export["occurence"].sum()

"""

#test

newSentence = "I think maliozer that you both will get on well with each other. "

wordInSentence = nltk.word_tokenize(newSentence)
wordInSentence=[word.lower() for word in wordInSentence if word.isalpha()]
testList = list()

for word in wordInSentence:
    stemmedWord = stemmer.stem(word)
    testList.append(stemmedWord)
  
sumPositive = 1
sumNegative = 1
for word in testList:
    if df_export[df_export.index == word].empty:
        next #pass the untrained word
    else:
        print(df_export[df_export.index == word])
    



