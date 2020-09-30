import nltk
from nltk import bigrams
from nltk import trigrams
from nltk import ngrams
from nltk import FreqDist
from nltk.util import pad_sequence

#Create a corpus
sentCorpus = ['I am Sam','Sam I am', 'I do not like green eggs and ham']

#Split the padded sentences into words 
words = []
for i in range(0,len(sentCorpus)):
    #Split the strings based on blankspace
    sen = sentCorpus[i].split(' ')
    #Pad at either end of the sentence with markers
    sent = list(pad_sequence(sen, pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>", n=2))
    #Extend the list by adding
    words.extend(sent)
print(words)

#Unigrams, bigrams, trigrams, quadgrams
listU = words
listB = list(bigrams(words))
listT = list(trigrams(words))
listQ = list(ngrams(words,4))

#Get total number of unigrams, bigrams, trigrams and quadgrams
cntU = len(listU)
cntB = len(listB)
cntT = len(listT)
cntQ = len(listQ)
print("Count of Unigrams: ", cntU, "Bigrams: ", cntB, "Trigrams: ", cntT, "Quadgrams: ",cntQ)
print("\nList of unigrams: ", listU)
print("\nList of Bigrams: ",listB)
print("\nList of Trigrams: ",listT)
print("\nList of Quadgrams: ",listQ)

#Frequency distribution of unigrams, bigrams, trigrams and quadgrams
fDist1 = nltk.FreqDist(listU)
fDist2 = nltk.FreqDist(listB)
fDist3 = nltk.FreqDist(listT)
fDist4 = nltk.FreqDist(listQ)

#Function to get count of specific unigram, bigrams, trigrams and quadgrams
def cntUgm(w1):
    for word, frequency in fDist1.most_common():
        if word == w1:
            f = frequency
    return f

def cntBgm(w1,w2):
    for word, frequency in fDist2.most_common():
        if word == (w1,w2):
            f = frequency
    return f

def cntTgm(w1,w2,w3):
    for word, frequency in fDist3.most_common():
        if word == (w1,w2,w3):
            f = frequency
    return f

def cntQgm(w1,w2,w3,w4):
    for word, frequency in fDist4.most_common():
        if word == (w1,w2,w3,w4):
            f = frequency
    return f
 
print("Number of times the Unigram 'Sam' occured: ", cntUgm('Sam'))
print("Number of times the Bigram 'I am' occured: ",cntBgm('I', 'am'))
print("Number of times the Trigram 'I am Sam' occured: ",cntTgm('I', 'am', 'Sam'))

#Bigram probabilities
def bgmProb(w1,w2):
    totCnt = 0
    bgmCnt = 0
    for key, value in fDist2.items():
        keylist = list(key)
        if keylist[0] == w1:
            totCnt = totCnt + value
            if keylist[1] == w2:
                bgmCnt = bgmCnt + value
    print("Probability of", w2, "|", w1)
    return bgmCnt/totCnt
    
print(bgmProb('<s>','I'))
print(bgmProb('<s>','Sam'))
print(bgmProb('I','am'))
print(bgmProb('Sam','</s>'))
print(bgmProb('am','Sam'))
print(bgmProb('I','do'))

#Trigram probabilities
def tgmProb(w1,w2,w3):
    totCnt = 0
    tgmCnt = 0
    for key, value in fDist3.items():
        keylist = list(key)
        if keylist[0] == w1:
            if keylist[1] == w2:
                totCnt = totCnt + value
                print(keylist, value)
                if keylist[2] == w3:
                    tgmCnt = tgmCnt + value
                    
    print("Probability of", w3, "|", w1, w2)
    return tgmCnt/totCnt
    
#Calculate Probability of occurrence of Sam given I am has occurred earlier
tgmProb('I','am','Sam')

#Quadgram probabilities
def qgmProb(w1,w2,w3,w4):
    totCnt = 0
    qgmCnt = 0
    for key, value in fDist4.items():
        keylist = list(key)
        if keylist[0] == w1:
            if keylist[1] == w2:
                if keylist[2] == w3:
                    totCnt = totCnt + value
                    print(keylist, value)
                    if keylist[3] == w4:
                        qgmCnt = qgmCnt + value
                    
    print("Probability of", w4, "|", w1, w2, w3)
    return qgmCnt/totCnt
  
  qgmProb('I','do','not','like')
