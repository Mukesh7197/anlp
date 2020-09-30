#Find the sender of the email - Ram or Raj
#A new mail arrives with just three words - motivate, profit and product
#Historical information provided

import pandas as pd
data = [['motivate',0.24,0.05],['profit',0.3,0.35],['product',0.26,0.35],['leadership',0.08,0.15],['operations',0.12,0.10]]
df = pd.DataFrame(data, columns = ['Word','Ram','Raj'])
df.set_index('Word', inplace = True)
print(df)

#Create a wordlist for search words and calculate Bayesian probability for Ram and Raj
#Max value of Bayesian product will be the sender of the email
wordList = ['motivate', 'profit', 'product']
probRam = 1
probRaj = 1
for i in wordList:
    valRam = df.loc[i,'Ram']
    valRaj = df.loc[i,'Raj']
    probRam = valRam*probRam
    probRaj = valRaj*probRaj
print("Probability mail sent by Ram is: ", probRam)
print("Probability mail sent by Raj is: ", probRaj)
print("Mail sent by Ram") if probRam > probRaj else print("Mail sent by Raj")

#Product sentiments
#Assume the following likelihood for each word being part of positive or negative review
#Equal prior probabilities for each class (P(positive) = 0.5 and P(negative) =0.5)
#What class Naive Bayes classifier would assign to the sentence "I do not like to fill in the application form"
data2 = [['I',0.09,0.16],['love',0.07,0.06],['to',0.05,0.07],['fill',0.29,0.06],
        ['credit',0.04,0.15],['card',0.08,0.11],['application',0.06,0.04]]
df2 = pd.DataFrame(data2, columns=['Word','Positive','Negative'])
#df2.set_index('Word', inplace = True)
print(df2)

words = ['I','do','not','like','to','fill','in','the','application','form']
#Out of vocab words are: do, not, like, in, the, form (six words)

#Create two separate empty lists and populate them with matched vocabulary and out of vocabulary words
wordsMatch = []
wordsNoMatch = []

for i in words:
    if (df2['Word'] == i).any():
        wordsMatch.append(i)
    else:
        wordsNoMatch.append(i)

#print("List of matched words: ", wordsMatch)
#print("List of out of vocabulary words: ", wordsNoMatch)
#print("Total number of words: ", len(words))
#print("Number of matched words: ", len(wordsMatch))
#print("Number of out of vocabulary words: ", len(wordsNoMatch))

#Subset df2 containing matched words into a new dataframe newDF
newDF = pd.DataFrame(columns=['Word','Positive','Negative'])

for i in wordsMatch:
    if (df2['Word'].str.contains(i)).any():
        newDF = newDF.append(df2.loc[df2['Word'] == i])
        
#Create a new dataframe called oov (out of vocabulary) with words from wordsNoMatch as words and probability values 0.5
oovDF = pd.DataFrame(columns=['Word','Positive','Negative'])
for i in range(len(wordsNoMatch)):
    oovDF.loc[i] = [wordsNoMatch[i]] + [0.5] + [0.5]

#Concatenate newDF and oovDF into one single dataframe and set the index as Word column
frames = [newDF, oovDF]
merged = pd.concat(frames, ignore_index=True)
merged.set_index('Word',inplace = True)
print(merged)

#Calculate Bayesian probability for positive and negative reviews
words = ['I','do','not','like','to','fill','in','the','application','form']
probPos = 1
probNeg = 1
for i in words:
    valPos = merged.loc[i,'Positive']
    valNeg = merged.loc[i,'Negative']
    probPos = valPos*probPos
    probNeg = valNeg*probNeg
print("Probability Review is positive: ", probPos)
print("Probability Review is negative: ", probNeg)
print("Positive Review") if probPos > probNeg else print("Negative Review")
