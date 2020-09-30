import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

frequency={}
words_emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))

for word in words_emma:
    count = frequency.get(word,0)
    frequency[word] = count+1

rank=1
column_header = ['Word','Rank','Frequency','Rank*Frequency']
df = pd.DataFrame(columns=column_header)

for word,freq in reversed(sorted(frequency.items(),key=itemgetter(1))):
    df.loc[word]=[word,rank,freq,rank*freq]
    rank = rank+1
    
#Subset the data to contain top 10 frequent words are stored in a new dataframe df2
df2 = df.head(10)

#Store the frequent words and its frequency as a list for plotting purposes
words = []
freq = []
for d in df2.index.values:
    words.append(d)
for d in df2["Frequency"].values:
    freq.append(d)

# plot the data
y_pos = np.arange(len(words))
plt.bar(y_pos, freq, align='center', alpha=0.5)
plt.xticks(y_pos, words)
plt.ylabel('Occurence')
plt.xlabel('Words')
plt.title('Frequency of Top 10 Terms')
plt.show()

# Zipf's law - max value of 11454 as it was found out earlier 
freqrank = []
for r in range(1,11,1):
    max = 11454
    freqrank.append(max/r)

#A new dataframe created to store the frequency as estimated by Zipf law
df3 = pd.DataFrame({'words' : words, 'Zipf law' : freqrank})

# plot the frequency data of emma-austen corpus with bar representation for actual and line representation for Zipf's law  
df3.plot(kind='line', x='words',y='Zipf law',color='red')
#Bar chart for the vocabulary
y_pos = np.arange(len(words))
plt.bar(y_pos, freq, align='center', alpha=0.5)
plt.xticks(y_pos, words)
plt.ylabel('Occurence')
plt.title('Actual vs Zipf law - Top 10 Terms')
plt.show()

# Mandelbrot approximation
freqm = []
for r in range(1,11,1):
    max = 11454
    freqm.append(max/(r+2.7))

#A new dataframe created to store the frequency as estimated by Mandelbrot approximation
df4 = pd.DataFrame({'words' : words, 'Mandelbrot' : freqm})

# Mandelbrot approximation plot the data
plt.plot('words','Mandelbrot',data=df4, color='blue')
plt.plot('words','Zipf law',data=df3, color='red')
y_pos = np.arange(len(words))
plt.bar(y_pos, freq, align='center', alpha=0.5)
plt.xticks(y_pos, words)
plt.ylabel('Occurence')
plt.title('Actual vs Zipf law Vs Mandelbrot for Top 10 Terms')
plt.show()
