####################################################################################################
#Major idea of coding is based on ANLP classes conducted by Prof. Ramaseshan as part of NPTEL
#Code improvements in terms of using scikit learn Perceptron model, SGD, Single layer NN,
#Two Layer NN and SVM

#Binary classifier

#Python script to check whether a given word is positive or negative (Sentiment Analysis)
#Collect words that are positive or negative from open source
#Use GloVe to get the word embedding
#Check whether word in GloVe exists in the positive or negative words collected
#If GloVe word is classified as positive, append 1 else append 0 for negative
#Develop a Perceptron model using sklearn
#Split the data available into training(80%) and testing(20%)
#Fit the model and evaluate accuracy of model
####################################################################################################
IMPORT LIBRARIES
####################################################################################################
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
####################################################################################################
#Website to pick up positive and negative words for this exercise

#http://ptrckprry.com/course/ssd/data/negative-words.txt 
#http://ptrckprry.com/course/ssd/data/positive-words.txt

#Download and store this file as Positive.txt and Negative.txt in current working directory
####################################################################################################

with open('Positive.txt', 'r') as f2:
    pos = f2.read()
positive = list(pos.splitlines())

with open('Negative.txt', 'r') as f2:
    neg = f2.read()
negative = list(neg.splitlines())

print("Number of positive words: ", len(positive)) #2006
print("Number of negative words: ", len(negative)) #4783
print("Total number of words: ", len(positive) + len(negative)) #6789

print("First ten words in Positive List: \n", positive[:10])
print("First ten words in Negative List: \n", negative[:10])

#First ten words in Positive List: 
 #['a+', 'abound', 'abounds', 'abundance', 'abundant', 'accessable', 'accessible', 'acclaim', 'acclaimed', 'acclamation']
#First ten words in Negative List: 
 #['2-faced', '2-faces', 'abnormal', 'abolish', 'abominable', 'abominably', 'abominate', 'abomination', 'abort', 'aborted']
 
#Include words that contain only alphabets
#Total words for consideration shall reduce from 6789 to 6562
positive_words = [w for w in positive if w.isalpha()]
negative_words = [w for w in negative if w.isalpha()]
print("First ten words in Positive List: \n", positive_words[:10])
print("First ten words in Negative List: \n", negative_words[:10])
print("Total number of words: ", len(positive_words) + len(negative_words)) #6562

#First ten words in Positive List: 
 #['abound', 'abounds', 'abundance', 'abundant', 'accessable', 'accessible', 'acclaim', 'acclaimed', 'acclamation', 'accolade']
#First ten words in Negative List: 
 #['abnormal', 'abolish', 'abominable', 'abominably', 'abominate', 'abomination', 'abort', 'aborted', 'aborts', 'abrade']
#Total number of words:  6562

####################################################################################################
# Visit https://nlp.stanford.edu/projects/glove/

#GloVe is an unsupervised learning algorithm for obtaining vector representations for words. 
#Training is performed on aggregated global word-word co-occurrence statistics from a corpus, 
#and the resulting representations showcase interesting linear substructures of the word vector space.

# Download glove.6B.zip available under download pre-trained word vectors
# Unzip the file to get 4 text files: glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt
# Store these files in the current working directory
####################################################################################################

# Open the file glove.6B.50d.txt

f = open('glove.6B.50d.txt', encoding = 'utf-8')
emb_dict = {}

#For every line in glove
#Check if the word in glove is present in either positive or negative wordlist
#If word from glove is present in positive_words append the vector with 1.0
#If word from glove is present in negative_words append the vector with 0.0

#This code takes few minutes to execute depending upon your computer configuration

for line in f:
    values = line.split(' ')
    word = values[0] ## The first entry is the word
    vector = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
    if word in positive_words:
        vector = np.append(vector, [1.0]) #Append 1.0 as last element if GloVe word is positive
        emb_dict[word] = vector
    if word in negative_words: 
        vector = np.append(vector, [0.0]) #Append 0.0 as last element if GloVe word is positive
        emb_dict[word] = vector
f.close()

#Only 6088 words in GloVe are available out of 6562 words available in list of positive_words and negative_words

print("Number of words in the embedded dictionary created: ", len(emb_dict)) #6088

#Check the content of elements of specific word and check for inclusion of 1 (positive) or 0 (negative)
print("One Hot Vector code for word support: \n", emb_dict["support"])
print("Number of elements for word support: ", len(emb_dict["support"])) #51 elements. Last element is either 1 or 0 depending upon positive or negative
print("One Hot Vector code for word abnormal: \n", emb_dict["abnormal"])
print("Number of elements for word abnormal: ", len(emb_dict["abnormal"])) #51 elements

####################################################################################################

#Define a function to predict sentiment of the word given. This is based on the last element of vector hot coded 

def predict_Sentiments(words):
    senti = emb_dict[words]
    print(words, "is Positive") if senti[50] == 1 else print(words, "is Negative")

#Checking whether words are positive or negative
predict_Sentiments("support") #support is Positive
predict_Sentiments("abnormal") abnormal is Negative

####################################################################################################

#Features are the one hot vector numbers(first 50 elements) associated with the word
#Labels are the last element in the value in dictionary
#A function that takes word as input and returns its 50d vector as features and 1d scalar as label

def features_and_labels(words):
    senti = emb_dict[words]
    features = senti[:50]
    labels = senti[50]
    return features, labels
####################################################################################################

#keys variable shall hold all the 6088 words in GloVe which can be resolved as positive or negative

keys = list(emb_dict.keys())

#Collect all features of a word (50 dimensions) in X variable
#Collect all labels (either 1 for Positive or 0 for negative in y variable)

X = []
y = []
for i in keys:
    features, labels = features_and_labels(i) #Call the function
    X.append(features)
    y.append(labels)
    
####################################################################################################

#Word is not available. Instead its features as a 50d vector and label as a 1d scalar are available
#Split the data 80% for training and 20% for testing for developing a perceptron model 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

totSam = len(X)
totPosSam = sum(y)
totNegSam = totSam - totPosSam

totTrgSam = len(X_train)
totPosTrgSam = sum(y_train)
totNegTrgSam = totTrgSam - totPosTrgSam

totTestSam = len(X_test)
totPosTestSam = sum(y_test)
totNegTestSam = totTestSam - totPosTestSam

total = ['Total', totSam, totPosSam, totNegSam]
training = ['Training', totTrgSam, totPosTrgSam, totNegTrgSam]
test = ['Test', totTestSam,totPosTestSam, totNegTestSam]

mylist = [total, training, test]

data = pd.DataFrame(mylist, columns = ['Type of Sample', 'Number of Samples', 'Positive', 'Negative'])

data.plot('Type of Sample' , ['Number of Samples','Positive','Negative'], kind='bar')

plt.show()
#Approximately 30% of total and training samples are positive
#Approximately 29% of test samples are positive. #So, majority sample of words are negative

####################################################################################################
#Make a perceptron classifier with learning rate as eta and max number of iterations = 5000
#Fit the features into the perceptron model

clf = Perceptron(eta0=0.00001, max_iter=5000)
clf.fit(X_train, y_train)

#Perceptron(eta0=1e-05, max_iter=5000)
####################################################################################################

#Predict the model accuracy on training and test data set
clf.score(X_train,y_train) #Training accuracy is 83.14%
clf.score(X_test,y_test) #Testing accuracy is 81.85%

####################################################################################################

# Function to check given words are positive or negative
# Collects the 50 dimensional vector from GloVe corresponding to the word
# Uses the model to predict whether the given word is positive or negative


def predSent(word):
    senti = emb_dict[word]
    val = clf.predict(senti[:-1].reshape(1,-1))
    print(word, "is Positive") if val == 1 else print(word, "is Negative")    
 
####################################################################################################
#Test few words
predSent("abnormal")
predSent("support")
####################################################################################################
#Let us create 10 test words in random from a list called keys
#Everytime this portion is run, it randomly generates 10 words from keys

wordTest = []
for i in range(10):
    n =random.randint(0,6088)
    word = keys[n]
    wordTest.append(word)
print(wordTest)

####################################################################################################

#Test these words for Sentiment analysis
#Training and testing accuracy is 83.14% and 81.14% respectively. 
#Errors in classification at times can be observed

for i in wordTest:
    predSent(i)
    
####################################################################################################
#Parameters of the model
clf.get_params()

#Weights associated with the model
print("Weights assigned by sklearn Perceptron: \n", clf.coef_[0])

####################################################################################################
#Mis classification errors in the test dataset

y_predict = clf.predict(X_test)
cnt = 0
for i in range(len(X_test)):
    if y_predict[i] != y_test[i]:
        cnt = cnt + 1
print("Total number of misclassification with Perceptron classifier: ", cnt) #221

####################################################################################################

#Using confusion matrix

yp = clf.predict(X_test)
y_actual =  [int(i) for i in y_test]
y_predict = [int(i) for i in yp]
cm = confusion_matrix(y_actual, y_predict)
indVal = ['Negative','Positive']
colVal = ['Negative','Positive']
df = pd.DataFrame(cm, index = indVal, columns=colVal)
print("Confusion Matrix for the test dataset \n")
print(df)

#Confusion Matrix for the test dataset 

#          Negative  Positive
#Negative       730       137
#Positive        84       267

####################################################################################################

#Classifies 730 words as Negative which is actually negative
#Classifies 137 words as Positive which is actually negative
#Classifies  84 words as Negative which is actually positive
#Classifies 267 words as Positive which is actually positive

#Total misclassification error is 84 + 137 = 221

####################################################################################################
STOCHASTIC GRADIENT CLASSIFIER
####################################################################################################

#Make a pipeline with standard scaler
#Model shall learn for a maximum iteration of 1000 with loss stopping criterion as 1e-3 

clfSGD = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
clfSGD.fit(X_train, y_train)

#Training and test accuracy of SGD classifier is more than the Perceptron Classifier
trgAccSGD = (clfSGD.score(X_train, y_train))*100#Training accuracy is 87.23%
testAccSGD = (clfSGD.score(X_test,y_test))*100 #Test accuracy is 88.34%

# Function to check given words are positive or negative
def predSentSGD(word):
    senti = emb_dict[word]
    val = clfSGD.predict(senti[:-1].reshape(1,-1))
    print(word, "is Positive") if val == 1 else print(word, "is Negative")   

#Mis classification error has reduced significantly from 221 to 142
y_predSGD = clfSGD.predict(X_test)
cnt = 0
for i in range(len(X_test)):
    if y_predSGD[i] != y_test[i]:
        cnt = cnt + 1
mcSGD = cnt
print("Total number of misclassification with SGD classifier: ", mcSGD)

####################################################################################################
SINGLE LAYER NEURAL NETWORK
####################################################################################################

#Input layer 50 dim vector, one hidden layer with 128 units and output layer with 2 units
#Model will be trained for 5000 epochs with learning rate of 0.00001 and lbfgs solver

clfMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128), random_state=1, max_iter=5000)
clfMLP.fit(X_train,y_train)

#Training accuracy is 100% (Overfitting data)
#However test accuracy of MLP classifier is more than the Perceptron and SGD Classifier

trgAccMLP = (clfMLP.score(X_train, y_train))*100#Training accuracy is 100% (overfitting)
testAccMLP = (clfMLP.score(X_test,y_test))*100 #Test accuracy is 87.03% (Better than SGD)

# Function to check given words are positive or negative
def predSentMLP(word):
    senti = emb_dict[word]
    val = clfMLP.predict(senti[:-1].reshape(1,-1))
    print(word, "is Positive") if val == 1 else print(word, "is Negative")
 
#Mis classification error increased to 158 FROM 142 
y_predMLP = clfMLP.predict(X_test)
cnt = 0
for i in range(len(X_test)):
    if y_predMLP[i] != y_test[i]:
        cnt = cnt + 1
mcMLP = cnt
print("Total number of misclassification in MLP classifier with single hidden layer: ", mcMLP)

####################################################################################################
TWO LAYER NEURAL NETWORK TO CHECK IMPROVEMENT IN TEST ACCURACY
####################################################################################################

clfMLP2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128, 128), random_state=1, max_iter=5000)
clfMLP2.fit(X_train,y_train)

#Training accuracy of MLP classifier with two hidden layers with 128 units each is 100%
#However test accuracy is around 87.93%. Marginal improvement

trgAccMLP2 = (clfMLP2.score(X_train, y_train))*100#Training accuracy is 100% (overfitting)
testAccMLP2 = (clfMLP2.score(X_test,y_test))*100 #Test accuracy is 87.93%

# Function to check given words are positive or negative
def predSentMLP2(word):
    senti = emb_dict[word]
    val = clfMLP2.predict(senti[:-1].reshape(1,-1))
    print(word, "is Positive") if val == 1 else print(word, "is Negative")   
    
#Mis classification error has decreased to 147 from 158 (Single Layer) but higher compared to SGD(142)
y_predMLP2 = clfMLP2.predict(X_test)
cnt = 0
for i in range(len(X_test)):
    if y_predMLP2[i] != y_test[i]:
        cnt = cnt + 1
mcMLP2 = cnt
print("Total number of misclassification in MLP classifier with two hidden layers: ", mcMLP2)

####################################################################################################
SVM CLASSIFIER
####################################################################################################

clfSVM = svm.SVC()
clfSVM.fit(X_train,y_train)

#Training and test accuracy of SVM scores the highest 
trgAccSVM = (clfSVM.score(X_train, y_train))*100#Training accuracy is 93.78%
testAccSVM = (clfSVM.score(X_test,y_test))*100 #Test accuracy is 89.66%

# Function to check given words are positive or negative
def predSentSVM(word):
    senti = emb_dict[word]
    val = clfSVM.predict(senti[:-1].reshape(1,-1))
    print(word, "is Positive") if val == 1 else print(word, "is Negative")
    
 #Mis classification error of SVM is the lowest at 126
y_predSVM = clfSVM.predict(X_test)
cnt = 0
for i in range(len(X_test)):
    if y_predSVM[i] != y_test[i]:
        cnt = cnt + 1
mcSVM = cnt
print("Total number of misclassification in SVM classifer: ", mcSVM)

####################################################################################################
#Store the training, test accuracy and misclassification errors of the models in a dataframe
####################################################################################################

perc = ['Perc', trgAccPerc, testAccPerc, mcPerc]
SGD = ['SGD', trgAccSGD, testAccSGD, mcSGD]
MLPSingle = ['Single Layer', trgAccMLP, testAccMLP, mcMLP]
MLPTwo = ['Two Layers', trgAccMLP2, testAccMLP2, mcMLP2]
SVM = ['SVM', trgAccSVM, testAccSVM, mcSVM]

mdl = ['Model Name','Trg Acc', 'Test Acc', 'Misclassification Errors']
           
listModels = [perc, SGD, MLPSingle, MLPTwo, SVM]

modelsDF = pd.DataFrame(listModels, columns = mdl)

####################################################################################################
#Plot training, test accuracy and misclassification errors of the models 
####################################################################################################

fig, (ax1, ax2, ax3) =plt.subplots(1,3, figsize = (20,10))
ax1.bar(modelsDF['Model Name'] , modelsDF['Trg Acc'], color='g')
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Model Name")
ax1.set_title("Training Accuracy")
ax2.bar(modelsDF['Model Name'] , modelsDF['Test Acc'],color='m')
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Model Name")
ax3.plot(modelsDF['Model Name'], modelsDF['Misclassification Errors'])
plt.xlabel("Model Name")
plt.ylabel("Number of Errors")
plt.title("Misclassification Errors")
plt.show()

#Highest training accuracy of 100% is MLP with either one or two Layers. Each layer has 128 units. 
#However its test accuracy is lower and hence misclassification error is higher compared to SVM 
#SVM has comparable training accuracy but slightly better test accuracy and lowest misclassification error.
#Best model is SVM as it has lowest misclassification error with highest test accuracy.
####################################################################################################
