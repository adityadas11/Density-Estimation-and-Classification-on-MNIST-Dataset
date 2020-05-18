#!/usr/bin/env python
# coding: utf-8

# In[42]:


import scipy
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn


# In[43]:


#loading dataset from file
npData = loadmat("mnist_data.mat")


# In[44]:


npData


# In[45]:


#extracting the training and testing data from the dataset
trainX = npData['trX']
trainY = npData['trY'][0]
testX = npData['tsX']
testY = npData['tsY'][0]


# In[46]:


#partioning data of 7 and 8 from the training dataset
sample7 = []
sample8 = []
for i in range(len(trainX)):
    if trainY[i] == 0:
        sample7.append(trainX[i])
    else:
        sample8.append(trainX[i])
        


# In[47]:


# extract two features from both samples
feat7 = np.zeros((len(sample7),2))
feat8 = np.zeros((len(sample8),2))


# In[53]:


#finding out the mean and standard deviation from each of these feature matrices
for i in range(len(sample7)):
    feat7[i][0] = np.mean(sample7[i])
    feat7[i][1] = np.std(sample7[i])
for i in range(len(sample8)):
    feat8[i][0] = np.mean(sample8[i])
    feat8[i][1] = np.std(sample8[i])


# In[54]:


#storing the prior probability of each sample which will be used in the Bayes formula
prob7 = len(sample7)/len(trainX)
prob8 = len(sample8)/len(trainX)
print(feat8)


# In[55]:


#calculating the mean and variance of each feature for drawing the gaussian distribution
mean7 = np.array([np.mean(feat7[:,0]),np.mean(feat7[:,1])])
var7 = np.array([np.var(feat7[:,0]),np.var(feat7[:,1])])
mean8 = np.array([np.mean(feat8[:,0]),np.mean(feat8[:,1])])
var8 = np.array([np.var(feat8[:,0]),np.var(feat8[:,1])])


# In[56]:


print(mean7)
print(var7)
print(mean8)
print(var8)


# In[57]:


#calculating the probability of each test data
totCorrect = 0
numofCorrect7 = numofCorrect8 = 0
for i in range(len(testX)):
    featTest = np.array([np.mean(testX[i]),np.std(testX[i])])
    #use the PDF to determine the probability 
    postProb7 = mvn.logpdf(featTest, mean=mean7, cov=var7)+np.log(prob7)
    postProb8 = mvn.logpdf(featTest, mean=mean8, cov=var8)+np.log(prob8)
    maxProb = 0
    if postProb7 > postProb8:
        maxProb = 0
    else:
        maxProb = 1
    if testY[i] == maxProb:
        totCorrect += 1
        if testY[i] == 0:
            numofCorrect7 += 1
        else:
            numofCorrect8 += 1
        


# In[58]:


#segregating 7 and 8 test data
test7 = []
test8 = []
for i in range(len(testX)):
    if testY[i] == 0:
        test7.append(testX[i])
    else:
        test8.append(testY[i])


# In[61]:


#Finally calculating the accuracy of Naive Bayes Classifer
naiveAccuracy = totCorrect/len(testY)
naiveAccuracyOf7 = numofCorrect7/len(test7)
naiveAccuracyOf8 = numofCorrect8/len(test8)
print(naiveAccuracy)


# In[126]:


# Sigmoid function
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


# In[179]:


#Logistic Regression
#Defining the Logistic regression function
def logisticRegression(features, target):
    # Setting number of epochs to 100000
    epochs = 100000
    # Setting learning rate to 0.003
    learningRate = 0.003
    
    # Adding intercept to image features
    intercept = np.ones((features.shape[0], 1))
    features = np.hstack((intercept, features))

    # Initializing the weights to zeros first
    weights = np.zeros(features.shape[1])

    # Training the logistic regression model
    for epoch in range(epochs):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learningRate * gradient
    
    return weights


# In[180]:


#defining the log likelihood
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    logLikelihood = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return logLikelihood


# In[181]:


#Making a matrix containing features of 7 and 8
X = np.concatenate((feat7,feat8))
#making Y
Y1 = np.zeros((len(sample7),1))
Y2 = np.ones((len(sample8),1))
Y = np.squeeze(np.asarray(np.concatenate((Y1, Y2))))


# In[188]:


#Training the model and getting the weights
weights = logisticRegression(X,Y)
#print(weights)


# In[183]:


#Extracting feature of test data
featTestX = np.zeros((len(testX),2))
featTestX7 = np.zeros((len(test7),2))
featTestX8 = np.zeros((len(test8),2))
for i in range(len(testX)):
    featTestX[i] = np.array([np.mean(testX[i]), np.std(testX[i])])
for i in range(len(test7)):
    featTestX7[i] = np.array([np.mean(test7[i]), np.std(test7[i])])
for i in range(len(test8)):
    featTestX8[i] = np.array([np.mean(test8[i]), np.std(test8[i])])


# In[184]:


#adding intercept to data
dataIntercept = np.hstack((np.ones((featTestX.shape[0], 1)),
                                 featTestX))
dataIntercept7 = np.hstack((np.ones((featTestX7.shape[0], 1)),
                                 featTestX7))
dataIntercept8 = np.hstack((np.ones((featTestX8.shape[0], 1)),
                                 featTestX8))


# In[185]:


#finding accuracy of Logistic Regression
finalScores = np.dot(dataIntercept, weights)
pred = np.round(sigmoid(finalScores))
accuracyLR = (pred == testY).sum().astype(float) / len(pred)

finalScores7 = np.dot(dataIntercept7, weights)
pred7 = np.round(sigmoid(finalScores7))
accuracyLR7 = (pred7 == 0).sum().astype(float) / len(pred7)

finalScores8 = np.dot(dataIntercept8, weights)
pred8 = np.round(sigmoid(finalScores8))
accuracyLR8 = (pred8 == 1).sum().astype(float) / len(pred8)


# In[186]:


#Printing all the accuracy in one place
print('Accuracy of Naive Bayes classifier : %f %%' % (naiveAccuracy*100))
print('Accuracy of Naive Bayes classifier for 7: %f %%' % (naiveAccuracyOf7*100))
print('Accuracy of Naive Bayes classifier for 8: %f %%' % (naiveAccuracyOf8*100))
print('Accuracy of Logistic Regression classifier : %f %%' % (accuracyLR*100))
print('Accuracy of Logistic Regression classifier for 7: %f %%' % (accuracyLR7*100))
print('Accuracy of Logistic Regression classifier for 8: %f %%' % (accuracyLR8*100))


# In[ ]:




