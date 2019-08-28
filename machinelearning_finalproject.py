import pickle
import pubchempy as pcp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD
import numpy as np
from sklearn import svm
import keras

testdata = pd.read_pickle('my_df.pickle')

#setting x and y datasets
xvalues = testdata.drop(['RT','isomeric_smiles'],axis=1)
yvalues = testdata[['RT']]


#set training data
st = StandardScaler()
xvalues = testdata.drop(['RT','isomeric_smiles','mol','fingerprint','cactvs_fingerprint'],axis=1)
xvalues = st.fit_transform(xvalues)
xtrain, xtest, ytrain, ytest = train_test_split(xvalues, yvalues,test_size=0.3, random_state = 38)

#builiding a simple nueral network
model = Sequential()
model.add(Dense(output_dim=5, input_dim=xvalues.shape[1]))
model.add(Activation("sigmoid"))
model.add(Dense(output_dim=1))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))
history = model.fit(xtrain, ytrain, nb_epoch=10000, batch_size=32)
y_pred = model.predict(xtest)
y_preddf = pd.DataFrame(y_pred)
ytest=ytest.reset_index()
ytest = ytest.drop(['index'],axis=1)
resultsdf = pd.concat([ytest,y_preddf],axis=1)
rms = (np.mean((ytest - y_pred)**2))**0.5
#s = np.std(y_test -y_pred)
print('Neural Network RMS', rms)
#record results
correct = 0 
incorrect = 0
listofper = []
for index,row in resultsdf.iterrows():
    percent = 0
    pred = row[0]
    actu = row['RT']
    percent = ((pred-actu)/actu) * 100
    diff = pred - actu
    listofper.append(percent)
    if diff < 5.00 and diff > -5.00:
        correct += 1
    else:
        incorrect += 1
resultsdf['CVsimplesNN'] = listofper
samples = len(ytest)
pos = (correct/samples) *100
neg = (incorrect/samples) * 100
print('accuracy is ' + str(pos))  

#models in a loop
sgd = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
ada = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
adad = keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adamax = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
epochs = 10000
finalresults = pd.DataFrame()
optimizers = [ada,adad,adam,adamax,nadam,rms,sgd]
accuracy = []
for count,o in enumerate(optimizers,0):
    model = Sequential()
    model.add(Dense(output_dim=5, input_dim=xvalues.shape[1]))
    model.add(Activation("sigmoid"))
    model.add(Dense(output_dim=1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer=o)
    history = model.fit(xtrain, ytrain, nb_epoch=epochs, batch_size=32)
    y_pred = model.predict(xtest)
    y_preddf = pd.DataFrame(y_pred)
    ytest=ytest.reset_index()
    ytest = ytest.drop(['index'],axis=1)
    interdf = pd.concat([ytest,y_preddf],axis=1)
    correct = 0 
    incorrect = 0
    listofper = []
    listofpre = []
    for index,row in interdf.iterrows():
        pred = row[0]
        listofpre.append(pred)
        actu = row['RT']
        percent = ((pred-actu)/actu) * 100
        diff = pred - actu
        listofper.append(percent)
        if diff < 5.00 and diff > -5.00:
            correct += 1
        else:
            incorrect += 1
        samples = len(ytest)
        pos = (correct/samples) *100
    accuracy.append([count,pos])
    resultsdf['cv ' + str(count)] = listofper
    resultsdf['predictions ' + str(count)] = listofpre
print(accuracy)   
resultsdf.to_pickle('results_df.pickle')     
    

#plot results
'''
import matplotlib.pyplot as plt
plt.scatter(ytrain,model.predict(xtrain), label = 'Train', c='blue')
plt.title('Neural Network Predictor')
plt.xlabel('Measured RT')
plt.ylabel('Predicted RT')
plt.scatter(ytest,model.predict(xtest),c='lightgreen', label='Test', alpha = 0.8)
plt.legend(loc=4)
plt.show()
'''



## svm
