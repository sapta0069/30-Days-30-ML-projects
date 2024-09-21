#importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#importing the data into pandas

sd = pd.read_csv('DAY-3\sonar_data.csv',header=None)


# Data wrangling 

#print(sd.head())
#print(sd.shape)
#print(sd.describe()) # describe() gives the statistical info about the data
#print(sd[60].value_counts()) # gives us the individual count of rocks and mines
#print(sd.groupby(60).mean()) # calculate the mean of eachv group

# Seperating the data and labels
X = sd.drop(columns=60, axis=1)
Y  = sd[60]
#print(X)
#print(Y)

# Training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
#print(X_train.shape,X_test.shape)

# Model Training 

model = LogisticRegression()

# Training the logistic regression model with training data

model.fit(X_train,Y_train)

# Accuracy on trainiing data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data : ', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data : ', test_data_accuracy)

# Making a predictive system 

input_data = (0.0240,0.0218,0.0324,0.0569,0.0330,0.0513,0.0897,0.0713,0.0569,0.0389,0.1934,0.2434,0.2906,0.2606,0.3811,0.4997,0.3015,0.3655,0.6791,0.7307,0.5053,0.4441,0.6987,0.8133,0.7781,0.8943,0.8929,0.8913,0.8610,0.8063,0.5540,0.2446,0.3459,0.1615,0.2467,0.5564,0.4681,0.0979,0.1582,0.0751,0.3321,0.3745,0.2666,0.1078,0.1418,0.1687,0.0738,0.0634,0.0144,0.0226,0.0061,0.0162,0.0146,0.0093,0.0112,0.0094,0.0054,0.0019,0.0066,0.0023)
input_data_as_numpyarr = np.asarray(input_data) # converting the data into numpy array
input_data_reshaped = input_data_as_numpyarr.reshape(1,-1) # reshaping the data as we are testing it for one instance only

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print('The object is a rock')
else:
    print('The object is a mine')