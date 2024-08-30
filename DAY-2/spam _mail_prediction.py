# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # converts the mail text into numerical values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection & Pre-processing
raw_mail_data = pd.read_csv('mail_data.csv')
print(raw_mail_data)
print(raw_mail_data.isnull().sum()) # There are no null values 
print(raw_mail_data.head())
print(raw_mail_data.shape)

# label spam mail as 0 , ham mail as 1 

raw_mail_data.loc[raw_mail_data['Category'] == 'spam' , 'Category']= '0'
raw_mail_data.loc[raw_mail_data['Category'] == 'ham' , 'Category']= '1'

# seperating the data as texts and label 

X = raw_mail_data['Message']

Y = raw_mail_data['Category']

print(X)

print(Y)

# Splitting the data into training data and test data

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,train_size=0.2,random_state=3)

# Feature Extraction i.e. transform the text data to feature vectors that can be used as input 
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_tset into integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#print(X_train_features)

# Training the logistic Regression Model
model = LogisticRegression()
model.fit(X_train_features,Y_train)

#Prediction on training data 
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

#Prediction on test data 
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

# Building a predictive System 
input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

input_data_features = feature_extraction.transform(input_mail)


prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')






