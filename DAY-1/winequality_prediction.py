import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('winequalityN.csv') # reads the file or document 


for col in df.columns:
    if df[col].isnull().sum() > 0:  # That means those columns having null values will be replaced by the average of that cokumn
        df[col]=df[col].fillna(df[col].mean())

df.hist(bins=20, figsize=(10, 10))
#plt.show()

df = df.drop(columns=['type']) # type was of value str so no need to make correlation ,you will get error 

plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f",
linewidths=0.5) # A correlation heatmap is a heatmap that shows a 2D correlation matrix between two discrete dimensions, using colored cells to represent data from usually a monochromatic scale.
#plt.show()

df.quality.unique() 

# So, there are 7 unique values which is not good. So, what we can do is, we can consider 1 if quality is above 5 and 0 if it is below 5. It’s a classification problem

df['best quality'] = [1 if x > 5 else 0 for x in df['quality']]


df.replace({'white': 1, 'red': 0}, inplace=True) # to handle the categorical column , we can do this

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2, random_state=89)

#print(xtrain.shape, xtest.shape) # this is done let us know the number of training and testing data along with the sample
 
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest) # normalizing the data


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
	models[i].fit(xtrain, ytrain)

	print(f'{models[i]} : ')
	print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
	print('Validation Accuracy : ', metrics.roc_auc_score(
		ytest, models[i].predict(xtest)))
	print()
     

     

 







     
