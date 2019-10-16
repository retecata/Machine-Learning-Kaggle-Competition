import numpy as np
import pandas
import math
import collections,csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

def drop_irrelevant_columns(df):
    del df['Hair Color']
    del df['Wears Glasses']
    del df['Instance']
    return df

def transform_categorical_data(df):
    # Transform profession data into categories codes
    df["Profession"] = df["Profession"].astype('category')
    df["profession_cat"] = df["Profession"].cat.codes

    df.replace(df["Profession"],df["profession_cat"])
    del df["Profession"]
    # Transform categorical data into numeric data
    df = pandas.get_dummies(data=df, sparse=False, drop_first=True, dummy_na=False)
    return df

df = pandas.read_csv('trainingdata.csv')
df.fillna(method='ffill',inplace=True)
#Should I do this or not?
df[df['Income in EUR'] < 0] = 0.001
df = drop_irrelevant_columns(df)
df['Gender'].replace('0','male', inplace=True)
df = transform_categorical_data(df)



X=df.drop(['Income in EUR'], axis=1)
y=df['Income in EUR']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test

#df.to_csv(r'scores.csv')
# Create linear regression object
print("create")
regressor = RandomForestRegressor(random_state=1234, n_jobs=-1, n_estimators=100, min_samples_split=2, min_samples_leaf=1, verbose=10)
print("Finished creating")
#clf = svm.SVC(gamma='scale')
y_train = y_train.apply(np.log)

# Train the model using the training sets
print("stat training")
regressor.fit(X_train, y_train)

#clf.fit(X_train,y_train)


print("RandomForestRegressor:" ,regressor.score(X_test,y_test.apply(np.log)))

# The mean squared error
#print("Mean squared error: %.2f"
     # % mean_squared_error(y_test,np.exp(pred)))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, np.exp(pred)))
