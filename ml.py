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
    del df['Size of City']
    del df['Instance']
    return df

def group(df,cols,threshold):
    for col in cols:
        counts = df[col].value_counts()
        index = counts[counts<=threshold].index
        df[col] = df[col].replace(index,"other")
    return df

def keep_known_columns(df,cols):
    for col in cols:
        df = df[pandas.notnull(df[col])]
    return df

# Read training data
df = pandas.read_csv('trainingdata.csv')

# Keep columns with known information
df['Year of Record'].fillna(round(df['Year of Record'].mean()),inplace=True)
df['Age'].fillna(round(df['Age'].mean()),inplace=True)
df['Body Height [cm]'].fillna(round(df['Body Height [cm]'].mean()),inplace=True)
df = keep_known_columns(df,['Year of Record','Gender','Age','Country','Profession','University Degree','Body Height [cm]'])
df = drop_irrelevant_columns(df)
print(df[df.isna().any(axis=1)])

# Group data together
df['Gender'].replace('0','male', inplace=True)
df['Gender'].replace('unknown','other', inplace=True)
df = group(df,["Profession","Country","Gender","University Degree"],4)
df = pandas.get_dummies(data=df, drop_first=False)

#df.to_csv(r'test3.csv',index=False)

df = df[df['Income in EUR'] >= 100]
df = df[df['Income in EUR'] <=4000000]



X=df.drop(['Income in EUR'], axis=1)
y=df['Income in EUR']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test

#df.to_csv(r'scores.csv')
# Create linear regression object
regr = linear_model.LinearRegression()



#clf = svm.SVC(gamma='scale')
y_train = y_train.apply(np.log)

# Train the model using the training sets
regr.fit(X_train, y_train)


#clf.fit(X_train,y_train)
missing_cols = set(X_train.columns ) - set(X_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X_train.columns]
# Make predictions using the testing set
pred = regr.predict(X_test)
print("LinearRegression:" ,regr.score(X_test,y_test.apply(np.log)))
print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test,np.exp(pred))))
# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(y_test, np.exp(pred)))
