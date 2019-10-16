import numpy as np
import pandas
import math
import collections,csv
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def drop_irrelevant_columns(dataframe):
    del dataframe['Hair Color']
    del dataframe['Size of City']
    del dataframe['Instance']
    return dataframe

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
df = keep_known_columns(df,['Year of Record','Gender','Age','Country','Profession','University Degree','Body Height [cm]'])
df = drop_irrelevant_columns(df)
print(df[df.isna().any(axis=1)])

# Group data together
df['Gender'].replace('0','male', inplace=True)
df['Gender'].replace('unknown','other', inplace=True)
df = group(df,["Profession","Country","Gender","University Degree"],4)

# Transform categorical data
df = pandas.get_dummies(data=df, drop_first=True)

# Necessary to take only positive values for using logs. Otherwise, we get an error.
df = df[df['Income in EUR'] > 100]

# Remove outliers
df = df[df['Income in EUR'] <= 4000000]

# Apply log to get normal distribution
df['Income in EUR'] = df['Income in EUR'].apply(np.log)

y = df['Income in EUR']  # Labels
X = df.drop(['Income in EUR'], axis=1) # Features

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) # 70% training and 30% test

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, y)

# Train the model using the training sets
#regr.fit(X_train, y_train)
#pred = regr.predict(X_test)
#print("Mean squared error: %.2f" % np.sqrt(mean_squared_error(np.exp(y_test),np.exp(pred))))

res = pandas.read_csv('finaldata.csv')
instance = res['Instance']
res['Year of Record'].fillna(res['Year of Record'].median(),inplace=True)
res['Age'].fillna(res['Age'].median(),inplace=True)
res['Body Height [cm]'].fillna(res['Body Height [cm]'].median(),inplace=True)
res.fillna(method='ffill',inplace=True)

res = drop_irrelevant_columns(res)
res['Gender'].replace('0','male', inplace=True)
res['Gender'].replace('unknown','other', inplace=True)
res = group(res,["Profession","Country","Gender","University Degree"],4)
res = pandas.get_dummies(data=res, drop_first=True)

X_test = res.drop(['Income'], axis=1)
# Get missing columns in the training test
missing_cols = set(X.columns ) - set(X_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
X_test = X_test[X.columns]

# Make predictions using the testing set
pred = regr.predict(X_test)

# Write to csv file
a=pandas.DataFrame.from_dict({
    'Instance' : instance,
    'Income': np.exp(pred),
})

a.to_csv(r'tonight.csv',index=False)
