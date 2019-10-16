# CSU44061 Machine-Learning Kaggle Competition
Given a vast dataset, predict annual income <br/>
**Important** Source for the file submitted: linear_regr.py <br/>
**Important** CSV for best public score: bestbest_submission.csv </br>
Rest of the files used for trying out new things.

## Approach
+ Only keep rows that do not contain NaN
+ Drop columns that were not contributing relevant information or were messing up the final value
+ Group together non-frequent terms
+ Use get_dummies(drop_first=True) to do something similar to hot-encoding.
+ Remove outliers and all negative income in order to apply the log to the Income
+ Try different regression models, best performer: LinearRegression
+ Apply the median value to nan values in the numerical columns found in the testing data, otherwise use fillna(method='ffill')
+ Make sure both training and testing data have same number of columns after hot encoding
