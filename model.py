import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import pickle

# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('insurance.csv')

# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

# encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

#splitting freature and target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# loading the Linear Regression model
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

# prediction on training data
training_data_prediction =regressor.predict(X_train)

# prediction on test data
test_data_prediction =regressor.predict(X_test)

# saving the model into the disk
pickle.dump(regressor, open('model.pkl' , 'wb'))

# loading the model to disk
model = pickle.load(open('model.pkl','rb'))


# input_data = (31,1,25,0,1,0)

# # changing input_data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)

# # reshape the array
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# prediction = regressor.predict(input_data_reshaped)
# print(prediction)

# print('The insurance cost is USD ', prediction[0])