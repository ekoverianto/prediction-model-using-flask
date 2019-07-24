#library
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

#load dataset
df = pd.read_csv('Salary.csv')

#Split
X= df.iloc[:, 0].values.reshape((-1, 1))
Y= df.iloc[:, -1].values

#call model regression
model = LinearRegression().fit(X, Y)

#save model
filename = 'model.sav'
joblib.dump(model, filename)

#load model
loaded_model = joblib.load(filename)

#prediction model
#loaded_model.predict(20)