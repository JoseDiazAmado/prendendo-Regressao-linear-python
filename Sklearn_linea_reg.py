import pandas as pd
import pprint
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# just load the boston Dataset (from load_boston file), simple enough.
boston = load_boston()
# pprint.pprint(boston)
'''
DataFrame()(data=None, index=None, columns=None, dtype=None, copy=False)
			data: loading array (structured data)  as an input, usually dict
			index : resulting Data frame index - Not mandatory
			columns : gives Lables to the result - Not mandatory- Highly recomended
			dtype: dtype, default None - Data type to force
			copy : boolean, default False - Copy data from inputs
'''
df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

'''
df_x contains the dataset with multiple features 
df_y contains the result from the respective features from the dataset  
'''
# print(df_x)
# printing df_x will show full dataset with the total rows and columns in the end.

# print(df_x.describe())
#  printing df_x.describe() will calculate the characterstics like count, mean, std(standard deviation), max etc

model = linear_model.LinearRegression()

# Selecting the LinearRegression() model from the  linear_model file present in the sklearn library  


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4 )
# Splitting the data (i.e. df_x, df_y) into testing and training datasets with random state of 4
model.fit(x_train,y_train)
print(model.fit(x_train,y_train))
# fit means training the model with the training data

# print(reg.coef_)
a = model.predict(x_test)
print(a[4] )
print(y_test)
# predict means it will linear Regression model prediction from the x_testing data
