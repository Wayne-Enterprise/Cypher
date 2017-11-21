# http://www.geeksforgeeks.org/linear-regression-python-implementation/

import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

boston = datasets.load_boston(return_X_y=False)
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.28, random_state=1)
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
print("Accuracy: {}".format(reg.score(x_test, y_test)))

