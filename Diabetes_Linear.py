from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes(return_X_y=False)
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.03, random_state=1)
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
print("Accuracy: {}".format(reg.score(x_test, y_test)))
