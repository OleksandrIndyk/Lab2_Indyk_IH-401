import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
input_file = "C:\\AI\\data_multivar_regr.txt" #Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
num_training = int(0.8 * len(X)) #Розбиття на тренувальні та тестові набори
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
linear_regressor = linear_model.LinearRegression() #Створення лінійного регресора і навчання
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test) #Прогнозування результатів
print("Linear Regressor performance:") #Метрики якості лінійної регресії
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
polynomial = PolynomialFeatures(degree=10) #Поліноміальна регресія 10 ступеня
X_train_transformed = polynomial.fit_transform(X_train)
poly_linear_model = linear_model.LinearRegression() #Створення і навчання поліноміального регресора
poly_linear_model.fit(X_train_transformed, y_train)
datapoint = [[7.75, 6.35, 5.56]] #Точка для передбачення
poly_datapoint = polynomial.transform(datapoint)
linear_pred = linear_regressor.predict(datapoint) #Прогнозування для точки даних
poly_pred = poly_linear_model.predict(poly_datapoint)
print("\nLinear regression prediction for datapoint {}: {:.2f}".format(datapoint[0], linear_pred[0]))
print("Polynomial regression prediction for datapoint {}: {:.2f}".format(datapoint[0], poly_pred[0]))
