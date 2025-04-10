#Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
#Завантаження та підготовка даних
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#Виведення форми датасету
print(dataset.shape)
#Виведення перших 20 записів
print(dataset.head(20))
#Виведення статистичних характеристик
print(dataset.describe())
#Виведення розподілу за класами
print(dataset.groupby('class').size())
#Візуалізація даних у вигляді box-plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
#Побудова гістограм розподілу атрибутів
dataset.hist()
pyplot.show()
#Матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.show()
#Поділ датасету на навчальну та контрольну вибірки
array = dataset.values
#Вибір перших 4-х стовпців
X = array[:,0:4]
#Вибір 5-го стовпця
y = array[:,4]
#Поділ X та y на навчальну та контрольну вибірки
X_train, X_validation, y_train, Y_validation = train_test_split(X, y, 
test_size=0.20, random_state=1)
#Завантаження алгоритмів моделі
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#Оцінка моделі на кожній ітерації
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#Порівняння алгоритмів
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
#Cтворюємо прогноз на контрольній вибірці
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)
#Оцінка точності
print("Точність на контрольній вибірці:", accuracy_score(Y_validation, predictions))
#Матриця помилок
print("Матриця помилок:")
print(confusion_matrix(Y_validation, predictions))
#Звіт про класифікацію
print("Звіт про класифікацію:")
print(classification_report(Y_validation, predictions))
#Отримання прогнозу
X_new = np.array([[4.7, 3.0, 1.6, 0.2]])
print('Форма масиву X_new:', X_new.shape)
predictions = model.predict(X_new)
print('Прогноз {для X_new:}', predictions)
