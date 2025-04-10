#Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
#Завантаження та підготовка даних
try:
    data = pd.read_csv(r"C:\Users\Aleksandr\Downloads\Новая папка\income_data.txt", delimiter=",", header=None)
except FileNotFoundError:
    print("Помилка: файл не знайдено. Перевірте шлях та назву файлу.")
    exit()
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
#Обробка пропущених значень
data.replace(' ?', np.nan, inplace=True)
data.dropna(inplace=True)
#Кодування категоріальних змінних
data = pd.get_dummies(data)
#Визначення ознак та цільової змінної
X = data.drop('income_ >50K', axis=1, errors='ignore')
y = data['income_ >50K'] if 'income_ >50K' in data.columns else data['income']
#Розділення на тренувальний та тестовий набори
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)
#Ініціалізація моделей для порівняння
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))  # Было лишняя скобка
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('LinearSVC', LinearSVC(max_iter=10000)))
#Оцінка моделей методом крос-валідації
results = []
names = []
print("Оцінка моделей (крос-валідація):\n")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")
#Візуалізація результатів
plt.boxplot(results, tick_labels=names)
plt.title('Порівняння алгоритмів класифікації')
plt.ylabel('Точність')
plt.show()
#Навчання та оцінка найкращої моделі
best_model = DecisionTreeClassifier()
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_validation)
#Виведення метрик якості
print("\nТочність на тестовій вибірці:", accuracy_score(y_validation, predictions))
print("\nМатриця помилок:")
print(confusion_matrix(y_validation, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(y_validation, predictions))