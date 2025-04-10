#Імпорт необхідних бібліотек
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#Шлях до вхідного файлу з даними
input_file = r"C:\Users\Aleksandr\Downloads\Новая папка\income_data.txt"
#Ініціалізація структур даних
X = []  #Масив ознак
y = []  #Масив міток класів
count_class1 = 0  #Лічильник класу <=50K
count_class2 = 0  #Лічильник класу >50K
max_datapoints = 25000  #Максимальна кількість записів
#Завантаження та обробка даних
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1
#Конвертація у numpy масив
X = np.array(X)
#Підготовка до кодування категоріальних ознак
label_encoder = []
X_encoded = np.empty(X.shape)
#Процес кодування даних
for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)
#Розділення на ознаки та цільову змінну
X_features = X_encoded[:, :-1].astype(int)
y_labels = X_encoded[:, -1].astype(int)
#Розбиття на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=5)
#Ініціалізація та навчання моделі з гаусовим ядром
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)
#Прогнозування на тестових даних
y_pred = classifier.predict(X_test)
#Обчислення метрик якості
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
#Виведення результатів
print("Гаусове ядро")
print(f"Акуратність: {round(accuracy * 100, 2)}%")
print(f"Точність: {round(precision * 100, 2)}%")
print(f"Повнота: {round(recall * 100, 2)}%")
print(f"F1-міра: {round(f1 * 100, 2)}%")
#Підготовка нового прикладу для класифікації
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', 
              '0', '40', 'United-States']
#Кодування вхідних даних
input_data_encoded = np.empty(len(input_data))
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(item)
    else:
        input_data_encoded[i] = label_encoder[count].transform([item])[0]
        count += 1
#Прогнозування та виведення результату
predicted_class = classifier.predict(input_data_encoded.reshape(1, -1))
predicted_label = label_encoder[-1].inverse_transform(predicted_class)[0]
print(f"Прогноз для вхідної точки: {predicted_label}")