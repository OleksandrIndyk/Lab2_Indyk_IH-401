#Імпорт набору даних Iris
from sklearn.datasets import load_iris
iris_dataset = load_iris()
#Виведення ключів словника даних
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
#Виведення опису набору даних
print(iris_dataset['DESCR'][:193] + "\n...")
#Виведення назв цільових класів
print("Назви відповідей:{}".format(iris_dataset['target_names']))
#Виведення назв ознак
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
#Аналіз типу та форми даних
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))
#Виведення перших п'яти прикладів
print("Перші 5 прикладів:", iris_dataset['data'][:5])
#Виведення міток цільових класів
print("Відповіді:\n{}".format(iris_dataset['target']))