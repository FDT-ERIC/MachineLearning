import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# col labels
col_attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


# load data
def load_data():
    # data path
    DATA_PATH = '../Data/iris.data.csv'
    # data
    data = pd.read_csv(DATA_PATH, header=None)
    # add attributes
    data.columns = col_attributes
    return data

# split data into train and test dataset
def split_data(data):
    y = data.pop('class')
    X = data
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# generate data
DATA = load_data()
X_train, X_test, y_train, y_test = split_data(DATA)

# define model
lr_model = LogisticRegression()

# train
lr_model.fit(X_train, y_train)

# predict
y_pred = lr_model.predict(X_test)

# report
print(classification_report(y_test, y_pred))

# accuracy, normalize=False 返回正确预测的样本数量；默认为True, 返回准确率
print(accuracy_score(y_test, y_pred, normalize=False))