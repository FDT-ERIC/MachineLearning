import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# =====================================================================================================================

# train data path
TRAIN_DATA_PATH = '../Data/full_sorted_int.csv'
# test data path
TEST_DATA_PATH = '../Data/test_without_qm__int.csv'

# =====================================================================================================================

# define GridSearchCV parameters
cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0,
              'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'binary:logistic'}

# define model
model = GridSearchCV(XGBClassifier(**ind_params),
                      cv_params,
                      scoring='accuracy',  # 准确度评价标准
                      cv=5,                # cross_validation，交叉验证
                      n_jobs=-1)           # 并行数，int：个数,-1：跟CPU核数一致, 1:默认值

# =====================================================================================================================

def XGB_TRAIN_EVA():

    # prepare train data
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    train_data.pop('index')
    Y_train = train_data.pop('income')
    X_train = train_data

    # prepare test data
    test_data = pd.read_csv(TEST_DATA_PATH)
    test_data.pop('index')
    Y_test = test_data.pop('income')
    X_test = test_data

    # model train
    model.fit(X_train, Y_train)
    model_best = model.best_estimator_
    Y_pred = model_best.predict(X_test)

    # print report
    print(classification_report(Y_test, Y_pred))

# =====================================================================================================================

XGB_TRAIN_EVA()