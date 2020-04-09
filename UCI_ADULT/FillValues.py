import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# col labels
col_attributes = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                  'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                  'hours_per_week', 'native_country', 'income']

# define GridSearchCV parameters
cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed': 0,
              'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'binary:logistic'}

# define model
model = GridSearchCV(XGBClassifier(**ind_params),
                      cv_params,
                      scoring='accuracy',
                      cv=5,
                      n_jobs=-1)

# =====================================================================================================================

# load data
def get_data():

    # data path
    train_path = './Data/adult.data.csv'
    test_path = './Data/adult.test.csv'

    # data
    train_set = pd.read_csv(train_path, header=None)
    test_set = pd.read_csv(test_path, header=None)

    # labels
    train_set.columns = col_attributes
    test_set.columns = col_attributes

    # replace '<=50K.' by '<=50K', replace '>50K.' by '>50K' in test set
    replace_dict = {' <=50K.': ' <=50K', ' >50K.': ' >50K'}
    test_set['income'] = test_set['income'].replace(replace_dict)

    return train_set, test_set

# =====================================================================================================================

# prepare data
train_set, test_set = get_data()

# =====================================================================================================================

# change object into int

def obj2int(data):
    attr_to_dic = {}
    for attr in col_attributes:
        if data[attr].dtype == 'object':
            word_to_idx = {}
            i = 1
            for item in list(data[attr].unique()):
                data.loc[data[attr] == item, attr] = i
                word_to_idx[item] = i
                i = i + 1
            attr_to_dic[attr] = word_to_idx
            # change type
            data[attr] = data[attr].astype('int64')
    return attr_to_dic

# for training data
attr_to_dic = obj2int(train_set)

# for testing data
for attr in attr_to_dic:
    dic = attr_to_dic[attr]
    for item in dic:
        test_set.loc[test_set[attr] == item, attr] = dic[item]
    # change type
    test_set[attr] = test_set[attr].astype('int64')

# =====================================================================================================================

# function for dealing with data (without question mask)
def dealQM(data):

    wor_idx = list(data['workclass'] != attr_to_dic['workclass'][' ?'])
    occ_idx = list(data['occupation'] != attr_to_dic['occupation'][' ?'])
    nat_idx = list(data['native_country'] != attr_to_dic['native_country'][' ?'])

    # define lists
    all_idx = []
    only_occ_idx, only_nat_idx = [], []
    wor_occ_idx = []
    opp_all_idx = []

    # create True and False index
    for i in range(len(data)):

        # for all without question mask
        if wor_idx[i] and occ_idx[i] and nat_idx[i]:
            all_idx.append(True)
        else:
            all_idx.append(False)

        # for all with question mask
        if (not wor_idx[i]) and (not occ_idx[i]) and (not nat_idx[i]):
            opp_all_idx.append(True)
        else:
            opp_all_idx.append(False)


        # only for occupation with question mask
        if wor_idx[i] and (not occ_idx[i]) and nat_idx[i]:
            only_occ_idx.append(True)
        else:
            only_occ_idx.append(False)
        # only for native_country with question mask
        if wor_idx[i] and occ_idx[i] and (not nat_idx[i]):
            only_nat_idx.append(True)
        else:
            only_nat_idx.append(False)

        # for both workclass and occupation with question mask
        if (not wor_idx[i]) and (not occ_idx[i]) and nat_idx[i]:
            wor_occ_idx.append(True)
        else:
            wor_occ_idx.append(False)


    # create data based on the indexes
    data_without = data[all_idx]        # all without question mask
    data_with = data[opp_all_idx]       # all with question mask
    data_only_occ = data[only_occ_idx]  # only occupation with question mask
    data_only_nat = data[only_nat_idx]  # only native_country with question mask
    data_wor_occ = data[wor_occ_idx]    # both workclass and occupation with question mask

    return data_without, data_with, data_only_occ, data_only_nat, data_wor_occ

# =====================================================================================================================

# training set
train_all_without_qm, train_all_with_qm, train_only_occ, train_only_nat, train_wor_occ = dealQM(train_set)

# testing set
test_all_without_qm, test_all_with_qm, test_only_occ, test_only_nat, test_wor_occ = dealQM(test_set)

# =====================================================================================================================

""" XGBoost for occupation only """

def XGB_OCC():

    # training data without qm
    Y_train_without_qm_copy = train_all_without_qm.copy()
    Y_train_without_qm = Y_train_without_qm_copy.pop('occupation')
    X_train_without_qm = Y_train_without_qm_copy

    # training data occ
    train_only_occ_copy = train_only_occ.copy()
    train_only_occ_copy.pop('occupation')
    X_train_occ = train_only_occ_copy

    # model train
    model.fit(X_train_without_qm, Y_train_without_qm)
    model_best = model.best_estimator_
    Y_pred_occ = model_best.predict(X_train_occ)

    # set prediction value into the data
    train_only_occ['occupation'] = Y_pred_occ

    # output csv file
    train_only_occ.to_csv('01_train_only_occ.csv')

    return train_only_occ

# =====================================================================================================================

""" XGBoost for native_country only """

def XGB_NAT():

    # training data without qm
    Y_train_without_qm_copy = train_all_without_qm.copy()
    Y_train_without_qm = Y_train_without_qm_copy.pop('native_country')
    X_train_without_qm = Y_train_without_qm_copy

    # training data nat
    train_only_nat_copy = train_only_nat.copy()
    train_only_nat_copy.pop('native_country')
    X_train_nat = train_only_nat_copy

    # model train
    model.fit(X_train_without_qm, Y_train_without_qm)
    model_best = model.best_estimator_
    Y_pred_nat = model_best.predict(X_train_nat)

    # set prediction value into the data
    train_only_nat['native_country'] = Y_pred_nat

    # output csv file
    train_only_nat.to_csv('02_train_only_nat.csv')

    return train_only_nat

# =====================================================================================================================

""" XGBoost for wor_occ """

def XGB_WOR_OCC():

    # training data without qm
    train_all_without_qm_copy = train_all_without_qm.copy()
    Y_train_without_qm_wor = train_all_without_qm_copy.pop('workclass')
    Y_train_without_qm_occ = train_all_without_qm_copy.pop('occupation')
    X_train_without_qm = train_all_without_qm_copy

    # training data
    train_wor_occ_copy = train_wor_occ.copy()
    train_wor_occ_copy.pop('workclass')
    train_wor_occ_copy.pop('occupation')
    X_train_wor_occ = train_wor_occ_copy

    # model train workclass
    model.fit(X_train_without_qm, Y_train_without_qm_wor)
    model_best = model.best_estimator_
    Y_pred_wor = model_best.predict(X_train_wor_occ)
    train_wor_occ['workclass'] = Y_pred_wor

    # model train occupation
    model.fit(X_train_without_qm, Y_train_without_qm_occ)
    model_best = model.best_estimator_
    Y_pred_occ = model_best.predict(X_train_wor_occ)
    train_wor_occ['occupation'] = Y_pred_occ

    # output csv file
    train_wor_occ.to_csv('03_train_wor_occ.csv')

    return train_wor_occ

# =====================================================================================================================

""" XGBoost for wor_occ_nat """

def XGB_WOR_OCC_NAT():

    # training data without qm
    train_all_without_qm_copy = train_all_without_qm.copy()
    Y_train_without_qm_wor = train_all_without_qm_copy.pop('workclass')
    Y_train_without_qm_occ = train_all_without_qm_copy.pop('occupation')
    Y_train_without_qm_nat = train_all_without_qm_copy.pop('native_country')
    X_train_without_qm = train_all_without_qm_copy

    # training data
    train_all_with_qm_copy = train_all_with_qm.copy()
    train_all_with_qm_copy.pop('workclass')
    train_all_with_qm_copy.pop('occupation')
    train_all_with_qm_copy.pop('native_country')
    X_train_wor_occ_nat = train_all_with_qm_copy

    # model train workclass
    model.fit(X_train_without_qm, Y_train_without_qm_wor)
    model_best = model.best_estimator_
    Y_pred_wor = model_best.predict(X_train_wor_occ_nat)
    train_all_with_qm['workclass'] = Y_pred_wor

    # model train occupation
    model.fit(X_train_without_qm, Y_train_without_qm_occ)
    model_best = model.best_estimator_
    Y_pred_occ = model_best.predict(X_train_wor_occ_nat)
    train_all_with_qm['occupation'] = Y_pred_occ

    # model train native_country
    model.fit(X_train_without_qm, Y_train_without_qm_nat)
    model_best = model.best_estimator_
    Y_pred_nat = model_best.predict(X_train_wor_occ_nat)
    train_all_with_qm['native_country'] = Y_pred_nat

    # output csv file
    train_all_with_qm.to_csv('04_train_all_with_qm.csv')

    return train_all_with_qm

# =====================================================================================================================

print('\n\n================================================== XGBoost ==================================================')

print('\n\n================================================== XGB_OCC ==================================================')
train_predict_occ = XGB_OCC()                  # occupation

print('\n\n================================================== XGB_NAT ==================================================')
train_predict_nat = XGB_NAT()                  # native_country

print('\n\n================================================ XGB_WOR_OCC ================================================')
train_predict_wor_occ = XGB_WOR_OCC()          # wor and occ

print('\n\n============================================== XGB_WOR_OCC_NAT ==============================================')
train_predict_wor_occ_nat = XGB_WOR_OCC_NAT()  # wor and occ and nat

# concat 'train_all_without_qm' and 'train_only_occ' and 'train_only_nat'
full_combine = pd.concat([train_all_without_qm, train_predict_occ, train_predict_nat,
                          train_predict_wor_occ, train_predict_wor_occ_nat], axis=0)

# output as a csv file
full_combine.to_csv('full.csv')