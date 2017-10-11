import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

print('Reading in input data')
features_train = pd.read_pickle('features_train.p')
labels_train = pickle.load(open('labels_train.p', 'rb'))
features_test = pd.read_pickle('features_test.p')

print('Training stage')
""" product_features = [
    'P_total_orders', # how many times the product was ordered
    'P_reorders',  # how many times the product was reordered
    'P_reorder_rate',
    'P_avg_position_in_basket'
    'P_avg_count_in_basket',
    'P_organic',
    'P_gluten_free',
    'P_vegan',
    'P_vegetarian',
    'P_lite',
    'P_lowfat',
    'P_aisle_id',
    'P_department_id'
]
user_features = [
    # basket related
    'U_total_orders',
    'U_total_products',
    'U_all_products',  # list of all products a user has ever ordered
    'U_total_distinct_products',
    'U_avg_basket_size',  # total products / no of orders
    'U_avg_days_bn_orders',
    'U_max_days_bn_orders',
    'U_min_days_bn_orders',
    # customer segementation
    'U_has_a_baby',
    'U_has_a_pet',
]
UP_features = [
    'UP_orders', # how many times a given product was ordered per user
    'UP_reorders',
    'UP_reorder_rate',  # how often a given user reorders a given product
    'UP_last_order_id',
    'UP_sum_pos_in_cart',
    'UP_days_since_prior_order'
]
"""
# features_selection = []

params = {
     'max_depth':10,
     'eta':0.02,
     'colsample_bytree':0.4,
     'subsample':0.75,
     'silent':1,
     'eval_metric':'logloss',
     'objective':'binary:logistic'
     }

num_rounds = 10
# dtrain = xgb.DMatrix(features_train[features_selection], label=labels_train)
dtrain = xgb.DMatrix(features_train, label=labels_train)
bst = xgb.train(params, dtrain, num_rounds)


print('Prediction on test set')
# dtest = xgb.DMatrix(features_test[features_selection])
dtest = xgb.DMatrix(features_test)
ypred = bst.predict(dtest)
features_test['pred'] = ypred

THRESHOLD = 0.22

def predicted_products(features_df, orders_df):
    d = dict()
    for row in features_df.itertuples():
        if row.pred > THRESHOLD:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in orders_df.index:
        if order not in d:
            d[order] = 'None'

    labels_pred = pd.DataFrame.from_dict(d, orient='index')
    labels_pred.reset_index(inplace=True)
    labels_pred.columns = ['order_id', 'products']
    return labels_pred

print('Generating predictions on test set')
labels_pred_test = predicted_products(features_test, test_orders)
labels_pred_test.to_csv('sub.csv', index=False)

print('calculating F1 score')
# processing for local validation
tmp = features_train.copy()
tmp['labels'] = labels_train
d = dict()
for row in (tmp[tmp['labels'] == True ]).itertuples():
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)
for order in train_orders.index:
    if order not in d:
        d[order] = 'None'
labels_true_train = pd.DataFrame.from_dict(d, orient='index')
labels_true_train.reset_index(inplace=True)
labels_true_train.columns = ['order_id', 'products']

ypred_train = bst.predict(dtrain)
features_train['pred'] = ypred_train
labels_pred_train = predicted_products(features_train, train_orders)

def f1_score(y_true_df, y_pred_df):
    f1 = []
    for y_true, y_pred in zip(y_true_df.itertuples(), y_pred_df.itertuples()):
        if y_true.products == 'None':
            y_true = set([0])
        else:
            y_true = set([int(p) for p in y_true.products.split()])

        if y_pred.products == 'None':
            y_pred = set([0])
        else:
            y_pred = set([int(p) for p in y_pred.products.split()])

        intersection =  len(y_true & y_pred)
        if intersection == 0:
            f1.append(0)
        else:
            precision = 1. * intersection / len(y_pred)
            recall = 1. * intersection  / len(y_true)
            f1.append(2. * precision * recall / (precision + recall))
    mean_f1 = np.mean(f1)
    return mean_f1

score = f1_score(labels_true_train, labels_pred_train)
print('F1 score', score)

print('Feature importances')
xgb.plot_importance(bst)
