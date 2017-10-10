import numpy as np
import pandas as pd
import gc
import pickle

# Read in pregenerated user, product and user x product features and create
# features train, labels_train and features_test that will be input into
# the algorithm

print('loading train')
order_products_train = pd.read_csv("input/order_products__train.csv", engine='c',
                       dtype={'order_id': np.uint32, 'product_id': np.uint16,
                              'add_to_cart_order': np.uint16, 'reordered': np.uint8})
print('loading orders')
orders = pd.read_csv("input/orders.csv",
    dtype={'order_id': np.uint32,
           'user_id': np.uint32,
           'order_number': np.uint32,
           'order_dow': np.uint8,
           'order_hour_of_day': np.uint8,
           'days_since_prior_order': np.float16})
orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2}).astype(np.uint8)

prod_f = pd.read_pickle("features/product_features.p")
user_f = pd.read_pickle("features/user_features.p")
up_f = pd.read_pickle("features/user_x_product_features.p")

print('test/train split')
orders.set_index('order_id', inplace=True, drop=False)
train_orders = orders[orders.eval_set == 1]
test_orders = orders[orders.eval_set == 2]

def build_features_df(
        orders_df, user_f, return_labels=False,
        train_orders_info=None):
    """ Creates a DataFrame where features will be stored.
    The columns of the DF are 'order_id', 'user_id', 'product_id'
    (a product ordered by the user in the past), 'user_x_product'
    (unique key for every user_id and product_id combination)

    Parameters
    ----------
    orders_df pd.DataFrame
        Df containing information about orders. Has to contain 'order_id'
        and 'reordered' columns.

    user_f pd.DataFrame
        Df containing information about users. Has to be indexed by 'user_id'
        and contain 'U_all_products' column (a set of all products ordered by
        the user historically)

    return_labels Boolean, default = False
        Indicated whether to return labels - i.e. if orders_df contains training
        set orders.

    train_orders_info default None
        Contains 'reordered' labels for each order x product combination.
        Relevant for building the list of labels for the training examples.


    Returns:
    --------
    res pd.DataFrame
        Df where the dictionary of all user products was expanded

        order_id  user_id  product_id  user_x_product
    0  2334745    1        1           100001
    1  2334745    1        2           100002
    2  2334745    1        25          100025
    3  3445458    1        45          100045
    4  3445458    1        9           100009
    5  14454542   2        1           200001

    labels list
        The list of labels (1 if product reordered, 0 otherwise) corresponding
        to (order_id, product_id) pairs in res. len(labels) = number of rows
        of res.
    """
    order_ids = []
    product_ids = []
    user_ids = []
    labels = []
    if train_orders_info is not None:
        train_orders_info.set_index(['order_id', 'product_id'],
            inplace=True)
        train_index_lookup = dict().fromkeys(train_orders_info.index.values)

    for order_id in orders_df.index:
        user_id = orders_df.loc[order_id]['user_id'].astype(np.uint64)
        for product_id in user_f.loc[user_id]['U_all_products']:
            order_ids.append(order_id)
            product_ids.append(product_id)
            user_ids.append(user_id)
            if return_labels:
                labels += [(order_id, product_id) in train_index_lookup]

    user_x_product = (product_ids + np.multiply(user_ids, 100000)).astype(np.int64)
    d = {'order_id' : order_ids, 'user_id' : user_ids, 'product_id' : product_ids,
        'user_x_product': user_x_product}
    return pd.DataFrame(d), labels

def add_features(data_df, f_name, f_df):
    """ Adds features to the DataFrame with data

    Parameters
    ----------
    data_df  pd.DataFrame
        Has to contain f_name in the columns

    f_name  str
        Feature name e.g. 'user_id'

    f_df  pd.DataFrame
        Contains features that will be added to data_df, through mapping
        by f_name. Has to be indexed by f_name.

    Returns
    -------
    data_df pd.DataFrame
        Input DF with appended features from f_df
    """

    for f in f_df.columns:
        data_df[f] = getattr(data_df, f_name).map(f_df[f])
    return data_df

print('creating features train')
features_train, labels_train = build_features_df(
    train_orders, user_f, return_labels=True,
    train_orders_info=order_products_train
)

print('creating features test')
features_test, _ = build_features_df(test_orders, user_f)

for features_df in [features_train, features_test]:
    print('adding user features')
    add_features(features_df, 'user_id', user_f.drop(['U_all_products'], axis=1))
    print('adding product features')
    add_features(features_df, 'product_id', prod_f)
    print('adding user x product features')
    add_features(features_df, 'user_x_product', up_f)
    print('adding order features')
    add_features(
        features_df, 'order_id', orders.drop(['order_id', 'user_id', 'eval_set'],
        axis=1)
    )

features_train.to_pickle('features_train.p')
features_test.to_pickle('features_test.p')
with open('labels_train.p', 'wb') as f:
    # labels_train is a list 
    pickle.dump(labels_train, f)
