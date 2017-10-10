import pandas as pd
import numpy as np

print('loading prior')
order_products_prior = pd.read_csv("../input/order_products__prior.csv", engine='c',
                       dtype={'order_id': np.uint32, 'product_id': np.uint16,
                              'add_to_cart_order': np.uint16, 'reordered': np.uint8})
print('loading products')
products = pd.read_csv("../input/products.csv")
products = products.astype({'product_id': np.uint16, 'department_id': np.uint8, 'aisle_id': np.uint8}, inplace=True)

print('loading orders')
orders = pd.read_csv("../input/orders.csv",
    dtype={'order_id': np.uint32,
           'user_id': np.uint32,
           'order_number': np.uint32,
           'order_dow': np.uint8,
           'order_hour_of_day': np.uint8,
           'days_since_prior_order': np.float16})
orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2}).astype(np.uint8)

print('preprocessing')
priors_x_orders = pd.merge(orders, order_products_prior, on='order_id', how='right')
priors_x_orders.set_index('order_id', drop=False, inplace=True)
priors_x_orders = priors_x_orders.merge(products, on='product_id', how='left')

print('User features')
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

def build_user_features(orders, priors_x_orders):
    """ Builds a DataFrame with user features

    Parameters
    ----------
    orders pd.DataFrame
        Contains information about orders. Needs to have 'user_id' in columns.

    priors_x_orders pd.DataFrame
        Needs to have at least 'product_id' and 'user_id' in the columns

    Returns
    -------
    user_f pd.DataFrame
         A DF containing user features
    """
    user_f = pd.DataFrame()
    user_f['U_total_orders'] = orders.groupby('user_id').size().astype(np.int8)  # bcs up to 100 prior orders
    user_f['U_total_products'] = priors_x_orders.groupby('user_id')['product_id'].size().astype(np.uint16)
    user_f['U_all_products'] = priors_x_orders.groupby('user_id')['product_id'].apply(set) # set in python has unique elements!
    user_f['U_total_distinct_products'] = user_f['U_all_products'].map(len).astype(np.uint16)
    user_f['U_avg_basket_size'] = (user_f['U_total_products'] / user_f['U_total_orders']).astype(np.float32)
    user_f['U_avg_days_bn_orders'] = priors_x_orders.groupby('user_id')['days_since_prior_order'].apply(np.mean).astype(np.float32)
    users_pet = priors_x_orders[priors_x_orders.department_id==8].user_id.unique()
    user_f['U_has_a_pet'] = user_f.index.isin(users_pet)
    users_baby = priors_x_orders[priors_x_orders.department_id==18].user_id.unique()
    user_f['U_has_a_baby'] = user_f.index.isin(users_baby)
    return user_f

user_f = build_user_features(orders, priors_x_orders)
user_f.to_pickle('user_features.p')
