import pandas as pd
import numpy as np

print('loading prior')
order_products_prior = pd.read_csv("../input/order_products__prior.csv", engine='c',
                       dtype={'order_id': np.uint32, 'product_id': np.uint16,
                              'add_to_cart_order': np.uint16, 'reordered': np.uint8})
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

print('User x product features')
# User x Product features
UP_features = [
    'UP_orders', # how many times a given product was ordered per user
    'UP_reorders',
    'UP_reorder_rate',  # how often a given user reorders a given product
    'UP_last_order_id',
    'UP_sum_pos_in_cart',
    'UP_days_since_prior_order'
]
def build_user_x_product_features(priors_x_orders):
    """ Builds a DataFrame with user specific product features

    Parameters
    ----------
    priors_x_orders pd.DataFrame
        Needs to have at least 'product_id' and 'reordered' in the columns

    Returns
    -------
    up_f pd.DataFrame
         A DF containing user_x_product features
    """
    up_f = priors_x_orders.copy()
    up_f['user_x_product'] = (priors_x_orders['product_id'] + priors_x_orders['user_id'].astype(np.int64) * 100000).values
    up_f.set_index('user_x_product', drop=False, inplace=True)  #?????
    up_f['UP_orders'] = (up_f.groupby('user_x_product')['order_id'].size()).astype(np.uint16)
    up_f['UP_reorders'] = (up_f.groupby('user_x_product')['reordered'].sum()).astype(np.uint16)
    up_f['UP_reorder_rate'] = up_f.UP_reorders / up_f.UP_orders
    up_f['UP_sum_pos_in_cart'] = (up_f.groupby('user_x_product')['add_to_cart_order'].sum()).astype(np.int16)
    up_f['UP_last_order_id'] = (up_f.groupby('user_x_product')['order_id'].agg('last')).astype(np.int32)
    up_f.rename(columns={'days_since_prior_order': 'UP_days_since_prior_order'}, inplace=True)
    up_f = up_f[['UP_orders', 'UP_reorders', 'UP_reorder_rate', 'UP_sum_pos_in_cart',
                'UP_last_order_id', 'UP_days_since_prior_order']]
    up_f = up_f[~up_f.index.duplicated(keep='first')]
    return up_f

up_f = build_user_x_product_features(priors_x_orders)
up_f.to_pickle('user_x_product_features.p')
