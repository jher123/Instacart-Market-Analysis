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
orders.eval_set = orders.eval_set.replace(
    {'prior': 0, 'train': 1, 'test':2}
).astype(np.uint8)

print('loading products, aisles and depts')
products = pd.read_csv("../input/products.csv")
products = products.astype(
    {'product_id': np.uint16, 'department_id': np.uint8, 'aisle_id': np.uint8},
    inplace=True
)
aisles = pd.read_csv("../input/aisles.csv")
depts = pd.read_csv("../input/departments.csv")

print('preprocessing')
priors_x_orders = pd.merge(orders, order_products_prior, on='order_id', how='right')
priors_x_orders.set_index('order_id', drop=False, inplace=True)

pr = pd.merge(products, depts, how='left', on='department_id')
pr = pd.merge(pr, aisles, how='left', on='aisle_id')
pr.set_index('product_id', inplace=True, drop=False)

del aisles
del products
del depts
del order_products_prior

priors_x_orders = priors_x_orders.merge(pr, on='product_id', how='left')

print('Product features')
# product features
product_features = [
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

def product_type(series, name):
    """ Checks product names in series to see whether they contain name and
    returns 1 if they do and 0 otherwise

    Parameters
    ---------
    series pd.Series
        Contains product names

    name str
        The str that we are checking for presence in each name

    Returns
    -------
    out list
        A list of Booleans (0 or 1)
    """
    out = []
    for product in series:
        out.append((name in product.split()) * 1)
    return out

def add_product_features(pr_df, features_list):
    """ Adds name related product features from the features_list
    using product_type

    Parameters
    ----------
    pr_df pd.DataFrame
        Indexed by product_id and contains 'product_name' for each product

    features_list list of str
        A list of strings that will be looked up in product names to check if
        product names contain them e.g. 'Vegan'

    Returns
    -------
    pr_df pd.DataFrame
        The products features DF with added name features.
    """
    for f in features_list:
        pr_df[f] = product_type(pr_df['product_name'], f)
    return pr_df

def build_product_features(pr, priors_x_orders):
    """ Builds a DataFrame with product features

    Parameters
    ----------
    pr pd.DataFrame
        Needs to have product_ids in the index and 'department_id' and
        'aisle_id' in columns

    priors_x_orders pd.DataFrame
        Needs to have at least 'product_id' and 'reordered' in the columns

    Returns
    -------
    prod_f pd.DataFrame
         A DF containing product features
    """
    prod_f = pd.DataFrame()
    prod_f['P_total_orders'] = priors_x_orders.groupby('product_id').size().astype(np.int32)
    prod_f['P_reorders'] = priors_x_orders.groupby('product_id')['reordered'].sum().astype(np.int32)
    prod_f['P_reorder_rate'] = prod_f['P_reorders'] / prod_f['P_total_orders']
    prod_f = prod_f.reset_index().set_index('product_id', drop=False)
    prod_f['P_vegan'] = prod_f.product_id.map(pr['Vegan'])
    prod_f['P_organic'] = prod_f.product_id.map(pr['Organic'])
    prod_f['P_gluten_free'] = prod_f.product_id.map(pr['Gluten'])
    prod_f['P_vegetarian'] = prod_f.product_id.map(pr['Vegetarian'])
    prod_f['P_lowfat'] = prod_f.product_id.map(pr['Lowfat'])
    prod_f['P_lite'] = prod_f.product_id.map(pr['Lite'])
    prod_f['P_department_id'] = prod_f.product_id.map(pr['department_id'])
    prod_f['P_aisle_id'] = prod_f.product_id.map(pr['aisle_id'])
    return prod_f


features_pr = ['Vegan', 'Vegetarian', 'Organic', 'Gluten', 'Lowfat', 'Lite']
add_product_features(pr, features_pr)
prod_f = build_product_features(pr, priors_x_orders)
prod_f.to_pickle('product_features.p')
