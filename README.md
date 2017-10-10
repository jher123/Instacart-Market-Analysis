# Instacart-Market-Analysis

The project is based on the Kaggle competiton https://www.kaggle.com/c/instacart-market-basket-analysis
The goal is to predict which products will be reordered by a customer of an online supermarket given the history 
of their previous orders. I generated a lot of features derived from information about the users' shopping behaviour as well as the products
and used the XGBoost algorithm to make predictions.

*How to run:*

0. Copy the data files into input folder from the Kaggle website.
1. Generate pickle files with user, product and user/product features by running scripts in feature
2. Run create_features.py to merge and process the above features and create data for training and testing
3. Run run_algo.py which trains an XGBoost model and creates predictions on the test set

*Requirements:*

python==3.6.1

numpy=='1.13.3'

pandas=='0.20.3'

xgboost==0.6
