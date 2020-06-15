#!/usr/bin/env python
# coding: utf-8

# In[2]:



## Reference :https://surprise.readthedocs.io/en/latest/getting_started.html#tuning-algorithm-parameters

## tuning th enumber factor of the svd function for movielens-1m


import pandas as pd
import numpy as np
import recmetrics
from sklearn.model_selection import train_test_split
from surprise import SVD
from surprise import GridSearch
#from sklearn.metrics import  mean_squared_error, make_scorer
import sklearn 
import time
from surprise import Reader
from surprise import Dataset

movie_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\movies.txt',  names= ['MovieID','Title','Genres'] , encoding='latin-1')
rating_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\ratings.txt',  names= ['UserID','MovieID','Rating','Timestamp'])
user_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\users.txt',  names= ['UserID','Gender','Age','Occupation','Zip-code'])

#### split the ratings table into taining and testing dataset

ratings_train, ratings_test = train_test_split(rating_data, stratify=rating_data['UserID'], test_size=0.30, random_state=42)
#

#
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings_train[['UserID','MovieID','Rating']], reader)

parameter_grid = {'n_factors':[50, 100, 150, 200, 250, 300] }



grid_search = GridSearch( SVD,  parameter_grid,  measures =['RMSE', 'MAE'] )


grid_search.evaluate(data)

best_parameters = grid_search.best_params 
print(best_parameters)

# best RMSE and MAE score
best_result = grid_search.best_score
print(best_result)  


# In[ ]:




