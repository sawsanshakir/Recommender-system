#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Reference :https://surprise.readthedocs.io/en/latest/getting_started.html#tuning-algorithm-parameters

## tuning th enumber factor of the svd function for Book-Crosssing




import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import recmetrics
from sklearn.model_selection import train_test_split
from surprise import SVD
from surprise import GridSearch
#from sklearn.metrics import  mean_squared_error, make_scorer
import sklearn 
import time
from surprise import Reader
from surprise import Dataset

#from sklearn.metrics import ml_metrics as metrics

book_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Books.csv',  names= ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM', 'imageURLL'] ,delimiter=";", encoding='latin-1')
rating_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Book-Ratings.csv',  names= ['UserID','ISBN','Rating'],delimiter=";", encoding='latin-1')
user_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Users.csv',  names= ['UserID','Gender','Age','Occupation','Zip-code'],delimiter=";", encoding='latin-1')

ratings_new = rating_data[rating_data.ISBN.isin(book_data.ISBN)]

ratings = rating_data[rating_data.UserID.isin(user_data.UserID)]


ratings_explicit = ratings_new[ratings_new.Rating != 0]
ratings_implicit = ratings_new[ratings_new.Rating == 0]


#print (ratings_new.shape)
print (ratings_explicit.shape)
#print (ratings_implicit.shape)
counts = ratings_explicit['ISBN'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['ISBN'].isin(counts[counts >= 10].index)]


count1 = ratings_explicit['UserID'].value_counts()

ratings_explicit = ratings_explicit[ratings_explicit['UserID'].isin(count1[count1 >= 20].index)]
#print(ratings_explicit['UserID'].value_counts())
print(ratings_explicit.shape)
#
#### split the ratings table into taining and testing dataset

ratings_train, ratings_test = train_test_split(ratings_explicit, stratify=ratings_explicit['UserID'],test_size=0.30, random_state=0)
#

#
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings_train[['UserID','ISBN','Rating']], reader)

parameter_grid = {'n_factors':[50,100,150,200,250,300] }



grid_search = GridSearch( SVD,  parameter_grid,  measures =['RMSE', 'MAE'] )


grid_search.evaluate(data)

best_parameters = grid_search.best_params 
print(best_parameters)

# best RMSE and MAE score
best_result = grid_search.best_score
print(best_result)  


# In[ ]:




