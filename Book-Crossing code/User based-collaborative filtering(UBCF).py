#!/usr/bin/env python
# coding: utf-8

# In[122]:


###### This program for implementation of user-based collaborative filtering (UBCF) for Book-Crossing dataset

import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from math import sqrt

### Read the ratings, movies and users tables 

book_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Books.csv',  names= ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM', 'imageURLL'] ,delimiter=";", encoding='latin-1')
rating_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Book-Ratings.csv',  names= ['UserID','ISBN','Rating'],delimiter=";", encoding='latin-1')
user_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Users.csv',  names= ['UserID','Location','Age'],delimiter=";", encoding='latin-1')


#dropping last three columns containing image URLs which will not be required for analysis
book_data.drop(['imageURLS', 'imageURLM', 'imageURLL'],axis=1,inplace=True)

#checking shapes of the datasets
print(rating_data.shape)
print(book_data.shape)
print(user_data.shape)
#


print(rating_data.Rating.unique())



# In[123]:


### remove ratings in ratings dataset which have books not exist in books dataset

ratings_new = rating_data[rating_data.ISBN.isin(book_data.ISBN)]

print (rating_data.shape)
print (ratings_new.shape)


# In[124]:


### ratings dataset should have users only which exist in users dataset
ratings = rating_data[rating_data.UserID.isin(user_data.UserID)]

print(ratings.shape)


# In[125]:


## Reference:https://github.com/csaluja/JupyterNotebooks-Medium/blob/master/Book%20Recommendation%20System.ipynb?source=post_page-----5ec959c41847----------------------

## finding implicit and explict ratings datasets

ratings_explicit = ratings_new[ratings_new.Rating != 0]
ratings_implicit = ratings_new[ratings_new.Rating == 0]


#print (ratings_new.shape)
print (ratings_explicit.shape)
#print (ratings_implicit.shape)

## considering users who have rated atleast 20 books and books which have atleast 10 ratings
counts = ratings_explicit['ISBN'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['ISBN'].isin(counts[counts >= 10].index)]


count1 = ratings_explicit['UserID'].value_counts()
print(count1)
ratings_explicit = ratings_explicit[ratings_explicit['UserID'].isin(count1[count1 >= 20].index)]
print(ratings_explicit['UserID'].value_counts())

print(ratings_explicit.shape)
#
print(len(ratings_explicit.UserID.unique()))
print(len(ratings_explicit.ISBN.unique()))


# In[126]:


### split the ratings table into taining and testing dataset

ratings_train, ratings_test = train_test_split(ratings_explicit, stratify=ratings_explicit['UserID'],test_size=0.30, random_state=0)
#

print(len(ratings_train.UserID.unique()))
print(len(ratings_train.ISBN.unique()))

print('ratings_train =')
print(ratings_train)
print()
print(len(ratings_test.UserID.unique()))
print(len(ratings_test.ISBN.unique()))


print('ratings_test =')
print(ratings_test)


# In[127]:


### create the training ratings matrix from the ratings set after hiding the testing set


# merge ratings set with the testing set using left join
training_df = pd.merge(ratings_explicit, ratings_test, left_index=True, right_index=True, how='left')

#replace the ratings in training set with 0 if it's in testing set
training_df['Rating_x'] = training_df.apply(lambda row: row['Rating_x'] if np.isnan(row['Rating_y']) else 0, axis=1)

training_df = training_df.drop (['UserID_y','ISBN_y','Rating_y'], axis=1).rename(columns={'UserID_x':'UserID', 'ISBN_x':'ISBN','Rating_x':'Rating'})
#print(training_df)

# craete datframe with rows as users and columns as books
training_ratings = training_df.pivot_table(values= "Rating", index= 'UserID' , columns='ISBN')



rating_train_matrix = training_ratings.as_matrix()

for i in range(0,rating_train_matrix.shape[0]):
    for j in range(0,rating_train_matrix.shape[1]):
        if rating_train_matrix[i][j] == 0:
            rating_train_matrix[i][j] = np.nan
            
print(rating_train_matrix)
print(rating_train_matrix.shape)


# In[128]:


### create testing set by keep the ratings in the testing set and replace the other ratings with zeros

test_df = ratings_explicit.copy()

testing_df = pd.merge(ratings_test, test_df, left_index=True, right_index=True, how='right')
testing_df= testing_df.drop(['UserID_x', 'ISBN_x','Rating_y' ], axis=1).rename(columns={'UserID_y':'UserID', 'Rating_x':'Rating', 'ISBN_y':'ISBN'})
testing_df =testing_df.fillna(0)    
#x = x.reindex()
print(testing_df)
testing_ratings = testing_df.pivot_table(values= "Rating", index= 'UserID' , columns='ISBN')

rating_test_matrix = testing_ratings.as_matrix()
rating_test_matrix[np.isnan(rating_test_matrix)] = 0
print(rating_test_matrix)

print(rating_test_matrix.shape)


# In[129]:


#### reference : https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html

#### find the ratings mean in training matrix for each user

mean_user_rating = np.nanmean(rating_train_matrix, axis=1) 

print('mean_user_ratings')
print(mean_user_rating)


# In[130]:


#### convert the training and testing dataset to matrix of user-bok with zeroes for unrated books
rating_train_matrix[np.isnan(rating_train_matrix)] = 0

print(rating_train_matrix)
print(rating_train_matrix.shape)


# In[131]:


### reference https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb
### normalized the ratings for each user 

ratings_diff = (rating_train_matrix - mean_user_rating[:, np.newaxis])
print('ratings_different')
print(ratings_diff)
print()

### return the unkown ratings in ratings_diff to zeros
for i in range(0,rating_train_matrix.shape[0]):
    for j in range(0,rating_train_matrix.shape[1]):
        if rating_train_matrix[i][j] == 0:
            ratings_diff[i][j] = 0

print('ratings_different')                       
print(ratings_diff)


# In[132]:


### reference https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb

### find the similarity between the users based on ratings activity

user_rating_sim = 1 - sklearn.metrics.pairwise.cosine_distances(rating_train_matrix)

#user_rating_sim = np.array(user_rating_sim)

print('user_rating_similarty')
print(user_rating_sim)


# In[133]:


### make binary matrix of zeros for unrated books and ones for rated books

new_train = np.zeros(rating_train_matrix.shape)
print(new_train)
for i in range(rating_train_matrix.shape[0]):
    for j in range(rating_train_matrix.shape[1]):
        if rating_train_matrix[i][j] != 0:
            new_train[i][j] = 1
              
print(new_train)


# In[134]:


### reference https://github.com/khanhnamle1994/movielens/blob/master/Content_Based_and_Collaborative_Filtering_Models.ipynb

## predict the ratings for all users

user_pred =   mean_user_rating[:, np.newaxis] + (np.array(user_rating_sim.dot(ratings_diff)) / np.array(np.abs(user_rating_sim).dot(new_train)))
print('user_prediction')


user_pred[np.isnan(user_pred)] = 0
print(user_pred)

print('user_prediction_shape', user_pred.shape)


# In[135]:


### Reference; Gorakala, Suresh Kumar “Building Recommendation” understand your data and user preferences to make intelligent, accurate, 
### and profitable decisions”, Packt Publishing, 2016

### find the arrays of real and predict ratings for training dataset to calculate the mean error 

print(rating_train_matrix.nonzero())

user_pred_train = user_pred[rating_train_matrix.nonzero()].flatten()
actual_train = rating_train_matrix[rating_train_matrix.nonzero()].flatten()
##
print('user_predict_train')
print(user_pred_train)
print()
print('actual_rating_train')
print(actual_train)

## calculate prediction errors
from sklearn import metrics 
print('MAE:',  metrics.mean_absolute_error(user_pred_train, actual_train))
print('MSE:',  metrics.mean_squared_error(user_pred_train, actual_train))
print('RMSE:', np.sqrt(metrics.mean_squared_error(user_pred_train, actual_train)))
##


# In[136]:


### find the arrays of real and predict ratings for testing dataset to calculate the mean error 

print(rating_test_matrix.nonzero())

user_pred_test = user_pred[rating_test_matrix.nonzero()].flatten()
actual_test = rating_test_matrix[rating_test_matrix.nonzero()].flatten()
##
print('user_prediction_test')
print(user_pred_test)
print()
print('actual_rating_test')
print(actual_test)
##

## calculate prediction errors
from sklearn import metrics 
print('MAE:',  metrics.mean_absolute_error(user_pred_test, actual_test))
print('MSE:',  metrics.mean_squared_error(user_pred_test, actual_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(user_pred_test, actual_test)))


# In[137]:


#### convert the predict matrix as dataframe

preds_df = pd.DataFrame(user_pred , columns = training_ratings.columns)
print('preds_rating_df')
print(preds_df)


# In[138]:


### find the all predicted ratings for user-id=1 as example 

userID =  1
user_row_number = userID - 1 # User ID starts at 1, not 0


sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1

sorted_user_predictions = pd.DataFrame(sorted_user_predictions).rename(columns={user_row_number:'predicted_rate'})

print( 'sorted_user_predictions for user', userID)
print( sorted_user_predictions)


# In[139]:


### create dataframe of real ratings and books information in the train and test set for user-id=254 as example 
book_matrix = pd.DataFrame(book_data)
 
user_info_test = ratings_test[ratings_test ['UserID'] == 254]
user_info_train = ratings_train[ratings_train ['UserID'] == 254]

#print(user_info_test)
#print(user_info_train)


user_full_test = (user_info_test.merge(book_matrix, how = 'left', left_on = 'ISBN', right_on = 'ISBN').sort_values(['Rating'], ascending=False))

user_full_train = (user_info_train.merge(book_matrix, how = 'left', left_on = 'ISBN', right_on = 'ISBN').sort_values(['Rating'], ascending=False))

print('user_full_test for user', userID)
print(user_full_test)
print('user_full_train for user', userID)
print(user_full_train)


# In[140]:


### recommend top-10 list for user-id=254 after removed the already rated books as example

already_rated_movies =  set(user_full_test['ISBN']).union(set(user_full_train['ISBN'])) 
print(already_rated_movies)
#
recommendations = (book_matrix[~book_matrix['ISBN'].isin(already_rated_movies)].                   merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'ISBN',right_on = 'ISBN').                   drop(['publisher'], axis =1).                   sort_values('predicted_rate', ascending = False).iloc[:10, :-1])

print(recommendations)


# In[141]:


### real ratings and books information about all the users in test set sorted by user id

all_user_full_test = (ratings_test.merge(book_matrix, how = 'left', left_on = 'ISBN', right_on = 'ISBN').                      drop(['publisher', 'yearOfPublication'], axis =1).                      sort_values(['UserID'], ascending=True))
print(all_user_full_test)


# In[142]:


#### create dataframe of all users with real and predicted ratings of the testing set

actual_test_rating = pd.DataFrame(actual_test)
actual_test_rating.rename(columns={'0':'Rating'})
#print(actual_test_rating)

pred_test_rating = pd.DataFrame(user_pred_test)
#print(pred_test_rating)

### index is the index of ratings in testing matrix 
index = pd.DataFrame(rating_test_matrix.nonzero()[0]).merge(pd.DataFrame(rating_test_matrix.nonzero()[1]), left_index = True, right_index= True)
#print(index)

ratings = actual_test_rating.merge(pred_test_rating, left_index = True, right_index= True)
#print(ratings)
predict = index.merge(ratings, left_index = True, right_index= True).rename(columns={'0_x_x':'UserID', '0_x_y':'Rating', '0_y_y':'predicted'})

UserID = predict.UserID +1

predict['UserID'] = UserID

print(predict)


# In[143]:


#### create dataframe of all users with real and predicted ratings of the testing set sorted descending by predicted rating 

all_user_real_test = ratings_test.merge(book_matrix, how = 'left', left_on = 'ISBN', right_on = 'ISBN').sort_values(['UserID', 'ISBN'], ascending=[True, True])

all_user_real_test = all_user_real_test.reset_index().drop(['index'], axis =1)
#print(all_user_real_test)

predictions = all_user_real_test.merge(predict, left_index = True, right_index= True)            .drop(['bookTitle','publisher', 'bookAuthor', 'yearOfPublication', 'UserID_y', '0_y_x', 'Rating_y'], axis =1)            .sort_values(['UserID_x', 'predicted'], ascending=[True, False])            .rename(columns={'UserID_x':'UserID', 'Rating_x':'Rating', 'predicted':'predicted_rate'})

print(predictions)


# In[144]:


#### reference : https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py

#### the code illustrates how to compute Precision at k and Recall at k metrics  for all the users.

from collections import defaultdict
#

import statistics 
p3_list=[]
p5_list=[]
p10_list=[]
p15_list=[]
p20_list=[]
p25_list=[]

r3_list=[]
r5_list=[]
r10_list=[]
r15_list=[]
r20_list=[]
r25_list=[]

###  the cut-off values for classify the books as recommened or not recommended
threshold= 6   


k=[3, 5, 10, 15, 20, 25]                     ## list of top-n recommended book
 
for i in predictions['UserID'].unique():     ### loop for all users
    
    
    for n_top in k:                          ### loop for all top-n recommended book



        ## First map the predictions to each user.
        user_est_true = defaultdict(list)
        for  index , row in predictions[predictions ['UserID'] == i].iterrows():
            user_est_true[i].append(( row['predicted_rate'], row['Rating']))
        #
       # print(user_est_true)
        precisions = dict()
        recalls = dict()
        #
        for uid, user_ratings in user_est_true.items():
        #
        ## Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)
        #
        ## Number of relevant items
            n_rel = sum((row['Rating'] >= threshold) for (_, row['Rating']) in user_ratings)
        #
        ## Number of recommended items in top k
            n_rec_k = sum((row['predicted_rate'] >= threshold) for (row['predicted_rate'], _) in user_ratings[:n_top])
        #
        ## Number of relevant and recommended items in top k

            n_rel_and_rec_k = sum(((row['Rating'] >= threshold) and (row['predicted_rate'] >= threshold)) for (row['predicted_rate'], row['Rating']) in user_ratings[:n_top])

        ## Precision@K: Proportion of recommended items that are relevant
        precisions[i] = n_rel_and_rec_k /n_top if n_rec_k != 0 else np.nan



        ## Recall@K: Proportion of relevant items that are recommended
        recalls[i] = n_rel_and_rec_k / n_rel if n_rel != 0 else np.nan


        print('TP(RelRec) =', n_rel_and_rec_k )
        print('TP+FP k-rec =', n_rec_k)
        #
        print('TP+FN rele =', n_rel)
        #

        print(precisions)
        print(recalls)

        if n_top == 3:
            p3_list.append(precisions[i])
            r3_list.append(recalls[i])
        elif n_top == 5:
            p5_list.append(precisions[i])
            r5_list.append(recalls[i])
        elif n_top == 10:
            p10_list.append(precisions[i])
            r10_list.append(recalls[i])
        elif n_top == 15:    
            p15_list.append(precisions[i])
            r15_list.append(recalls[i])
        elif n_top == 20:
            p20_list.append(precisions[i])
            r20_list.append(recalls[i])
        else :
            p25_list.append(precisions[i])
            r25_list.append(recalls[i])

### finding the mean of users' precisions and recalls for the algorithm    

print('meanP@3K =',np.nansum(p3_list)/np.count_nonzero(~np.isnan(p3_list)))
print('meanR@3K =',np.nansum(r3_list)/np.count_nonzero(~np.isnan(r3_list)))            
print('meanP@5K =',np.nansum(p5_list)/np.count_nonzero(~np.isnan(p5_list)))
print('meanR@5K =',np.nansum(r5_list)/np.count_nonzero(~np.isnan(r5_list)))
print('meanP@10K =',np.nansum(p10_list)/np.count_nonzero(~np.isnan(p10_list)))
print('meanR@10K =',np.nansum(r10_list)/np.count_nonzero(~np.isnan(r10_list)))
print('meanP@15K =',np.nansum(p15_list)/np.count_nonzero(~np.isnan(p15_list)))
print('meanR@15K =',np.nansum(r15_list)/np.count_nonzero(~np.isnan(r15_list)))
print('meanP@20K =',np.nansum(p20_list)/np.count_nonzero(~np.isnan(p20_list)))
print('meanR@20K =',np.nansum(r20_list)/np.count_nonzero(~np.isnan(r20_list)))
print('meanP@25K =',np.nansum(p25_list)/np.count_nonzero(~np.isnan(p25_list)))
print('meanR@25K =',np.nansum(r25_list)/np.count_nonzero(~np.isnan(r25_list)))


# In[145]:


#### the code illustrates how to compute true positive rate at k and false positive rate at k metrics for all the users.

from collections import defaultdict
#import statistics 

fp3_list=[]
fp5_list=[]
fp10_list=[]
fp15_list=[]
fp20_list=[]
fp25_list=[]

tp3_list=[]
tp5_list=[]
tp10_list=[]
tp15_list=[]
tp20_list=[]
tp25_list=[]


###  the cut-off values for classify the books as recommened or not recommended
threshold= 6   




k=[3, 5, 10, 15, 20, 25]                     ## list of top-n recommended books
     
for i in predictions['UserID'].unique():     ### loop for all users
    
    
    for n_top in k:                         ### loop for all top-n recommended book


       
        ## First map the predictions to each user.
        user_est_true = defaultdict(list)
        for  index , row in predictions[predictions ['UserID'] == i].iterrows():
            user_est_true[i].append(( row['predicted_rate'], row['Rating']))
        #
        #print(user_est_true)
        fp = dict()
        tp = dict()
        #
        for uid, user_ratings in user_est_true.items():
        #
        ## Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)
        #
        ## Number of relevant and irrelevant items
            n_rel = sum((row['Rating'] >= threshold) for (_, row['Rating']) in user_ratings)
            
            n_irrel =  sum((row['Rating'] < threshold) for (_, row['Rating']) in user_ratings)
        #
        ## Number of recommended items in top k
            n_rec_k = sum((row['predicted_rate'] >= threshold) for (row['predicted_rate'], _) in user_ratings[:n_top])
        #
        ## Number of relevant and recommended items in top k
        ## Number of irrelevant and recommended items in top k
        
            n_rel_and_rec_k = sum(((row['Rating'] >= threshold) and (row['predicted_rate'] >= threshold)) for (row['predicted_rate'], row['Rating']) in user_ratings[:n_top])
            
            n_irrel_and_rec_k = sum(((row['Rating'] < threshold) and (row['predicted_rate'] >= threshold)) for (row['predicted_rate'], row['Rating']) in user_ratings[:n_top])
        
        ## false positive rate @K: Proportion of recommended items that are irrelevant
        
        fp[i] = n_irrel_and_rec_k / n_irrel if n_irrel != 0 else np.nan
        
        
        
        ## true positve value @K: Proportion of relevant items that are recommended
        
        tp[i] = n_rel_and_rec_k / n_rel if n_rel != 0 else np.nan
   
        
        print('(RelRec) =', n_rel_and_rec_k )
        print('irrel k-rec =', n_irrel_and_rec_k)
        print('n_irrel =', n_irrel)
        print('rele =', n_rel)
        #

        print(fp)
        print(tp)
        
        if n_top == 3:
            fp3_list.append(fp[i])
            tp3_list.append(tp[i])
        elif n_top == 5:
            fp5_list.append(fp[i])
            tp5_list.append(tp[i])
        elif n_top == 10:
            fp10_list.append(fp[i])
            tp10_list.append(tp[i])
        elif n_top == 15:    
            fp15_list.append(fp[i])
            tp15_list.append(tp[i])
        elif n_top == 20:
            fp20_list.append(fp[i])
            tp20_list.append(tp[i])
        else:
            fp25_list.append(fp[i])
            tp25_list.append(tp[i])
            
###  finding the mean of users' tp and fp for the algorithm    

print('fp@3K =',np.nansum(fp3_list)/np.count_nonzero(~np.isnan(fp3_list)))
print('tp@3K =',np.nansum(tp3_list)/np.count_nonzero(~np.isnan(tp3_list)))            
print('fp@5K =',np.nansum(fp5_list)/np.count_nonzero(~np.isnan(fp5_list)))
print('tp@5K =',np.nansum(tp5_list)/np.count_nonzero(~np.isnan(tp5_list)))
print('fp@10K =',np.nansum(fp10_list)/np.count_nonzero(~np.isnan(fp10_list)))
print('tp@10K =',np.nansum(tp10_list)/np.count_nonzero(~np.isnan(tp10_list)))
print('fp@15K =',np.nansum(fp15_list)/np.count_nonzero(~np.isnan(fp15_list)))
print('tp@15K =',np.nansum(tp15_list)/np.count_nonzero(~np.isnan(tp15_list)))
print('fp@20K =',np.nansum(fp20_list)/np.count_nonzero(~np.isnan(fp20_list)))
print('tp@20K =',np.nansum(tp20_list)/np.count_nonzero(~np.isnan(tp20_list)))
print('fp@25K =',np.nansum(fp25_list)/np.count_nonzero(~np.isnan(fp25_list)))
print('tp@25K =',np.nansum(tp25_list)/np.count_nonzero(~np.isnan(tp25_list)))


# In[146]:


#### create list of lists of recommended books and list of lists of relevant books for all users  

all_users_act=[]         ### list of relevant books 
all_users_pred =[]       ### list of recommended books

threshold = 6
for i in predictions['UserID'].unique():
   
       
    ## First map the predictions to each user.
    
    user_est_true = defaultdict(list)
    for  index , row in predictions[predictions ['UserID'] == i].iterrows():
        user_est_true[i].append(( row['ISBN'], row['predicted_rate'], row['Rating']))
    #
    #print(user_est_true)

    #
    for uid, user_ratings in user_est_true.items():
    #
    ## Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
      
    user_actual  = []
    user_predict = []
    
    for rate in user_ratings:  
        
        if rate[2] >= threshold :

            user_actual.append(rate[0])
        
        if  rate[1] >= threshold:

             user_predict.append(rate[0])  
        
    all_users_act.append(user_actual)
    all_users_pred.append(user_predict)
    
print(len(all_users_act))
print('######################################')
print(len(all_users_pred))


# In[147]:


### reference : https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb
### finf mean avarage precision (MAP@k) and mean avarage recall (MAR@k)

import recmetrics
mark = []                  
k=[3, 5, 10, 15, 20, 25]
for n_top in k:
    mark.append(recmetrics.mark(all_users_act, all_users_pred, n_top))
    
print('mark=', mark)

#
#
#
import ml_metrics
#
mapk = []
k=[3, 5, 10, 15, 20, 25]
for n_top in k:
    mapk.append(ml_metrics.mapk(all_users_act, all_users_pred, n_top))
#    
print('mapk=', mapk)


# In[148]:


### reference : https://gist.github.com/bwhite/3726239
### find the Score is normalized discounted cumulative gain (ndcg)


## function to find the discounted cumulative gain. the arguments are 
## 1) r : is a list of lists of relevant book for all user. it is binary list which is 1 if the book is recommended and relevant or 0 otherwise
## 2) k : is top_n recommended book
## 3) moethod : If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...] 
            ## If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

def dcg_at_k( r, k, method = 0 ):   

    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            # return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return np.sum(r / denominator_table[:r.shape[0]])
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


### Normalized discounted cumulative gain

def get_ndcg( r, k, method = 0):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)  ## find the maximam dcg@ k
    dcg_min = dcg_at_k(sorted(r), k, method)                ## ## find the minmam dcg@ k
    #assert( dcg_max >= dcg_min )

    if not dcg_max:
        return 0.

    dcg = dcg_at_k(r, k, method)

    #print dcg_min, dcg, dcg_max

    return (dcg - dcg_min) / (dcg_max - dcg_min)  



threshold = 6

from collections import defaultdict
dcg3_list=[]
dcg5_list=[]
dcg10_list=[]
dcg15_list=[]
dcg20_list=[]
dcg25_list=[]


ndcg = dict()

k=[3, 5, 10, 15, 20, 25]
for i in predictions['UserID'].unique():
    
###  First map the predictions to each user by create dictionary of userid as key and 
### prediction and actual ratings as values.

    user_est_true = defaultdict(list)
    for  index , row in predictions[predictions ['UserID'] == i].iterrows():
        user_est_true[i].append(( row['predicted_rate'], row['Rating']))

### create binary list of relevant movies 1 if list which is 1 if the movie is recommended and relevant or 0 otherwise
    
    relevant  = []   
   
    for uid, user_ratings in user_est_true.items():
        #
        ## Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
    
        for rate in user_ratings:                   
            if rate[0] >= threshold  and  rate[1] >= threshold:
                relevant.append(1)
            else:
                relevant.append(0)

#    print(relevant)
    

    for n_top in k:
    #    
        ndcg[i] = get_ndcg( relevant, n_top, method = 0)
        #        
        #    
        #
        if n_top == 3:
            dcg3_list.append(ndcg[i])
        elif n_top == 5:
            dcg5_list.append(ndcg[i])
        elif n_top == 10:
            dcg10_list.append(ndcg[i])
        elif n_top == 15:    
            dcg15_list.append(ndcg[i])
        elif n_top == 20:
            dcg20_list.append(ndcg[i])
        else:
            dcg25_list.append(ndcg[i])

### find the mean of NDCG @ top_n for all user   

print('NDCG top_n 3 =', np.nansum(dcg3_list)/np.count_nonzero(~np.isnan(dcg3_list)))     
print('NDCG top_n 5 =', np.nansum(dcg5_list)/np.count_nonzero(~np.isnan(dcg5_list)))      

print('NDCG top_n 10 =', np.nansum(dcg10_list)/np.count_nonzero(~np.isnan(dcg10_list)))      
print('NDCG top_n 15 =', np.nansum(dcg15_list)/np.count_nonzero(~np.isnan(dcg15_list)))      
print('NDCG top_n 20 =', np.nansum(dcg20_list)/np.count_nonzero(~np.isnan(dcg20_list)))      
print('NDCG top_n 25 =', np.nansum(dcg25_list)/np.count_nonzero(~np.isnan(dcg25_list)))      


# In[149]:


### reference : https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb
### catalog coverage 

import recmetrics

all_users=[]            ## create list contain top_n recommended books
k=[3, 5, 10, 15, 20, 25]
catalog = ratings_explicit.ISBN.unique().tolist()     ### list of unique movies that available in the ratings dataset
for n_top in k:
    for user_list in all_users_pred:
     
        all_users.append(user_list[:n_top])

   
    coverage = recmetrics.coverage(all_users, catalog)   ## find the coerage on top_n 
    print( 'coverage top_n',n_top,'='  , coverage)
#


# In[ ]:




