#!/usr/bin/env python
# coding: utf-8

# In[1]:


## This program for implementation of singular value decomposition (SVD) for ML-1m dataset

import pandas as pd
import numpy as np
import recmetrics
from sklearn.model_selection import train_test_split
from collections import defaultdict
import time


movie_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\movies.txt',  names= ['MovieID','Title','Genres'] , encoding='latin-1')
rating_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\ratings.txt',  names= ['UserID','MovieID','Rating','Timestamp'])
user_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\users.txt',  names= ['UserID','Gender','Age','Occupation','Zip-code'])


#### split the ratings table into taining and testing dataset

ratings_train, ratings_test = train_test_split(rating_data, stratify=rating_data['UserID'], test_size=0.30, random_state=42)
#
print('ratings_train =')
print(ratings_train)
print()
print('ratings_test =')
print(ratings_test)


#####################################################################################################


# In[2]:


### create the training ratings matrix from the ratings set after hiding the testing set


# merge ratings set with the testing set using left join
training_df = pd.merge(rating_data, ratings_test, left_index=True, right_index=True, how='left')

#replace the ratings in training set with 0 if it's in testing set
training_df['Rating_x'] = training_df.apply(lambda row: row['Rating_x'] if np.isnan(row['Rating_y']) else 0, axis=1)

training_df = training_df.drop (['UserID_y','MovieID_y','Rating_y','Timestamp_y'], axis=1).rename(columns={'UserID_x':'UserID', 'MovieID_x':'MovieID','Rating_x':'Rating'})
print(training_df)

# craete datframe with rows as users and columns as movies
training_ratings = training_df.pivot_table(values= "Rating", index= 'UserID' , columns='MovieID')



rating_train_matrix = training_ratings.as_matrix()

for i in range(0,rating_train_matrix.shape[0]):
    for j in range(0,rating_train_matrix.shape[1]):
        if rating_train_matrix[i][j] == 0:
            rating_train_matrix[i][j] = np.nan
            
print(rating_train_matrix)
print(rating_train_matrix.shape)


# In[3]:


### create testing set by keep the ratings in the testing set and replace the other ratings with zeros

test_df = rating_data.copy()

testing_df = pd.merge(ratings_test, test_df, left_index=True, right_index=True, how='right')
testing_df= testing_df.drop(['UserID_x', 'MovieID_x','Rating_y','Timestamp_x' ], axis=1).rename(columns={'UserID_y':'UserID', 'Rating_x':'Rating', 'MovieID_y':'MovieID'})
testing_df =testing_df.fillna(0)    

print(testing_df)
testing_ratings = testing_df.pivot_table(values= "Rating", index= 'UserID' , columns='MovieID')

rating_test_matrix = testing_ratings.as_matrix()
rating_test_matrix[np.isnan(rating_test_matrix)] = 0
print(rating_test_matrix)

print(rating_test_matrix.shape)


# In[4]:


#### reference : https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmean.html

#rating_df = pd.DataFrame(rating_data)

mean_user_ratings = np.nanmean(rating_train_matrix, axis=1)
print(mean_user_ratings)


# In[5]:


#### convert the training and testing dataset to matrix of user-movie with zeroes for unrated movies

#### convert the training and testing dataset to matrix of user-movie with zeroes for unrated movies
rating_train_matrix[np.isnan(rating_train_matrix)] = 0

print(rating_train_matrix)
print(rating_train_matrix.shape)


# In[6]:


mean_u_ratings = rating_train_matrix - mean_user_ratings.reshape(-1, 1)
print(mean_u_ratings)

print(len(mean_u_ratings))

for i in range(0,rating_train_matrix.shape[0]):
    for j in range(0,rating_train_matrix.shape[1]):
        if rating_train_matrix[i][j] == 0:
            mean_u_ratings[i][j] = 0

print(mean_u_ratings)
          


# In[7]:


### reference: https://github.com/khanhnamle1994/movielens/blob/master/SVD_Model.ipynb

## use scipy function to do the singular value decomposition

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(mean_u_ratings, k = 50)

sigma = np.diag(sigma)
print(sigma)


# In[8]:


### Reference:  https://simplyml.com/generating-recommendations/

## predict the ratings by multiplying the three matrices U, sigma and Vt

predict_rating = np.dot(np.dot(U, sigma), Vt) + mean_user_ratings.reshape(-1, 1)
print(predict_rating)
print(predict_rating.shape)



# In[9]:


### Reference; Gorakala, Suresh Kumar “Building Recommendation” understand your data and user preferences to make intelligent, accurate, 
### and profitable decisions”, Packt Publishing, 2016

### find the arrays of real and predict ratings for training dataset to calculate the mean error 

print(rating_train_matrix.nonzero())

user_pred_train = predict_rating[rating_train_matrix.nonzero()].flatten()
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


# In[10]:


### find the arrays of real and predict ratings for testing dataset to calculate the mean error 

print(rating_test_matrix.nonzero())

user_pred_test = predict_rating[rating_test_matrix.nonzero()].flatten()
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


# In[11]:


#### convert the predict matrix as dataframe

preds_df = pd.DataFrame(predict_rating , columns = training_ratings.columns)
print('preds_rating_df')
print(preds_df)


# In[12]:


### find the all predicted ratings for user-id=1 as example 

userID =  1
user_row_number = userID - 1 # User ID starts at 1, not 0
movie_matrix = pd.DataFrame(movie_data)

sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1

sorted_user_predictions = pd.DataFrame(sorted_user_predictions).rename(columns={user_row_number:'predicted_rate'})

print( 'sorted_user_predictions for user', userID)
print( sorted_user_predictions)


# In[13]:


### create dataframe of real ratings and movie information in the train and test set for user-id=1 as example 



user_info_test = ratings_test[ratings_test ['UserID'] == user_row_number+1]
user_info_train = ratings_train[ratings_train ['UserID'] == user_row_number+1]

#print(user_info_test)
#print(user_info_train)


user_full_test = (user_info_test.merge(movie_matrix, how = 'left', left_on = 'MovieID', right_on = 'MovieID').sort_values(['Rating'], ascending=False))

user_full_train = (user_info_train.merge(movie_matrix, how = 'left', left_on = 'MovieID', right_on = 'MovieID').sort_values(['Rating'], ascending=False))

print('user_full_test for user', userID)
print(user_full_test)
print('user_full_train for user', userID)
print(user_full_train)

print('###################################################')

### recommend top-10 list for user-id=1 after removed the already rated movies as example


already_rated_movies =  set(user_full_test['MovieID']).union(set(user_full_train['MovieID'])) 
print(already_rated_movies)
#
recommendations = (movie_matrix[~movie_matrix['MovieID'].isin(already_rated_movies)].                   merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'MovieID',right_on = 'MovieID').                   sort_values('predicted_rate', ascending = False).iloc[:10, :-1])

print(recommendations)
#


# In[14]:


### real ratings and movies information about all the users in test set sorted by user id

all_user_full_test = (ratings_test.merge(movie_matrix, how = 'left', left_on = 'MovieID', right_on = 'MovieID').sort_values(['UserID'], ascending=True))
print(all_user_full_test)


# In[15]:


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


# In[16]:


#### create dataframe of all users with real and predicted ratings of the testing set sorted descending by predicted rating 

all_user_real_test = ratings_test.merge(movie_matrix, how = 'left', left_on = 'MovieID', right_on = 'MovieID').sort_values(['UserID', 'MovieID'], ascending=[True, True])

all_user_real_test = all_user_real_test.reset_index().drop(['index'], axis =1)
#print(all_user_real_test)

predictions = all_user_real_test.merge(predict, left_index = True, right_index= True)            .drop(['Timestamp', 'Title', 'Genres', 'UserID_y', '0_y_x', 'Rating_y'], axis =1)            .sort_values(['UserID_x', 'predicted'], ascending=[True, False])            .rename(columns={'UserID_x':'UserID', 'Rating_x':'Rating', 'predicted':'predicted_rate'})

print(predictions)


# In[17]:


##### reference : https://github.com/NicolasHug/Surprise/blob/master/examples/precision_recall_at_k.py
#
##### the code illustrates how to compute Precision at k and Recall at k metrics  for all the users.
#


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

###  the cut-off values for classify the movies as recommened or not recommended
threshold= 3.5   


k=[3, 5, 10, 15, 20, 25]                    ## list of top-n recommended movies

for i in predictions['UserID'].unique():     ### loop for all users
    
   
    for n_top in k:                          ### loop for all top-n recommended movies



        ## First map the predictions to each user.
        user_est_true = defaultdict(list)
        for  index , row in predictions[predictions ['UserID'] == i].iterrows():
            user_est_true[i].append(( row['predicted_rate'], row['Rating']))
        #
        #print(user_est_true)
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
        precisions[i] = n_rel_and_rec_k / n_top if n_rec_k != 0 else np.nan



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


# In[18]:


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

###  the cut-off values for classify the movies as recommened or not recommended
threshold= 3.5   


k=[3, 5, 10, 15, 20, 25]                    ## list of top-n recommended movies


for i in predictions['UserID'].unique():     ### loop for all users
    
   
    for n_top in k:                           ### loop for all top-n recommended movies


       
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
            n_rec_k = sum((row['predicted_rate'] >= threshold) for (row['predicted_rate'], _) in user_ratings)
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


# In[19]:


#### create list of lists of recommended movies and list of lists of relevant movies for all users  

all_users_act=[]         ### list of relevant movies 
all_users_pred =[]       ### list of recommended movies

threshold = 3.5
for i in range(1,6041):
   
       
    ## First map the predictions to each user.
    
    user_est_true = defaultdict(list)
    for  index , row in predictions[predictions ['UserID'] == i].iterrows():
        user_est_true[i].append(( row['MovieID'], row['predicted_rate'], row['Rating']))
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


# In[20]:


### reference : https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb
### find mean avarage precision (MAP@k) and mean avarage recall (MAR@k)

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


# In[21]:


### reference : https://gist.github.com/bwhite/3726239
### find the Score is normalized discounted cumulative gain (ndcg)


## function to find the discounted cumulative gain. the arguments are 
## 1) r : is a list of lists of relevant movies for all user. it is binary list which is 1 if the movie is recommended and relevant or 0 otherwise
## 2) k : is top_n recommended movies
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



threshold = 3.5

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


# In[22]:


### reference : https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb
### catalog coverage 

import recmetrics

all_users=[]            ## create list contain top_n recommended movies
k=[3, 5, 10, 15, 20, 25]
catalog = rating_data.MovieID.unique().tolist()     ### list of unique movies that available in the ratings dataset
for n_top in k:
    for user_list in all_users_pred:
     
        all_users.append(user_list[:n_top])

   
    coverage = recmetrics.coverage(all_users, catalog)   ## find the coerage on top_n 
    print( 'coverage top_n',n_top,'='  , coverage)
#


# In[ ]:




