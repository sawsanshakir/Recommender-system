#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt


movie_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\movies.txt',  names= ['MovieID','Title','Genres'] , encoding='latin-1')
rating_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\ratings.txt',  names= ['UserID','MovieID','Rating','Timestamp'])
user_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\users.txt',  names= ['UserID','Gender','Age','Occupation','Zip-code'])

print(rating_data.dtypes)

from plotly.offline import init_notebook_mode, plot, iplot

import plotly.graph_objs as go

#### Reference:https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b

### find ratings distribution

init_notebook_mode(connected=True)
data = rating_data['Rating'].value_counts().sort_index(ascending=False)
trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / rating_data.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               )
layout = dict(title = 'Distribution Of {} movies-ratings'.format(rating_data.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)


plt.show()


# In[2]:


## count the number of users for each rate scale

rating_data.groupby(['Rating'])['UserID'].count()


# In[16]:


### find the sparcity of the ratings dataset

n_users = len(rating_data["UserID"].unique())
n_movies = len(rating_data["MovieID"].unique())

print ('Matrix sparcity: ' + str(100.0 - (100.0 * len(rating_data) / (n_users * n_movies))) + ' %')

sparsity = len(rating_data)/(n_users*n_movies)
print("sparsity of ratings is %.2f%%" %(sparsity*100))


# In[3]:


## find distribution of ratings per movie 

plt.hist(rating_data.groupby(['MovieID'])['MovieID'].count())
plt.xlim(0, 2500)
plt.ylim(10, 2800, 10)
plt.title('Rating Distribution\n')
plt.xlabel('Movies')
plt.ylabel('Count')

plt.show()


# In[7]:


## find distribution of ratings per user

plt.hist(rating_data.groupby(['UserID'])['UserID'].count())
plt.xlim(10, 1500)
plt.ylim(0, 5000, 10)
plt.title('Rating Distribution\n')
plt.xlabel('Users')
plt.ylabel('Count')

plt.show()


# In[66]:


### find the age distribution

import matplotlib.pyplot as plt
plt.hist(user_data.Age, bins=[1,18,25,35,45,50,56])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')

plt.show()
#1:  "Under 18"
#	* 18:  "18-24"
#	* 25:  "25-34"
#	* 35:  "35-44"
#	* 45:  "45-49"
#	* 50:  "50-55"
#	* 56:  "56+"


# In[11]:


import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from math import sqrt
#from sklearn.metrics import ml_metrics as metrics

movie_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\movies.txt',  names= ['MovieID','Title','Genres'] , encoding='latin-1')
rating_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\ratings.txt',  names= ['UserID','MovieID','Rating','Timestamp'])
user_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\users.txt',  names= ['UserID','Gender','Age','Occupation','Zip-code'])
print(rating_data.shape)
#

ratings_train, ratings_test = train_test_split(rating_data, stratify=rating_data['UserID'], test_size=0.30, random_state=42)
#
print(ratings_train.shape)
print(ratings_test.shape)



# In[19]:


## minimum number of ratings for a user in the rating dataset
rating_data.groupby('UserID').count()[['Rating']].min()


# In[20]:


## maximum number of ratings for a user in the rating dataset
rating_data.groupby('UserID').count()[['Rating']].max()


# In[15]:


## minimum number of ratings for a user in the training rating dataset
ratings_train.groupby('UserID').count()[['Rating']].min()


# In[16]:


## minimum number of ratings for a user in the testing rating dataset

ratings_test.groupby('UserID').count()[['Rating']].min()


# In[17]:


## maximum number of ratings for a user in the training rating dataset
ratings_train.groupby('UserID').count()[['Rating']].max()


# In[18]:


## maximum number of ratings for a user in the testing rating dataset
ratings_test.groupby('UserID').count()[['Rating']].max()


# In[ ]:




