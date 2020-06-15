#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from math import sqrt
#from sklearn.metrics import ml_metrics as metrics

book_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Books.csv',  names= ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM', 'imageURLL'] ,delimiter=";", encoding='latin-1')
rating_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Book-Ratings.csv',  names= ['UserID','ISBN','Rating'],delimiter=";", encoding='latin-1')
user_data = pd.read_csv(r'C:\Users\Soona\Recommender sys\book crossing\BX-Users.csv',  names= ['UserID','Location','Age'],delimiter=";", encoding='latin-1')
print(rating_data.shape)
print(book_data.shape)
print(user_data.shape)
#


#
#print(rating_data)
#rating_data.ISBN = rating_data.ISBN.astype(str)

print(rating_data.Rating.unique())


# In[58]:


ratings_new = rating_data[rating_data.ISBN.isin(book_data.ISBN)]

print (rating_data.shape)
print (ratings_new.shape)


# In[59]:


ratings = rating_data[rating_data.UserID.isin(user_data.UserID)]

print(ratings.shape)


# In[76]:


ratings_explicit = ratings_new[ratings_new.Rating != 0]
ratings_implicit = ratings_new[ratings_new.Rating == 0]
#
#
##print (ratings_new.shape)
print (ratings_explicit.shape)


# In[61]:


from plotly.offline import init_notebook_mode, plot, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go

init_notebook_mode(connected=True)
data = ratings_explicit['Rating'].value_counts().sort_index(ascending=False)
trace = go.Bar(x = data.index,

               text = ['{:.3f} %'.format(val) for val in (data.values / rating_data.shape[0] * 100)],

               textposition = 'outside',

               textfont = dict(color = '#000000'),

               y = data.values,

               )
layout = dict(title = 'Distribution Of {} books-ratings'.format(rating_data.shape[0]),

              xaxis = dict(title = 'Rating'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)

plt.show()


# In[62]:


ratings_explicit.groupby(['Rating'])['UserID'].count()


# In[63]:


n_users = len(ratings_explicit["UserID"].unique())
n_books = len(ratings_explicit["ISBN"].unique())

#print ('Matrix sparcity:' + str(100.0 - (100.0 * len(rating_data) / n_m )) + ' %')
print ('Matrix sparcity: ' + str(100.0 - (100.0 * len(ratings_explicit) / (n_users * n_books))) + ' %')

sparsity = len(ratings_explicit)/(n_users*n_books)
print("sparsity of ratings is %.3f%%" %(sparsity*100))


# In[73]:


plt.hist(ratings_explicit.groupby(['ISBN'])['ISBN'].count())
plt.xlim(0, 80)
plt.ylim(10, 5500, 10)
plt.title('Rating Distribution\n')
plt.xlabel('ISBN')
plt.ylabel('Count')

plt.show()


# In[72]:


plt.hist(ratings_explicit.groupby(['UserID'])['UserID'].count())
plt.xlim(10, 600)
plt.ylim(0, 1200, 10)
plt.title('Rating Distribution\n')
plt.xlabel('Users')
plt.ylabel('Count')

plt.show()


# In[66]:


import matplotlib.pyplot as plt
plt.hist(user_data.Age, bins=[10,20,30,40,50,50,60,70,80,90,100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')

plt.show()


# In[77]:





## considering users who have rated atleast 20 books and books which have atleast 10 ratings
counts = ratings_explicit['ISBN'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['ISBN'].isin(counts[counts >= 10].index)]


count1 = ratings_explicit['UserID'].value_counts()
print(count1)
ratings_explicit = ratings_explicit[ratings_explicit['UserID'].isin(count1[count1 >= 20].index)]
print(ratings_explicit['UserID'].value_counts())

print(ratings_explicit.shape)
#

### split the ratings table into taining and testing dataset

ratings_train, ratings_test = train_test_split(ratings_explicit, stratify=ratings_explicit['UserID'],test_size=0.30, random_state=0)
#

print('ratings_train =')
print(ratings_train.shape)
print()


print('ratings_test =')
print(ratings_test.shape)


# In[78]:


ratings_train.groupby('UserID').count()[['Rating']].min()


# In[79]:


ratings_train.groupby('UserID').count()[['Rating']].max()


# In[80]:


ratings_test.groupby('UserID').count()[['Rating']].min()


# In[81]:


ratings_test.groupby('UserID').count()[['Rating']].max()


# In[ ]:




