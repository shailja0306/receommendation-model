#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


# In[3]:


movies.head()


# In[4]:


ratings.head()


# In[5]:


final_ratings = ratings.pivot(index='movieId', columns='userId', values='rating')
final_ratings.head()


# In[6]:


final_ratings.fillna(0, inplace=True)
final_ratings.head()


# In[7]:


no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movie_voted = ratings.groupby('userId')['rating'].agg('count')


# In[8]:


user_voted_threshold = 10

f, ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_user_voted.index, no_user_voted, color='mediumseagreen')
plt.axhline(y=user_voted_threshold, color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()


# In[9]:


final_ratings = final_ratings.loc[no_user_voted[no_user_voted > user_voted_threshold].index, :]


# In[10]:


movie_voted_threshold = 50

f, ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movie_voted.index, no_movie_voted, color='mediumseagreen')
plt.axhline(y=movie_voted_threshold, color='r')
plt.xlabel('UserId')
plt.ylabel('No. of movies voted by user')
plt.show()


# In[11]:


final_ratings = final_ratings.loc[:, no_movie_voted[no_movie_voted > movie_voted_threshold].index]


# In[12]:


final_ratings


# In[13]:


csr_data = csr_matrix(final_ratings.values)
final_ratings.reset_index(inplace=True)


# In[14]:


csr_data.data


# In[15]:


knn = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=20)
knn


# In[16]:


knn.fit(csr_data)


# In[17]:


def get_movie_recommendation(movie_name):
  n_movies_to_recommend = 10
  movie_list = movies[movies['title'].str.contains(movie_name)]
  if len(movie_list):
    movie_idx = movie_list.iloc[0]['movieId']
    movie_idx = final_ratings[final_ratings['movieId'] == movie_idx].index[0]
    distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend+1)
    rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    recommend=[]
    for val in rec_movie_indices:
      movie_idx = final_ratings.iloc[val[0]]['movieId']
      idx = movies[movies['movieId'] == movie_idx].index
      recommend.append({
          'Title': movies.iloc[idx]['title'].values[0],
          'Distance': val[1]
      })
    df = pd.DataFrame(recommend, index = range(1,n_movies_to_recommend+1))
    return df['Title'].tolist()
  else:
    return "No movie found with that name. Please check your input."


# In[18]:


a = get_movie_recommendation('American Beauty')
a


# In[19]:


type(a)


# In[20]:


model = get_movie_recommendation('movie_name')


# In[21]:


import pickle


# In[23]:


knnpickle =  open('model_pickle', 'wb')
pickle.dump(knn, knnpickle )


# In[26]:



 pickle.load(open('model_pickle', 'rb'))


# In[ ]:


mp


# In[ ]:




