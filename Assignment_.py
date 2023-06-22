#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[8]:


book=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment10\books.csv",encoding='latin-1')


# In[9]:


book


# In[10]:


book.head()


# In[11]:


book= book.drop(['Unnamed: 0'], axis=1).rename(columns={"User.ID": "User_ID", "Book.Title": "Book_Title","Book.Rating": "Book_Rating" })


# In[12]:


book


# In[13]:


User_ID_unique=book.User_ID.unique()


# In[14]:



User_ID_unique=pd.DataFrame(User_ID_unique)


# In[15]:


len(book.Book_Title.unique())


# In[16]:


user_book = book.pivot_table(index='User_ID',
                                 columns='Book_Title',
                                 values='Book_Rating')


# In[17]:


user_book


# In[18]:


user_book.index = book.User_ID.unique()


# In[19]:


user_book


# In[20]:


user_book.fillna(0, inplace=True)


# In[21]:


user_book


# In[22]:


user_sim = 1 - pairwise_distances( user_book.values,metric='cosine')


# In[23]:


user_sim


# In[24]:


user_sim.shape


# In[25]:


user_sim_df = pd.DataFrame(user_sim)


# In[26]:


user_sim_df.index = book.User_ID.unique()
user_sim_df.columns = book.User_ID.unique()


# In[27]:


user_sim_df.iloc[0:5, 0:5]


# In[28]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[29]:


user_sim_df.idxmax(axis=1)


# In[30]:


book[(book['User_ID']==276729) | (book['User_ID']==276726)]


# In[31]:


user_1=book[book['User_ID']==276729]


# In[32]:


user_2=book[book['User_ID']==276726]


# In[33]:


user_2.Book_Title


# In[34]:


user_1.Book_Title


# In[35]:


pd.merge(user_1,user_2,on='Book_Title',how='outer')


# In[ ]:




