#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


df = pd.read_csv("/home/master/Documents/datasets/train.csv")


# In[14]:


df.head()


# In[15]:


df.describe()


# In[49]:



fig = plt.figure(figsize = (10,5))

df['Survived'].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5)

plt.title('Survived')


# In[27]:


plt.scatter(df['Survived'] , df['Age'] , alpha = 0.5)


# In[29]:


df['Pclass'].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5)

plt.title('Pclass')


# In[36]:


for x in range(1,4):
    df['Age'][df['Pclass'] == x].plot(kind = 'kde' , alpha = 0.5)
plt.title('Class with respect to Age')
plt.legend(('1st' , '2nd' , '3rd'))


# In[37]:


df['Embarked'].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5)

plt.title('Embarked')


# In[38]:


df['Survived'][df['Sex'] == 'male'].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5)

plt.title('Men Survived')


# In[40]:


df['Survived'][df['Sex'] == 'female'].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5 , color = 'r')

plt.title('Women Survived')


# In[46]:


df['Sex'][df['Survived'] == 1].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5 , color = ['r', 'b'])

plt.title('Sex of Survived')


# In[50]:




fig = plt.figure(figsize = (10,5))

for x in range(1,4):
    df['Survived'][df['Pclass'] == x].plot(kind = 'kde' , alpha = 0.5)
plt.title('Class with respect to survived')
plt.legend(('1st' , '2nd' , '3rd'))


# In[53]:


df['Survived'][(df['Sex'] == 'male') & (df['Pclass'] == 1)].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5 )

plt.title('Rich Men Survived')


# In[54]:


df['Survived'][(df['Sex'] == 'male') & (df['Pclass'] == 3)].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5 )

plt.title('Poor Men Survived')


# In[55]:


df['Survived'][(df['Sex'] == 'female') & (df['Pclass'] == 1)].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5 , color = 'r')

plt.title('Rich Women Survived')


# In[58]:


df['Survived'][(df['Sex'] == 'female') & (df['Pclass'] == 3)].value_counts(normalize = True).plot(kind = 'bar' , alpha = 0.5 , color = 'r' )

plt.title('Poor Women Survived')


# In[ ]:





# In[ ]:





# In[ ]:




