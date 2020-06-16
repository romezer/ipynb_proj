#!/usr/bin/env python
# coding: utf-8

# # Question 1  

# # 1. Db import

# In[1]:


import pandas as pd;
import numpy as np;

data = pd.read_csv("data.csv", sep = '|');
df = data.drop_duplicates()
df.shape


# In[2]:


df.head(10)


# In[3]:


df.info()


# In[4]:


df.describe()


# # 2. Preprocessing

# # i

# In[5]:


# counting missing values for each col
df.isnull().sum()
df_cleaned = df.copy()


# In[6]:


# remove cloumns with more then 60% nulls
for column in df_cleaned:
    if (100 * df_cleaned[column].isnull().sum()/len(df_cleaned.index)) > 60:
        print(df_cleaned[column].name + ' p: ' + str(100 * df_cleaned[column].isnull().sum()/len(df_cleaned.index) ) )
        del df[column]
        


# In[7]:


len(df_cleaned.index)


# # ii

# In[8]:


df_cleaned.info()


# In[9]:


import matplotlib.pyplot as plt
#Histogram grid
df_cleaned.hist(figsize=(20,20), xrot=-45)
# show
plt.show()


# In[10]:


# if feature have less then 10% nulls we will drop that enteries
for col in df_cleaned:
    if (100 * df_cleaned[col].isnull().sum()/len(df_cleaned.index)) < 10:
        df_cleaned[col] = df_cleaned[col].dropna()


# In[11]:


# as we can see in the plots above most of the featues distrebutes ~ normally -> we will fill the nulls with the mean
for column in df_cleaned:
    df_cleaned[column] = df_cleaned[column].fillna((df_cleaned[column].mean()))


# In[12]:


df_cleaned.describe()


# In[13]:


df_cleaned.hist(column = ['ViolentCrimesPerPop numeric'], bins = 30, color = 'red', alpha = 0.8)
plt.show()


# In[14]:


# Feature Engineering; collinearity
import seaborn as sns
corr = df_cleaned.corr()
fig = plt.figure(figsize = (16, 12))
sns.heatmap(corr, vmax = 0.8)
plt.show()


# In[15]:


corrT = data.corr(method = 'pearson').round(4)
corrT = corrT.sort_values(by=['ViolentCrimesPerPop numeric'])
corrT['ViolentCrimesPerPop numeric']


# In[16]:


X = df_cleaned.iloc[:, 0:126]
y = df_cleaned.iloc[:, 126]

from sklearn.model_selection import train_test_split

seed = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = seed)

print(X.shape)
print(y.shape)


# In[17]:


X.head()


# In[ ]:





# In[18]:


from sklearn.preprocessing import StandardScaler

# Standardize features by removing the mean and scaling to unit variance

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # 3. SHAP

# # i

# In[19]:


# Model Selection - Cross Validation

def myplot(XX, yy):
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=seed)

    results = []
    names = []

    for name, model in models:
        cv_results = cross_val_score(model, XX, yy, cv = cv, scoring = make_scorer(r2_score))
        results.append(cv_results)
        names.append(name)
        msg = "%s: %.3f (+/- %.3f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('R2')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = []
models.append(('LR', LinearRegression()))
models.append(('SVR', SVR()))
models.append(('DTR', DecisionTreeRegressor(random_state=seed)))
models.append(('RFR (100 Trees)', RandomForestRegressor(n_estimators=100, random_state=seed)))


# In[21]:


myplot(X_train, y_train)


# In[22]:


# Build the model with the random forest regression algorithm:
model = RandomForestRegressor(random_state=seed, n_estimators=100)
model.fit(X_train, y_train)


# # ii

# ### Shap values definition: Shapley values calculate the importance of a feature by comparing what a model predicts with and without the feature

# # iii

# In[23]:


pip install shap


# In[24]:


import shap
shap_values = shap.TreeExplainer(model).shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns)


# In[25]:


shap.summary_plot(shap_values, X_test, feature_names=X.columns)


# ## 1. PctIlleg: percentage of kids born to never married - seems reasonble -> less handled kids might become criminals
# ##  2. PctKids2Par: percentage of kids in family housing with two parents -> doesn't sounds reasonble to explain crime
# ## 3. racePctWhite: percentage of population that is caucasian -> sounds reasonble that some ethnic groups more related to crime.
# 

# ## Red/blue: Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are in blue.

# # iv

# In[26]:


df_rand = df_cleaned.sample(n=3)
df_rand.head()


# In[27]:



# Initialize your Jupyter notebook with initjs(), otherwise you will get an error message.
shap.initjs()

# Write in a function
def shap_plot(j):
    explainerModel = shap.TreeExplainer(model)
    shap_values_Model = explainerModel.shap_values(df_rand)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], df_rand.iloc[[j]])
    return(p)


# In[28]:


shap_plot(0)


# In[29]:


shap_plot(1)


# In[30]:


shap_plot(2)


# ## as we saw before the most important feattres that impact local sampales are: PctKids2Par, racePctWhite.
# ## PctKids2Par and racePctWhite seem to be important features in all 3 samples as expected from global interpability.

# # 4. LIME

# # i

# In[31]:


pip install lime


# In[32]:


import lime
import lime.lime_tabular


# In[33]:


explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.columns, class_names=['ViolentCrimesPerPop numeric'],  verbose=True, mode='regression')


# In[34]:


df_rand_x = df_rand.drop(df_rand.columns[len(df_rand.columns)-1], axis=1)


# In[35]:


df_rand_x_val = df_rand_x.values


# In[ ]:





# In[36]:


exp1 = explainer.explain_instance(df_rand_x_val[0], model.predict, num_features=5)


# In[37]:


exp1.show_in_notebook(show_table=True)


# In[38]:


exp2 = explainer.explain_instance(df_rand_x_val[1], model.predict, num_features=5)


# In[39]:


exp2.show_in_notebook(show_table=True)


# In[40]:


exp3 = explainer.explain_instance(df_rand_x_val[1], model.predict, num_features=5)


# In[41]:


exp3.show_in_notebook(show_table=True)


# In[42]:


exp1.as_list()


# In[43]:


exp2.as_list()


# In[44]:


exp3.as_list()


# ## As we can the most important features according to lime model are:
# ## PctKids2Par
# ## racePctWhite

# # ii

# ## We can see high correlation betweem shap and lime, the most important features are: PctKids2Par and racePctWhite.
