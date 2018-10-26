
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sqlite3


# In[2]:


# Create your connection. 
cnx = sqlite3.connect('mortgage.db') 
loan_df_new = pd.read_sql_query("SELECT * FROM loan_data", cnx)


# In[3]:


# Droppping the additional index column
loan_df_new = loan_df_new.drop('index', axis=1)
loan_df_new.head(5)


# In[4]:


loan_df_new.shape


# In[5]:


#dropping 7 columns which have same value for all rows.  
#Current total should be 44 columns
loan_df_new = loan_df_new.drop(['state_name','state_abbr','state_code','respondent_id','owner_occupancy_name',
                                'lien_status_name','agency_abbr'],1)


# In[6]:


loan_df_new.shape


# In[8]:


# Checking datatypes to convert all categorical to numerical
loan_df_new.dtypes


# In[9]:


# Converting categorical columns to numerical with one-hot encoding technique (15 columns in total)

unique_agency = loan_df_new['agency_name'].value_counts()
print(unique_agency)

agency_dummy = pd.get_dummies(loan_df_new['agency_name'],prefix = 'agency')
loan_df_new = pd.concat([loan_df_new,agency_dummy],axis=1)
loan_df_new.shape


# In[10]:


unique_ethnicity = loan_df_new['applicant_ethnicity_name'].value_counts()
print(unique_ethnicity)

ethnicity_dummy = pd.get_dummies(loan_df_new['applicant_ethnicity_name'],prefix = 'ethnicity')
loan_df_new = pd.concat([loan_df_new,ethnicity_dummy],axis=1)
loan_df_new.shape


# In[11]:


unique_race = loan_df_new['applicant_race_name_1'].value_counts()
print(unique_race)

race_dummy = pd.get_dummies(loan_df_new['applicant_race_name_1'],prefix = 'race')
loan_df_new = pd.concat([loan_df_new,race_dummy],axis=1)
loan_df_new.shape


# In[12]:


unique_sex = loan_df_new['applicant_sex_name'].value_counts()
print(unique_sex)

sex_dummy = pd.get_dummies(loan_df_new['applicant_sex_name'],prefix = 'sex')
loan_df_new = pd.concat([loan_df_new,sex_dummy],axis=1)
loan_df_new.shape


# In[13]:


unique_coethnicity = loan_df_new['co_applicant_ethnicity_name'].value_counts()
print(unique_coethnicity)

coethnicity_dummy = pd.get_dummies(loan_df_new['co_applicant_ethnicity_name'],prefix = 'coethnicity')
loan_df_new = pd.concat([loan_df_new,coethnicity_dummy],axis=1)
loan_df_new.shape


# In[14]:


unique_corace = loan_df_new['co_applicant_race_name_1'].value_counts()
print(unique_corace)

corace_dummy = pd.get_dummies(loan_df_new['co_applicant_race_name_1'],prefix = 'corace')
loan_df_new = pd.concat([loan_df_new,corace_dummy],axis=1)
loan_df_new.shape


# In[15]:


unique_cosex = loan_df_new['co_applicant_sex_name'].value_counts()
print(unique_cosex)

cosex_dummy = pd.get_dummies(loan_df_new['co_applicant_sex_name'],prefix = 'cosex')
loan_df_new = pd.concat([loan_df_new,cosex_dummy],axis=1)
loan_df_new.shape


# In[16]:


unique_county = loan_df_new['county_name'].value_counts()
print(unique_county)

county_dummy = pd.get_dummies(loan_df_new['county_name'],prefix = 'county')
loan_df_new = pd.concat([loan_df_new,county_dummy],axis=1)
loan_df_new.shape


# In[17]:


unique_hoepa = loan_df_new['hoepa_status_name'].value_counts()
print(unique_hoepa)

hoepa_dummy = pd.get_dummies(loan_df_new['hoepa_status_name'],prefix = 'hoepa')
loan_df_new = pd.concat([loan_df_new,hoepa_dummy],axis=1)
loan_df_new.shape


# In[18]:


unique_purpose = loan_df_new['loan_purpose_name'].value_counts()
print(unique_purpose)

purpose_dummy = pd.get_dummies(loan_df_new['loan_purpose_name'],prefix = 'purpose')
loan_df_new = pd.concat([loan_df_new,purpose_dummy],axis=1)
loan_df_new.shape


# In[19]:


unique_type = loan_df_new['loan_type_name'].value_counts()
print(unique_type)

type_dummy = pd.get_dummies(loan_df_new['loan_type_name'],prefix = 'type')
loan_df_new = pd.concat([loan_df_new,type_dummy],axis=1)
loan_df_new.shape


# In[20]:


unique_msamd = loan_df_new['msamd_name'].value_counts()
print(unique_msamd)

msamd_dummy = pd.get_dummies(loan_df_new['msamd_name'],prefix = 'msamd')
loan_df_new = pd.concat([loan_df_new,msamd_dummy],axis=1)
loan_df_new.shape


# In[21]:


unique_preapp = loan_df_new['preapproval_name'].value_counts()
print(unique_preapp)

preapp_dummy = pd.get_dummies(loan_df_new['preapproval_name'],prefix = 'preapp')
loan_df_new = pd.concat([loan_df_new,preapp_dummy],axis=1)
loan_df_new.shape


# In[22]:


unique_prop = loan_df_new['property_type_name'].value_counts()
print(unique_prop)

prop_dummy = pd.get_dummies(loan_df_new['property_type_name'],prefix = 'prop')
loan_df_new = pd.concat([loan_df_new,prop_dummy],axis=1)
loan_df_new.shape


# In[23]:


unique_purchase = loan_df_new['purchaser_type_name'].value_counts()
print(unique_purchase)

purchase_dummy = pd.get_dummies(loan_df_new['purchaser_type_name'],prefix = 'purchase')
loan_df_new = pd.concat([loan_df_new,purchase_dummy],axis=1)
loan_df_new.shape


# In[24]:


#drop the original categorical columns

loan_df_fin = loan_df_new.drop(['agency_name','applicant_ethnicity_name','applicant_race_name_1',
                               'applicant_sex_name','co_applicant_ethnicity_name','co_applicant_race_name_1',
                               'co_applicant_sex_name','county_name','hoepa_status_name','loan_purpose_name',
                               'loan_type_name','msamd_name','preapproval_name','property_type_name',
                               'purchaser_type_name'],1)
loan_df_fin.shape


# In[25]:


#imputing missing values in columns with mean values

null = loan_df_fin['applicant_income_000s'].isnull().sum()
loan_df_fin['applicant_income_000s'] = loan_df_fin['applicant_income_000s'].fillna(loan_df_fin.applicant_income_000s.mean())


# In[26]:


null = loan_df_fin['census_tract_number'].isnull().sum()
loan_df_fin['census_tract_number'] = loan_df_fin['census_tract_number'].fillna(loan_df_fin.census_tract_number.mean())
loan_df_fin = loan_df_fin.dropna()


# In[27]:


#shuffling rows for uniform distrubution

from sklearn.utils import shuffle

loan_df_fin = shuffle(loan_df_fin)


# In[28]:


unique_action = loan_df_new['action_taken_name'].value_counts()
print(unique_action)


# In[29]:


# Removing the last class as it has only one row
loan_df_new = loan_df_new[(loan_df_new[['action_taken_name']] != 'Preapproval request approved but not accepted').all(axis=1)]


# In[30]:


# Creating a new dataframe for target variable
target_df = pd.DataFrame(loan_df_fin['action_taken_name'])
loan_df_fin = loan_df_fin.drop(['action_taken_name'],1)


# In[31]:


loan_df_fin.shape


# In[32]:


# Perform test train split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(loan_df_fin, target_df, test_size=0.2)


# In[34]:


conn = sqlite3.connect("mortgage.db")

X_train.to_sql("X_train", conn, if_exists="replace")
X_test.to_sql("X_test", conn, if_exists="replace")

Y_train.to_sql("Y_train", conn, if_exists="replace")
Y_test.to_sql("Y_test", conn, if_exists="replace")

