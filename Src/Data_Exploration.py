
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sqlite3 
 


# In[2]:


mortgage_df = pd.read_csv("https://api.consumerfinance.gov:443/data/hmda/slice/hmda_lar.csv?$where=as_of_year%3D2017+and+property_type+in+(1%2C2)+and+owner_occupancy%3D1+and+lien_status%3D1+and+state_abbr%3D'CA'+and+county_name+in+('San+Francisco+County'%2C'Santa+Clara+County'%2C'Sacramento+County')&$limit=10000000&$offset=0")


# In[3]:


mortgage_df.shape


# In[4]:


# Creating concise table to understand the datatype and missing values for each column

data_type = mortgage_df.dtypes.values
    
number_of_missing_values = mortgage_df.isnull().sum().values

total_values = mortgage_df.shape[0]
      
concise ={'Attributes': mortgage_df.columns,
          'Data_Type': data_type,
          'No_of_Missing_values': number_of_missing_values,
          'Total_Rows': total_values }

column_names = {'Attributes','Data_Type','No_of_Missing_values', 'Total_Rows'}

concise_df = pd.DataFrame(concise, columns = column_names)
concise_df


# In[5]:


# Description for each numeric column
mortgage_df.describe()


# In[6]:


# Percentage for missing data in each column
number_of_missing_values = mortgage_df.isnull().sum()
percentage_missing_values = (mortgage_df.isnull().sum()/ mortgage_df.shape[0])*100

Missing_df = pd.concat([number_of_missing_values, percentage_missing_values], axis=1, 
                         keys=['Number_Of_Missing_Values', 'Percentage_Of_Missing_Values'])
Missing_df


# In[7]:


unique_count = []
for attr in mortgage_df.columns:
    unique_count.append(mortgage_df[attr].unique().shape[0])

print(unique_count)


# In[8]:


#Dropping columns with more than 50% null values
#Dropped 27 columns from 78 
#Current total is 51 columns

mortgage_df = mortgage_df.drop(['applicant_race_2','applicant_race_3','applicant_race_4','applicant_race_5',
                                'applicant_race_name_2','applicant_race_name_3','applicant_race_name_4','applicant_race_name_5',
                                'application_date_indicator','co_applicant_race_2','co_applicant_race_3','co_applicant_race_4',
                                'co_applicant_race_5','co_applicant_race_name_2','co_applicant_race_name_3','co_applicant_race_name_4',
                                'co_applicant_race_name_5','denial_reason_1','denial_reason_2','denial_reason_3',
                                'denial_reason_name_1','denial_reason_name_2','denial_reason_name_3','edit_status',
                                'edit_status_name','sequence_number','rate_spread'],1)


# In[34]:


mortgage_df.shape


# In[35]:


#Information on target variable
target_unique = mortgage_df['action_taken_name'].value_counts()


# In[36]:


count = target_unique.tolist()
label = mortgage_df.action_taken_name.unique().tolist()


# In[37]:


df = pd.DataFrame({'Class': label, 'count': count})
ax = df.plot.barh(x='Class', y='count',color = 'green')
plt.savefig('/Users/vineevineela/Downloads/220_Assignment/Images/target.svg',bbox_inches = "tight")
plt.clf()


# In[38]:


conn = sqlite3.connect("mortgage.db")
mortgage_df.to_sql("loan_data", conn, if_exists="replace")

