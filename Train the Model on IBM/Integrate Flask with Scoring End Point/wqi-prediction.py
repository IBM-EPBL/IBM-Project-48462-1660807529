#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import probplot
from scipy.stats import zscore


# In[5]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='1k6vPDym7SOUGXZ5WFFm05ZdwWtL8Jg1qrMs95vhHyIa',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'custommodeldeployment-donotdelete-pr-azgkeuzvaom7wf'
object_key = 'water_data.csv'
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='y47B_bAa3OuwbnH7yVZpORhrz9tUovsg3rqfVny54v1X',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.ap.cloud-object-storage.appdomain.cloud')

bucket = 'hydropureefficientwaterqualityana-donotdelete-pr-ueaw6nlihayuev'
object_key = 'water_data.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)
df.head()


# In[6]:


# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
metadata_1 = {
    'IAM_SERVICE_ID': 'iam-ServiceId-83ecb90b-6d9c-41ca-b595-cb9d7b88d1d5',
    'IBM_API_KEY_ID': '1k6vPDym7SOUGXZ5WFFm05ZdwWtL8Jg1qrMs95vhHyIa',
    'ENDPOINT': 'https://s3.private.us.cloud-object-storage.appdomain.cloud',
    'IBM_AUTH_ENDPOINT': 'https://iam.cloud.ibm.com/oidc/token',
    'BUCKET': 'custommodeldeployment-donotdelete-pr-azgkeuzvaom7wf',
    'FILE': 'water_dataX.csv'
}
# Checking for datatypes of the dataset
df.dtypes


# In[7]:


df.head()


# In[8]:


# Changing column names
df = df.rename(columns={"D.O. (mg/l)": "DO", "CONDUCTIVITY (µmhos/cm)": "Conductivity", "B.O.D. (mg/l)": "BOD", "NITRATENAN N+ NITRITENANN (mg/l)": "NI", "FECAL COLIFORM (MPN/100ml)": "Fec_col", "TOTAL COLIFORM (MPN/100ml)Mean": "Tot_col"})


# 

# In[9]:


# Converting object data type to numeric
def convert_to_numeric(df):
    num_col = df.shape[1]
    # Start from index 3
    for index in range(3, num_col):
        col_name = df.iloc[:, index].name
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df

df = convert_to_numeric(df)
df.dtypes


# In[10]:


# Replacing string NAN values with actual NAN value (np.nan)
def convert_to_nan(df):
    n_col = df.shape[1]
    for index in range(n_col):
        df.iloc[:, index]  = df.iloc[:, index].replace("NAN", np.nan)
    return df

df = convert_to_nan(df)


# In[11]:


# Checking for missing values
df.isnull().sum().sort_values()


# * Since I have 1900 training examples, I can replace each missing value with median of the column (to avoid outliers)

# In[12]:


# Replacing NULL values with median of column
# Selecting numeric data
df_num = df.select_dtypes(exclude="object")
df_num_col = df_num.columns
imputer = SimpleImputer(strategy="median")

df_num = imputer.fit_transform(df_num)
df_num = pd.DataFrame(df_num, columns=df_num_col)


# In[13]:


# Filling Categorical missing values
df_cat = df.select_dtypes(include="object")
df_cat.isnull().sum()


# In[14]:


# Here we can fill these values by obeserving other attributes
# Example - 
pd.set_option('mode.chained_assignment', None)
df_cat_copy = df_cat.copy()

df_cat_copy[df_cat_copy["STATION CODE"] == "1330"]
# Station Code with value 1330 will have Location - TAMBIRAPARANI which belongs in STATE - TAMIL NADU
# I can replace all the NAN occurences in STATE with TAMILNADU
df_cat_copy["STATE"][df_cat_copy["STATION CODE"] == "1330"] = df_cat_copy["STATE"][df_cat_copy["STATION CODE"] == "1330"].fillna("TAMILNADU")

df_cat_copy[df_cat_copy["STATION CODE"] == "1330"]


# In[15]:


# The NULL values in LOCATION attribute are in STATE attribute (ex - above dataframe)
# So, I will fill the NULL values in LOCATION attribute with the corresponding STATE value
# And, the state value will be replaced by the function "fill_state" defined below this cell

def fill_locations(df_cat):
    location_null = df_cat[df_cat["LOCATIONS"].isnull()]
    location_null_indices = location_null.index
    for index in location_null_indices:
        state_value = location_null["STATE"][index]
        location_null["LOCATIONS"][index] = state_value
        location_null["STATE"][index] = np.nan
    df_cat[df_cat["LOCATIONS"].isnull()] = location_null
    return

fill_locations(df_cat_copy)
df_cat_copy[df_cat_copy["STATION CODE"] == "1330"]


# In[16]:


df_cat_copy[df_cat_copy["LOCATIONS"] == "TAMBIRAPARANI AT ARUMUGANERI, TAMILNADU"]


# In[17]:


# Now I can fill NULL values of Station code with corresponding LOCATION value(ex - above dataframe)
# I will match the corresponding LOCATION value with other same LOCATION value and insert the value of station code

def fill_code(df_cat):
    station_null = df_cat[df_cat["STATION CODE"].isnull()]
    station_null_indices = station_null.index
    for index in station_null_indices:
        stat_code = np.nan
        location_index = station_null["LOCATIONS"][index]
        code_at_location = df_cat["STATION CODE"][df_cat["LOCATIONS"] == location_index]
        for index_code in code_at_location.index:
            if (code_at_location[index_code] != np.nan):
                stat_code = code_at_location[index_code]
                break
        station_null["STATION CODE"][index] = stat_code
    df_cat[df_cat["STATION CODE"].isnull()] = station_null
    return

fill_code(df_cat_copy)
df_cat_copy[df_cat_copy["LOCATIONS"] == "TAMBIRAPARANI AT ARUMUGANERI, TAMILNADU"]


# In[18]:


# Filling all state NAN values which have corresponding station code value
def fill_state(df_cat):
    station_code = df_cat["STATION CODE"].unique()
    for index in range(station_code.shape[0]):
        if (station_code[index] != np.nan):
            df_state = df_cat["STATE"][df_cat["STATION CODE"] == station_code[index]] 
            state_values = df_cat["STATE"][df_cat["STATION CODE"] == station_code[index]]
            state = np.nan
            for index_state in range(state_values.shape[0]):
                if (state_values.iloc[index_state] != np.nan):
                    state = state_values.iloc[index_state]
                    break
            df_state_fill = df_state.fillna(state) 
            df_cat["STATE"][df_cat["STATION CODE"] == station_code[index]] = df_state_fill
    return
fill_state(df_cat_copy)
df_cat_copy[df_cat_copy["STATION CODE"] == "1330"]


# In[19]:


df_cat_copy.isnull().sum()


# In[20]:


df_cat_copy[df_cat_copy["STATE"].isnull()]


# * The above are the missing values which are left and does not satisfy any condition
# * So, according to the location these values will be filled manually
# 

# In[21]:


# The first location KABBANI AT MUTHANKARA is in STATE Kerela
df_cat_copy["STATE"][1106] = "KERALA"
df_cat_copy["STATE"][1107] = "KERALA"
df_cat_copy["STATE"][1650] = "CHANDIGARH"
df_cat_copy["STATE"][1651] = "CHANDIGARH"
df_cat_copy["STATE"][1652] = "CHANDIGARH"
df_cat_copy["STATE"][1770] = "CHANDIGARH"
df_cat_copy["STATE"][1771] = "CHANDIGARH"
df_cat_copy["STATE"][1772] = "CHANDIGARH"
df_cat_copy["STATE"][1784] = "DAMAN & DIU"
df_cat_copy["STATE"][1785] = "DAMAN & DIU"
df_cat_copy["STATION CODE"][1784] = "0000" # I am setting this according to myself
df_cat_copy["STATION CODE"][1785] = "0000"


# In[22]:


df_cat = df_cat_copy
df_cat.isnull().sum()


# 

# In[23]:


df_num.isnull().sum()


# In[24]:


df_final = pd.concat([df_cat, df_num], axis=1)
df_final.isnull().sum()


# In[25]:


# These are the samples which don't contain any attribute
# The filled attributes are median of corresponding columns
# So it is best to remove them
df_null = df_final[(df_final["STATION CODE"].isnull()) & (df_final["LOCATIONS"].isnull()) & (df_final["STATE"].isnull())]
df_null_indices = df_null.index
df_final.drop(df_null_indices, axis=0, inplace=True)
df_null


# In[26]:


df_final.isnull().sum()


# In[27]:


df_final.shape


# * The final dataset contains 1898 samples and 12 attributes

# In[28]:


# PLotting PDFs of all the numeric attributes in the dataset

df_num_final = df_final.select_dtypes(exclude="object")

def plot_kde(df):
    n_col = df.shape[1]
    for index in range(n_col):
        col_index = df.iloc[:, index]
        fig, ax = plt.subplots(1,1, figsize=(7, 5))
        sns.kdeplot(data=df, x=col_index.name)
        
plot_kde(df_num_final)


# In[29]:


# Here, almost all kde plots are Gaussian Like 
# Using Z-Score Normalization to detect outliers

df_num_final_norm = zscore(df_num_final, axis=0)


def indices_of_greater_than_3(df_norm):
    indices_arr = []
    n_col = df_norm.shape[1]
    for index in range(n_col):
        col_index = df_norm.iloc[: ,index]
        greater_than_3 = df_norm[col_index > 3]
        greater_than_3_index = greater_than_3.index
        indices_arr.extend(greater_than_3_index)
    return indices_arr

indices_arr = indices_of_greater_than_3(df_num_final_norm)
print("Number of outliers using Z-Score method-",len(indices_arr))
df_final.iloc[indices_arr, :]


# In[30]:


df_final.drop(indices_arr, axis=0, inplace=True)
df_final.shape


# In[31]:


# KDE plots after removal of outliers
# plot_kde(df_final.select_dtypes(exclude="object"))


# ## Water Quality Index Calculation
# 
# I will be using "Weighted Arithmetic Water Quality Index Method" to calculate WQI of each water sample. The formula to calculate WQI is - 
# 
# **WQI = ∑ qi × wi / ∑ wi**
# 
# Here wi - Unit weight of ith parameter
# 
#      qi - Quality estimate scale of each parameter, it is calculated with the formula - 
# 
# **qi = 100 × ( Vi − VIdeal / Si − VIdeal )**
# 
# Here Vi - Measured value of ith parameter
# 
#      Videal - Ideal value of ith parameter in pure water
#      
#      Si - Standard value recommended for ith parameter
# 
# wi is calculated by the formula - 
# 
# **wi = K /  Si**
# 
# Here K is proportionality constant which is - 
# 
#  **K = 1 /  ∑ Si**

# ##### Standard value recommended for parameters (Si) - 
# 
# Dissolved oxygen, mg/L               10
# 
# pH                                   8.5
# 
# Conductivity, µS/cm                  1000
# 
# Biological oxygen demand, mg/L       5
# 
# Nitrate, mg/L                        45
# 
# Fecal coliform/100 mL                100
# 
# Total coliform/100 mL                1000

# ##### Values of unit weights for each parameter (wi) - 
# 
# Dissolved Oxygen                     0.2213
# 
# pH                                   0.2604
# 
# Conductivity                         0.0022
# 
# Biological Oxygen Demand             0.4426
# 
# Nitrate                              0.0492
# 
# Fecal Coliform                       0.0221
# 
# Total Coliform                       0.0022

# ##### Ideal value of parameters (Videal) - 
# 
# Dissolved Oxygen                     14.6
# 
# pH                                   7.0
# 
# Conductivity                         0
# 
# Biological Oxygen Demand             0
# 
# Nitrate                              0
# 
# Fecal Coliform                       0
# 
# Total Coliform                       0

# In[32]:


# Calculating Water Quality Index of each sample
df_num_final = df_final.select_dtypes(exclude="object")
# Dropping year and Temp attribute because they are not used for computing WQI
df_num_final.drop(["year", "Temp"], axis=1, inplace=True)

# Weight Vector(wi)
wi = np.array([0.2213, 0.2604, 0.0022, 0.4426, 0.0492, 0.0221, 0.0022])

# Standard values of parameters(si)
si = np.array([10, 8.5, 1000, 5, 45, 100, 1000])

# Ideal values of paramters(vIdeal)
vIdeal = np.array([14.6, 7, 0, 0, 0, 0, 0])

def calc_wqi(sample): 
    wqi_sample = 0
    num_col = 7
    for index in range(num_col):
        v_index = sample[index] # Obeserved value of sample at index
        v_index_ideal = vIdeal[index] # Ideal value of obeserved value
        w_index = wi[index] # weight of corresponding parameter of obeserved value
        std_index = si[index] # Standard value recommended for obeserved value
        q_index = (v_index - v_index_ideal) / (std_index - v_index_ideal)
        q_index = q_index * 100 # Final qi value of obeserved value
        wqi_sample += q_index*w_index
    return wqi_sample


# In[33]:


# Computing WQI for the whole dataset
def calc_wqi_for_df(df):
    wqi_arr = []
    for index in range(df.shape[0]):
        index_row = df.iloc[index, :]
        wqi_row = calc_wqi(index_row)
        wqi_arr.append(wqi_row)
    return wqi_arr


# In[34]:


wqi_arr = calc_wqi_for_df(df_num_final)
# Converting oridnary array to numpy array
wqi_arr = np.array(wqi_arr)
wqi_arr = np.reshape(wqi_arr, (-1, 1))

# Resetting index values of the dataframes
wqi_arr_df = pd.DataFrame(wqi_arr, columns=["WQI"]).reset_index()
df_final = df_final.reset_index()


# In[35]:


# Combining dataframe of WQI and dataframe of attributes 
df_wqi = pd.concat([df_final, pd.DataFrame(wqi_arr, columns=["WQI"])], axis=1)
df_wqi.drop("index", axis=1, inplace=True)
df_wqi.shape


# In[36]:


# These are samples with negative WQI
df_wqi[(df_wqi["WQI"] < 0)]


# In[37]:


# Removing the samples with negative WQI
df_neg_indices = df_wqi[(df_wqi["WQI"] < 0)].index
df_wqi.drop(df_neg_indices, axis=0, inplace=True)


# Now I will set level of water contamination based on WQI -
# 
#           WQI Range                   Classification
# 
#          Less than 25                  Excellent(3)
#             26–50                         Good(2)
#             51-75                         Poor(1)
#         Greater than 75                 Very Poor(0)

# In[38]:


df_wqi["WQI clf"] = df_wqi["WQI"].apply(lambda x: (4 if (x <= 25)
                                        else(3 if (26<=x<=50)
                                        else(2 if (51<=x<=75)
                                        else(2 if (76<=x<=100)
                                        else 0)))))


# In[39]:


df_wqi.tail()


# In[40]:


df_wqi.describe()


# In[41]:


# f,ax = plt.subplots(figsize=(11, 11))
# sns.heatmap(df_wqi.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[42]:


plot_kde(df_wqi.select_dtypes(exclude="object"))


# In[43]:


df_wqi.describe()


# In[44]:


features = list(df_wqi.columns)[3:11]
data_f = df_wqi[features]
data_f.describe()


# In[45]:


features = list(df_wqi.columns)[:]
data_cluster = df_wqi['WQI clf']
data_cluster.describe()


# In[46]:


# normalize data
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(data_f)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


# In[47]:


data_f


# In[48]:


# data_f.apply(lambda col: col.drop_duplicates().reset_index(drop=True))


# In[49]:


# data_f.describe()


# In[50]:


Y = data_cluster


# In[51]:


features = list(df_wqi.columns)[3:11]
X = df_wqi[features]
X.describe()
X.dtypes


# In[52]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=30)


# In[64]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
# model = LinearRegression()
# # model = LogisticRegression(solver='liblinear')
# model.fit(X_train,y_train)
# model.score(X_test,y_test)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
score = neigh.score(X_test,y_test)
print("score: ",score)
y_pred = neigh.predict(X_test)
print(classification_report(y_test,y_pred, zero_division=1))


# In[65]:


clf = RandomForestClassifier(n_estimators = 100) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
 
# metrics are used to find accuracy or error
from sklearn import metrics 
print()
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# In[66]:


from sklearn.ensemble import RandomForestClassifier
forest_reg = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=42)
forest_reg.fit(X_train,y_train)


# In[67]:


X_train[0]


# In[68]:


get_ipython().system('pip install ibm_watson_machine_learning')


# In[69]:


from ibm_watson_machine_learning import APIClient
wml_credentials = {
                    "url": "https://us-south.ml.cloud.ibm.com",
                    "apikey":"52-LcxwCL4xYpFsN3lS7NdbNGhgPHdk2rriIX2j2w9yQ"
                  }
client = APIClient(wml_credentials)


# In[70]:


def guide_from_space_name(client,space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']["name"] == space_name)['metadata']['id'])


# In[74]:


space_uid = guide_from_space_name(client,'models')
print("Space UID = "+ space_uid)


# In[77]:


client.set.default_space(space_uid)


# In[78]:


client.software_specifications.list()


# In[98]:


metadata = {
            client.spaces.ConfigurationMetaNames.NAME: 'water-quality-index-prediction',
            client.spaces.ConfigurationMetaNames.DESCRIPTION: 'For my models'
            }
space_details = wml_client.spaces.store(meta_props=metadata)
space_uid = wml_client.spaces.get_uid(space_details)
   
# set the default space
wml_client.set.default_space(space_uid)

# see available meta names for software specs
print('Available software specs configuration:', wml_client.software_specifications.ConfigurationMetaNames.get())
wml_client.software_specifications.list()

asset_id = 'undefined'
pe_asset_id = 'undefined'


# In[ ]:


software_spec_uid = client.software_specifications.get_uid_by_name("default_py3.7")
software_spec_uid
# client.software_specifications.list()


# In[95]:


model_details = client.repository.store_model(model=forest_reg,meta_props={
    client.repository.ModelMetaNames.NAME:"water-quality-index-prediction",
    client.repository.ModelMetaNames.TYPE:"scikit-learn_0.23",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid}
                                             )
model_id = client.repository.get_model_uid(model_details)


# In[82]:


model_id


# In[ ]:




