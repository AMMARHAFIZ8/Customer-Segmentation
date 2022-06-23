# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:49:07 2022

@author: ACER
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input

from  sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping 
import datetime

#cramers corrected stat
import scipy.stats as ss
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

from sklearn.model_selection import train_test_split


#%%
CSV_PATH = os.path.join(os.getcwd(),'train.csv')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'log',log_dir)
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
#%% EDA

# Step 1 Data Loading
df = pd.read_csv(CSV_PATH)
df_copy = df.copy()
# Step 2 Data Inspection
df.head(10)
df.tail(10)

df.info() #to check if theres Nan 
df.describe().T ## to check percentile,mean, min-max, count
df.duplicated().sum() # check for total duplicated data
df[df.duplicated()] # check for total duplicated values
df.isnull().sum()
df.columns #Get column names
df.boxplot() # to check summary of a set of data


# to visualize your data
#to see number each cat. in dataset
# categorical data
categorical=['job_type','marital','education','default','housing_loan','personal_loan','communication_type',
             'prev_campaign_outcome','term_deposit_subscribed']
for cat in categorical:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()
    
# continuous data
#to see the distribution in the dataset
continuous=['customer_age','balance','day_of_month','last_contact_duration','num_contacts_in_campaign',
            'num_contacts_prev_campaign']
for con in continuous:
    plt.figure()
    sns.distplot(df[con])
    plt.show()


# Step 3 Data cleaning

df = df.drop_duplicates()
df.info() # all duplicated has been removed

msno.matrix(df) #to visualize Nans in the data
msno.bar(df) #to visualize Nans in the data

df = df.drop(['days_since_prev_campaign_contact','id'], axis=1)

column_names = ['job_type','marital','education','housing_loan','personal_loan','communication_type',
             'prev_campaign_outcome', 'customer_age','balance','day_of_month','last_contact_duration','default',
                         'num_contacts_prev_campaign','num_contacts_in_campaign','term_deposit_subscribed']

df['job_type'] = df['job_type'].fillna(df['job_type'].mode())
df['marital'] = df['marital'].fillna(df['marital'].mode())
df['education'] = df['education'].fillna(df['education'].mode())
df['housing_loan'] = df['housing_loan'].fillna(df['housing_loan'].mode())
df['personal_loan'] = df['personal_loan'].fillna(df['personal_loan'].mode())
df['communication_type'] = df['communication_type'].fillna(df['communication_type'].mode())
df['prev_campaign_outcome'] = df['prev_campaign_outcome'].fillna(df['prev_campaign_outcome'].mode())
df['customer_age'] = df['customer_age'].fillna(df['customer_age'].median())
df['balance'] = df['balance'].fillna(df['balance'].median())
df['day_of_month'] = df['day_of_month'].fillna(df['day_of_month'].median())
df['last_contact_duration'] = df['last_contact_duration'].fillna(df['last_contact_duration'].median())
df['default'] = df['default'].fillna(df['default'].mode())
df['num_contacts_prev_campaign'] = df['num_contacts_prev_campaign'].fillna(df['num_contacts_prev_campaign'].median())
df['num_contacts_in_campaign'] = df['num_contacts_in_campaign'].fillna(df['num_contacts_in_campaign'].median())
df['term_deposit_subscribed'] = df['term_deposit_subscribed'].fillna(df['term_deposit_subscribed'].mode())

df.info()
df.describe().T
msno.matrix(df) #to visualize Nans in the data
msno.bar(df) #to visualize Nans in the data

#Categorical vs Categorical data using Cramer's V
for cat in categorical:
    print(cat)
    confussion_mat = pd.crosstab(df[cat], df['term_deposit_subscribed']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))
    
#Step 4) Features Selection


# To change string data into numeric
le= LabelEncoder()

for i in categorical:    
    df[i]= le.fit_transform(df[i])


# Regression analysis
# continuous data

#Features selection using corr
continuous_data = df.loc[:,continuous]
continuous_data.corr
corr = continuous_data.corr()

plt.figure()
sns.heatmap(corr,annot=True,cmap='Reds')
plt.show()



# continuous vs categorical data

for con in continuous:
    lr= LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=1),df['term_deposit_subscribed'])
    print(con + ' '+ str(lr.score(np.expand_dims(df[con],axis=1),df['term_deposit_subscribed'])))

#select features
#from above analysis
#keep last_contact_duration job_type marital

X= df.loc[:,['communication_type','job_type','housing_loan','prev_campaign_outcome']]
y= df['term_deposit_subscribed']


#Step 5) Data pre-processing

# #OneHotEncoding for target
ohe = OneHotEncoder(sparse=False)
y= ohe.fit_transform(np.expand_dims(y,axis=-1))

OHE_PATH = os.path.join(os.getcwd(), 'OHE.pkl')
with open(OHE_PATH, 'wb') as file:
    pickle.dump(ohe, file)
# mms = MinMaxScaler()
# df = mms.fit_transform(np.expand_dims(df['term_deposit_subscribed'], axis=-1))

#Standard scaling for features

# std = StandardScaler()
# scaled_X= ss.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, 
                                                    random_state=123)


#%% model development

# # Machine Learning

# def simple_two_layer_model(nb_class,nb_features,drop_rate=0.2,num_node=32):
# #Sequential Approach



nb_features = np.shape(X)[1:]
nb_class =len(np.unique(y_train,axis=0))


drop_rate = 0.2
num_node=32

model =  Sequential() # To create a container
model.add(Input(shape=nb_features))
model.add(Dense(num_node,activation='linear',name='Hidden_Layer1'))
model.add(BatchNormalization())
model.add(Dropout(drop_rate))
model.add(Dense(num_node,activation='linear',name='Hidden_Layer2'))
model.add(BatchNormalization())
model.add(Dropout(drop_rate))
# model.add(Dense(num_node,activation='relu',name='Hidden_Layer3'))
# model.add(BatchNormalization())
# model.add(Dropout(drop_rate))
model.add(Dense(nb_class,activation='softmax',name='Output_Layer'))
model.summary()


#wrapping container
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 


tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=5)

# X_train = np.expand_dims(X_train,axis=-1)
hist = model.fit(x=X_train,y=y_train,batch_size=128,epochs=10,
                      validation_data=(X_test,y_test),
                       callbacks=[tensorboard_callback,early_stopping_callback])



hist.history.keys()
training_loss = hist.history['loss']
training_acc = hist.history['acc']
validation_acc = hist.history['val_acc']
validation_loss = hist.history['val_loss']


plt.figure()
plt.plot(training_loss)
plt.plot(validation_loss)

plt.legend(['train_loss','val_loss'])
plt.show()

plt.figure()
plt.plot(training_acc)
plt.plot(validation_acc)

plt.legend(['train_acc','val_acc'])
plt.show()

#%% model evaluation

results = model.evaluate(X_test,y_test)
print(results) # loss and acc metrics

# pred_y = classifier.predict(x_test) # to get pred_y and compared with

# temp_x = np.expand_dims(x_test[0],axis=0)
# temp_y = y_test[0]

# pred_y = np.argmax(classifier.predict(temp_x))
# true_y = np.argmax(y_test[0])

pred_y = np.argmax(model.predict(X_test),axis=1)
true_y = np.argmax(y_test,axis=1)


cm = confusion_matrix(true_y, pred_y)
cr = classification_report(true_y, pred_y)
print(cm)
print(cr)

label=['0','1']
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label)
disp.plot(cmap=plt.cm.Reds)
plt.show()

#%% Model Architecture

plot_model(model,show_shapes=True, show_layer_names=(True))

#%% model saving

model.save(MODEL_SAVE_PATH)














