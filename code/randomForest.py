# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:08:48 2016

@author: bohdan
"""

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import json
from label_decoders import *

config = json.load(open('settings.json'))
train = pd.read_csv(config['train'])
test = pd.read_csv(config['test'])

# combine train and test
all_data = train.append(test)

# create any new variables    
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]
all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)
all_data.apply(lambda x: sum(x.isnull()),1)
all_data['countna'] = all_data.apply(lambda x: sum(x.isnull()),1)
all_data.fillna(-1, inplace=True)
all_data['Response'] = all_data['Response'].astype(int)

# split train and test
train_ohd = all_data[all_data['Response']>0].copy()
test_ohd = all_data[all_data['Response']<1].copy()

features=train_ohd.columns.tolist()
features = [x.replace('=','_') for x in features]
features = [x.replace('_','i') for x in features]
train_ohd.columns = features
features_t=test_ohd.columns.tolist()
features_t = [x.replace('=','i') for x in features_t]
features_t = [x.replace('_','i') for x in features_t]
test_ohd.columns = features_t

features.remove("Id")
features.remove("Response")

train_ohd['rf1'] = [0]*train_ohd.shape[0]
train_ohd['rf2'] = [0]*train_ohd.shape[0]
train_ohd['rf3'] = [0]*train_ohd.shape[0]
train_ohd['rf4'] = [0]*train_ohd.shape[0]
train_ohd['rf5'] = [0]*train_ohd.shape[0]
train_ohd['rf6'] = [0]*train_ohd.shape[0]
train_ohd['rf7'] = [0]*train_ohd.shape[0]
train_ohd['rf8'] = [0]*train_ohd.shape[0]
train_ohd['rf9'] = [0]*train_ohd.shape[0]
train_ohd['rf10'] = [0]*train_ohd.shape[0]
train_ohd['rf11'] = [0]*train_ohd.shape[0]
train_ohd['rf12'] = [0]*train_ohd.shape[0]
train_ohd['rf13'] = [0]*train_ohd.shape[0]


l = train_ohd.shape[0]
ind_list = [(range(0,l/10), filter(lambda x: x not in range(0,l/10), range(0,l))), 
            (range(l/10,l/10*2), filter(lambda x: x not in range(l/10,l/10*2), range(0,l))),
            (range(l/10*2,l/10*3), filter(lambda x: x not in range(l/10*2,l/10*3), range(0,l))),
            (range(l/10*3,l/10*4), filter(lambda x: x not in range(l/10*3,l/10*4), range(0,l))),
            (range(l/10*4,l/10*5), filter(lambda x: x not in range(l/10*4,l/10*5), range(0,l))),
            (range(l/10*5,l/10*6), filter(lambda x: x not in range(l/10*5,l/10*6), range(0,l))),
            (range(l/10*6,l/10*7), filter(lambda x: x not in range(l/10*6,l/10*7), range(0,l))),
            (range(l/10*7,l/10*8), filter(lambda x: x not in range(l/10*7,l/10*8), range(0,l))),
            (range(l/10*8,l/10*9), filter(lambda x: x not in range(l/10*8,l/10*9), range(0,l))),
            (range(l/10*9,l), filter(lambda x: x not in range(l/10*9,l), range(0,l)))]


ld = [labels_decoder1,labels_decoder2,labels_decoder3,labels_decoder4,labels_decoder5,labels_decoder6,labels_decoder7,
      labels_decoder8,labels_decoder9,labels_decoder10,labels_decoder11,labels_decoder12,labels_decoder13]


i = 0
for l in ld:
    i = i + 1    
    
    for j in range(10):
        
        X_1, X_2 = ind_list[j][1], ind_list[j][0]
        y_1, y_2 = train_ohd.iloc[X_1]['Response'], train_ohd.iloc[X_2]['Response']
        
        rf = RandomForestClassifier(n_estimators=500, random_state=1)
        rf.fit(train_ohd[features].iloc[X_1],l(y_1))
        train_ohd['rf%s' % (i)].iloc[X_2] = rf.predict_proba(train_ohd[features].iloc[X_2]).T[1]


train_ohd.to_csv(config['train_rf'],index=0)

y = train_ohd['Response']

i = 0
for l in ld:
    i = i + 1    

###1
    rf = RandomForestClassifier(n_estimators=500, random_state=1)
    rf.fit(train_ohd[features],l(y))
    test_ohd['rf%s' % (i)] = rf.predict_proba(test_ohd[features]).T[1]

test_ohd.to_csv(config['test_rf'],index=0)