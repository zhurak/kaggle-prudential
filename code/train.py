# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:47:00 2016

@author: bohdan
"""

import os

if __name__ == '__main__':
    
    execfile('logRegression.py')
    print 'Logistic Regression is done'

    execfile('randomForest.py')
    print 'Random Forest Classifier is done'

    execfile('xgboost_binary.py')
    print 'Xgboost binary is done'

    execfile('xgboost_multisoft.py')
    print 'Xgboost multisoft is done'
    

    