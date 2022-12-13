# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
from sklearn.preprocessing import PolynomialFeatures 


loaded_model= pickle.load(open('D:/SHANTANU/InternFastFind/Mini Project 3 -Child Mortality Web App/model.sav','rb'))

poly = PolynomialFeatures(degree=2)
years=10
print(f'Prediction -Morality Rate Of World {2020.5+years} will be:',end=' ')
print(loaded_model.predict(poly.fit_transform([[2020.5+years]])))