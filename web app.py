# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 00:52:39 2022

@author: savita
"""


import pickle
from sklearn.preprocessing import PolynomialFeatures 
import streamlit as st

loaded_model= pickle.load(open('D:/SHANTANU/InternFastFind/Mini Project 3 -Child Mortality Web App/model.sav','rb'))

# creating a function for prediction

def prediciton(years):
    poly = PolynomialFeatures(degree=2)
    print("Prediction -Morality Rate Of World will be:",end=' ')
    return(loaded_model.predict(poly.fit_transform([[2020.5+years]])))


def main():
    
    # giving the title for the web app
    st.title('Child Mortality Rate Web App')
    
    #getting the input variable

    years= st.number_input('Number of years after 2020.5')
    
    # code from prediction
    
    result=''
    
    #creating a button
    if st.button('Final Prediction'):
        result=prediciton(years)
        
    st.success(result)
    
    
if __name__=='__main__':
    main()

        
    