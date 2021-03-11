import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

import joblib

def run_ml_app():

    st.subheader('머신러닝 화면입니다.')

    # 1.유저한테 입력을 받는다.
    # 성별
    gender=st.radio('성별을 선택하세요',['남자','여자'])
    if gender=='남자':
        gender=1
    else:
        gender=0
    
    age=st.number_input('나이 입력',min_value=0,max_value=120)
    salary=st.number_input('연봉 입력',min_value=0)
    dept=st.number_input('빚 입력',min_value=0)
    worth=st.number_input('자산 입력',min_value=0)

    # 예측한다

    # 1.모델을 불러온다
    model=tensorflow.keras.models.load_model('data/car_ai.h5')
    # 2.넘파이 어레이 만든다.
    new_data=np.array([gender,age,salary,dept,worth])
    # 3.피처스케일링 한다.
    new_data=new_data.reshape(1,-1)
    
    sc_X=joblib.load('data/sc_x.pkl')
    
    new_data=sc_X.transform(new_data)
    # 예측한다
    y_pred=model.predict(new_data)
    
    # 예측결과는 스케일링 된 결과이므로 다시 돌려야한다
    sc_y=joblib.load('data/sc_y.pkl')

    y_pred_original=sc_y.inverse_transform(y_pred)

    btn=st.button('결과 보기')
    if btn:
        # st.write('예측된 결과입니다. {:,.1f} 달러의 차를 살수 있습니다.'.format(y_pred_original[0,0]))
        st.write('예측된 결과입니다. %.1f 달러의 차를 살수 있습니다.' % y_pred_original[0][0])



   

  


    
    

    


  
