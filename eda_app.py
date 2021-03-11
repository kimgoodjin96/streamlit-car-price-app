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

def run_eda_app():
    st.subheader('EDA 화면입니다.')
    car_df = pd.read_csv('data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')

    radio_menu=['데이터프레임','통계치']
    selected_radio=st.radio('선택하세요',radio_menu)
    
    if selected_radio=='데이터프레임':
        st.dataframe(car_df)
    
    elif selected_radio=='통계치':
        st.dataframe(car_df.describe())


    car_df = pd.read_csv('data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')
    columns=car_df.columns
    columns=list(columns)

    selected_columns=st.multiselect('보고싶은 컬럼을 선택하시오',columns)
    
    if len(selected_columns)!=0:
        st.dataframe(car_df[selected_columns])
    else:
        st.write('선택한 컬럼이 없습니다')
    
    
    # 상관계수를 화면에 보여주도록 만듭니다.
    # 멀티셀렉트에 컬럼명을 보여주고,
    # 해당 컬럼들에 대한 상관계수를 보여주세요.
    # 단,컬럼들은 숫자 컬럼들만 멀티셀렉트에 나타나야합니다.

    car_corr=car_df.columns[car_df.dtypes!=object]  

    selected_corr=st.multiselect('상관계수를 보고싶은 컬럼을 선택하세요',car_corr)
    
    if len(selected_corr)!=0:
        st.dataframe(car_df[selected_corr].corr())
        
        st.pyplot(sns.pairplot(car_df[selected_corr]))
    #  위에서 선택한 컬럼들을 이용해서 시본의 페어플롯을 그린다.
    

    else:
        st.write('선택한 컬럼이 없습니다')

    
    # 컬럼을 하나만 선택하면 ,해당 컬럼의 min 과 max에 
    # 해당하는 사람의 데이터를 화면에 보여주는 기능

    car_minmax=car_df.columns[car_df.dtypes!=object]  

    selected_minmax=st.selectbox('컬럼을 선택해주세요',car_minmax)

  
    st.write('Max 값에 해당되는 사람의 데이터')
    Max_customer_data=car_df.loc[car_df[selected_minmax]==car_df[selected_minmax].max(),]
    st.dataframe(Max_customer_data)

    st.write('Min 값에 해당되는 사람의 데이터')
    Max_customer_data=car_df.loc[car_df[selected_minmax]==car_df[selected_minmax].min(),]
    st.dataframe(Max_customer_data)
    

    # 고객이름을 검색할수 있는 기능 개발
    search=st.text_input('이름을 입력하세요').lower()
    # search_data=car_df['Customer Name'].str.lower().to_frame()['Customer Name'].str.contains(search)
    search_data=car_df['Customer Name'].str.contains(search,case=False)
    if len(search)!=0:
        st.dataframe(car_df.loc[search_data,])
    else:
        st.write('이름을 입력하세요')
        

    


