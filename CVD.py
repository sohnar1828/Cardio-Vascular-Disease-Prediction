import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import streamlit as st

def main():
    st.title("Cardio Vascular Diseas Prediction")
    st.sidebar.title("Parameters")
    st.sidebar.markdown("Patient Detials")
    
    st.sidebar.subheader("Age (in Years)")
    age = st.sidebar.number_input("Enter Age",min_value=None, max_value=140)

    st.sidebar.subheader("Gender")
    st.sidebar.subheader("|1: Male, 2:Female|")
    gender =  st.sidebar.selectbox("Enter gender",(1,2))
        
    st.sidebar.subheader("Height (in cm)")
    height = st.sidebar.number_input("Enter height",min_value=50, max_value=300)
    
    st.sidebar.subheader("Weight (in kg)")
    weight = st.sidebar.number_input("Enter weight",min_value=10, max_value=250)
    
    st.sidebar.subheader("Systolic blood pressure")
    ap_hi = st.sidebar.number_input("Enter Systolic Blood Pressure")
    
    st.sidebar.subheader("Diastolic blood pressure")
    ap_lo = st.sidebar.number_input("Enter Diastolic Blood Pressure")

    st.sidebar.subheader("Cholestrol")
    st.sidebar.subheader("| 1: normal, 2: above normal, 3: well above normal |")
    cholesterol = st.sidebar.selectbox("Choose Cholestrol level",(1, 2, 3))
    
    st.sidebar.subheader("Glucose")
    st.sidebar.subheader("| 1: normal, 2: above normal, 3: well above normal |")
    gluc = st.sidebar.selectbox("Choose Glucose Levels",(1, 2, 3))
    
    st.sidebar.subheader("Active Smoker")
    st.sidebar.subheader("| 1: Yes, 0: No |")
    smoke = st.sidebar.selectbox("Choose Habits",(1, 0))

    st.sidebar.subheader("Alcohol consumption")
    st.sidebar.subheader("| 1: Yes, 0: No |")
    alco = st.sidebar.selectbox("Do you consume Alcohol?",(1, 0))

    st.sidebar.subheader("Physical Activity")
    st.sidebar.subheader("| 1: Yes, 0: No |")
    active = st.sidebar.selectbox( ("Do you excersie or play sport"),(1,0))
    
    return age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active

if __name__ == '__main__':
    age, gender, height, weight, ap_hi, ap_lo,  cholesterol, gluc, smoke, alco, active = main()

    cholestrol_dict = {'cholesterol_1': False, 'cholesterol_2': False, 'cholesterol_3': False}
    cholestrol_dict['cholesterol_' + str(cholesterol)] = True

    gluc_dict = {'gluc_1': False, 'gluc_2': False, 'gluc_3': False}
    gluc_dict['gluc_' + str(gluc)] = True

    smoke_dict = {'smoke_0': False, 'smoke_1': False}
    smoke_dict['smoke_' + str(smoke)] = True

    alco_dict = {'alco_0': False, 'alco_1': False}
    alco_dict['alco_' + str(alco)] = True

    active_dict = {'active_0': False, 'active_1': False}
    active_dict['active_' + str(active)] = True
    
    patient_data = {'age': age, 'gender': gender, 'ap_hi':ap_hi, 'ap_lo': ap_lo, 'BMI': weight/((height/100)**2)}
    patient_data.update(cholestrol_dict)
    patient_data.update(gluc_dict)
    patient_data.update(smoke_dict)
    patient_data.update(alco_dict)
    patient_data.update(active_dict)

    
    pat_det = pd.DataFrame([patient_data])
    
    # print(pat_det.info())
    
    if st.button('Predict'):
    
        loaded_rf = joblib.load("./random_forest.joblib")
        output = loaded_rf.predict(pat_det)
        op = output[0]
        if op == 1 :st.write('There is a high probablity of You having cardio vascular Ailment')
        else: ('There is a lowprobablity of You having cardio vascular Ailment')

    
