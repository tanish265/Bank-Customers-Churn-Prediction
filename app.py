

import numpy as np
#import pickle
import joblib
import streamlit as st 

#pickle_in = open("churnXGB.pkl","rb")
#classifier=pickle.load(pickle_in)

# Load the model from the file 
XGB_from_joblib = joblib.load('xgb1.joblib.dat')  

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_bank_customer_churn(creditScore,age,tenure,balance,
                                numofprod,estimatedsalary,balancesalaryratio,
                                tenurebyage,CreditScoreGivenAge,HasCrCard,
                                IsActiveMember,gender,geography):
    
    inputl=[[creditScore,age,tenure,balance,
            numofprod,estimatedsalary,balancesalaryratio,
            tenurebyage,CreditScoreGivenAge,HasCrCard,
            IsActiveMember]]
    if(gender==0):
        inputl[0].append(1)
        inputl[0].append(0)
    else:
        inputl[0].append(0)
        inputl[0].append(1)
        
    if(geography==0):
        inputl[0].append(1)
        inputl[0].append(0)
        inputl[0].append(0)
        
    elif(geography==1):
        inputl[0].append(0)
        inputl[0].append(1)
        inputl[0].append(0)
     
    else:
        inputl[0].append(0)
        inputl[0].append(0)
        inputl[0].append(1)
        
    input2 = np.array(inputl).reshape((1,-1))
    prediction=XGB_from_joblib.predict(input2)
    print(prediction)
    return prediction



def main():
    st.title("")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Customer Churn Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    creditScore = st.number_input("CreditScore")
    
    age = st.number_input("Age",min_value=1)
    tenure = st.number_input("Tenure (in years)")
    balance = st.number_input("Balance")
    numofprod= st.number_input("Number of Products")
    estimatedsalary= st.number_input("Estimated Salary",min_value=0.1)
    balancesalaryratio=balance/estimatedsalary
    tenurebyage=tenure/age
    CreditScoreGivenAge=creditScore/age
    HasCrCard= st.number_input("Has Credit Card? 1 for yes,-1 for No")
    IsActiveMember=st.number_input("IsActiveMember? 1 for yes,-1 for No ")
    geography=st.number_input("Geography-- 0 for France,1 for Germany,2 for Spain")
    gender=st.number_input("Gender-- 0 for female,1 for male ")
    
    creditScore=(creditScore-350)/500
    age=(age-18)/74
    tenure=tenure/10
    balance=balance/250898
    numofprod=(numofprod-1)/3
    estimatedsalary=(estimatedsalary-11.58)/199980.42
    balancesalaryratio=balancesalaryratio/10614.65
    tenurebyage=tenurebyage/0.556
    CreditScoreGivenAge=(CreditScoreGivenAge-4.86)/42.03
    
    result=""
    if st.button("Predict"):
        result=predict_bank_customer_churn(creditScore,age,tenure,balance,
                                           numofprod,estimatedsalary,balancesalaryratio,
                                           tenurebyage,CreditScoreGivenAge,HasCrCard,
                                           IsActiveMember,gender,geography)
    st.success('The output is {}'.format(result))
    
    st.text("Output 0 means customer not exited")
    st.text("Output 1 means customer exited")
    
    if st.button("About"):
        st.text("By- Tanish Gupta")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()