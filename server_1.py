import streamlit as st 
import pickle
import pandas as pd

print('Successfully executed ')

model = pickle.load(open('model.pkl', 'rb'))



def main():
    #Setting Application title
    st.title('Telco Customer Churn Prediction App')
    html_temp = """
    <div style="background-color:Blue;padding:20px">
    <h2 style="color:white;text-align:center;">Streamlit Diabetes Predictor </h2>
    </div>
    """

    #Setting Application description
   
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    
     
   
    st.info("Input data below")
    #Based on our optimal features selection
    st.subheader("Demographic data")
    gender = st.selectbox('Gender:', ('Male', 'Female'))
    SeniorCitizen = st.selectbox('Senior Citizen:', (1, 0))
    Partner=st.selectbox('Partner:', ('Yes', 'No'))
    Dependents = st.selectbox('Dependent:', ('Yes', 'No'))
    st.subheader("Payment data")
    tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
    Contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
    PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
    MonthlyCharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
    TotalCharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)
    st.subheader("Services signed up for")
    MutlipleLines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
    PhoneService = st.selectbox('Phone Service:', ('Yes', 'No'))
    InternetService = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
    OnlineBackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
    DeviceProtection=st.selectbox("Does the customer have device protection",('Yes','No'))
    TechSupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
    StreamingTV = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
    StreamingMovies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))
    data = {
            'gender':gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner':Partner,
            'Dependents': Dependents,
            'tenure':tenure,
            
            'PhoneService': PhoneService,
            'MultipleLines': MutlipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection':DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract':Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod':PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
            }
    features_df = pd.DataFrame.from_dict([data])
    #st.markdown("<h3></h3>", unsafe_allow_html=True)
    #st.write('Overview of input is shown below')
    #st.markdown("<h3></h3>", unsafe_allow_html=True)
    #st.dataframe(features_df)
    #Preprocess inputs
    col_names = pickle.load(open('cat_col.pkl', 'rb'))
    print(col_names)
    for col in col_names:
        
        features_df[col]=features_df[col].astype('category')
        name=col+'.pkl'
        
        enc=pickle.load(open(name, 'rb'))
        
        features_df[col]=enc.transform(features_df[col])

    prediction = model.predict(features_df)
    if st.button('Predict'):
        if prediction == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')



   
if __name__ == '__main__':
        main()