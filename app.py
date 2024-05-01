import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import joblib

# Load the SVM model
model = joblib.load('\models\diabetes_model.joblib')

# Function to predict diabetes
def predict_diabetes(input_features):
    prediction = model.predict([input_features])
    return prediction[0]

# Main function
if __name__ == '__main__':
    st.markdown("## Group 3 Project:\nDiabetes Prediction")
    with st.sidebar:
        selected = option_menu('Diabetes Prediction App',
                          
                          ['Predict Diabetes',
                           'Our Prediction Records',
                           'About Us'],
                          icons=['heart','book','info'],
                          default_index=0)
        
    
    if selected =="Predict Diabetes":
        # Collect user inputs
        Pregnancies=st.text_input('Number of times pregnant',0,10,0)
        Glucose = st.text_input('Plasma glucose concentration at 2 Hours in an oral glucose tolerance test (GTIT)', 0, 200,1)
        BloodPressure=st.text_input('DiaStolic Blood Pressure (mm Hg)',50,180,2)
        SkinThickness=st.text_input('Triceps skin fold Thickness (mm)',0,122,3)
        Insulin=st.text_input('2-Hour Serum insulin (µh/ml)',0,846,4)
        BMI = st.text_input('BMI',0,60,5)
        DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function')
        Age=st.text_input('Age(years)',10 ,81,7)
        # Add other input fields as needed
        # Create a dictionary with user inputs
        input_data = {
            'Pregnancies' : Pregnancies,
            'Glucose': Glucose,
            'BloodPressure' : BloodPressure,
            'SkinThickness' : SkinThickness,
            'Insulin' : Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
            'Age' : Age
            # Add other input fields here
        }

        # Convert input data to a list for prediction
        input_features = [input_data[feature] for feature in input_data]

        # Predict
        if st.button('Predict'):
            prediction = predict_diabetes(input_features)
            f = open("user_records.txt", "a")
            f.write("\n")
            new_data = str([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction,Age,prediction])
            leng = len(new_data)
            f.write(new_data[1:leng-1]) 
            f.close()
            
            if prediction == 1:
                st.write('__You may have diabetes.__')
            else:
                st.write('__You may not have diabetes.__')

    if selected == "Our Prediction Records":
        st.markdown("<h3 style='text-align: center;'>PREDICTION RECORDS OF OUR PREVIOUS USERS</h1>", unsafe_allow_html=True)
        f = pd.read_csv("user_records.txt")
        st.table(f)
        st.markdown("____")
        st.write("All the records are stored only for academic and research purpose & will not be used for any other means.")
        
    if selected == "About Us":
        st.markdown("<h2 style='text-align: center;'>ABOUT</h2>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<p style='text-align: center;'>This is an academic project made by B.Tech Computer Science And Engineering 3rd year student.</p>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<h4 style='text-align: center;'>Developed and maintained by</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Ritabrata Dey</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>ritabratadey1296@gmail.com</p>", unsafe_allow_html=True)
        st.markdown("____")
