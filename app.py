import streamlit as st
import joblib

# Load the SVM model
model = joblib.load('models\diabetes_model.joblib')

# Function to predict diabetes
def predict_diabetes(input_features):
    prediction = model.predict([input_features])
    return prediction[0]

# Main function
def main():
    st.markdown("## Group 3 Project:\nDiabetes Prediction")
    

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
        if prediction == 1:
            st.write('You may have diabetes.')
        else:
            st.write('You may not have diabetes.')

if __name__ == '__main__':
    main()
