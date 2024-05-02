import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Load the SVM model
model = joblib.load("models/diabetes_model.joblib")
diabetes_dataset = pd.read_csv('diabetes-pima-indian-dataset.csv')
X = diabetes_dataset.drop(columns='Outcome',axis = 1)
Y = diabetes_dataset['Outcome']
#data standardization
scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
X= standardized_data
Y = diabetes_dataset['Outcome']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.3, stratify=Y, random_state=6)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
test_data_accuracy = test_data_accuracy*100
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
                           'About Me'],
                          icons=['heart','book','info'],
                          default_index=0)
        
    
    if selected =="Predict Diabetes":
        # Collect user inputs
        Name = st.text_input('Enter your name:')
        Pregnancies=st.text_input('Number of times pregnant (Range : 0 - 17)',0,10,0)
        Glucose = st.text_input('Plasma glucose concentration at 2 Hours in an oral glucose tolerance test (GTIT) (Range : 0 - 199)', 0, 200,1)
        BloodPressure=st.text_input('DiaStolic Blood Pressure (mm Hg) (Range : 0 - 122)',50,180,2)
        SkinThickness=st.text_input('Triceps skin fold Thickness (mm) (Range : 0 - 99)',0,122,3)
        Insulin=st.text_input('2-Hour Serum insulin (µh/ml) (Range : 0 - 846)',0,846,4)
        BMI = st.text_input('BMI (Range : 0 - 67.1)',0,60,5)
        DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function (Range : 0.078 - 2.42)')
        Age=st.text_input('Age(years) (Range : 21 - 81)',10 ,81,21)
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
            new_data = str([Name,Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction,Age,prediction])
            leng = len(new_data)
            f.write(new_data[1:leng-1]) 
            f.close()
            
            if prediction == 1:
                st.write('__You may have diabetes.__')
                st.write('Accuracy:',round(test_data_accuracy,3),'%')
            else:
                st.write('__You may not have diabetes.__')
                st.write('Accuracy:',round(test_data_accuracy,3),'%')

    if selected == "Our Prediction Records":
        st.markdown("<h3 style='text-align: center;'>PREDICTION RECORDS OF OUR PREVIOUS USERS</h1>", unsafe_allow_html=True)
        f = pd.read_csv("user_records.txt")
        #st.table(f)
        st.table(f.style.set_table_attributes('style="width:100%;"'))
        st.markdown("____")
        st.write("All the records are stored only for academic and research purpose & will not be used for any other means.")
        
    if selected == "About Me":
        st.markdown("<h2 style='text-align: center;'>ABOUT</h2>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<p style='text-align: center;'>This is an academic project made by B.Tech Computer Science And Engineering 3rd year student.</p>", unsafe_allow_html=True)
        st.markdown("____")
        st.markdown("<h4 style='text-align: center;'>Developed and maintained by</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Ritabrata Dey</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>ritabratadey1296@gmail.com</p>", unsafe_allow_html=True)
        st.markdown("____")
