import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import streamlit as st
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns

def connect_mysql():
    conn = mysql.connector.connect(
        host="localhost", 
        user="root", 
        password="2004",  
        database="diabetes_db"
    )
    return conn

def store_user_info(conn, user_info):
    cursor = conn.cursor()
    query = "INSERT INTO user_details (name, age, gender, mobile, location) VALUES (%s, %s, %s, %s, %s)"
    cursor.execute(query, user_info)
    conn.commit()
    cursor.close()
    
def store_user_feedback(conn, feedback):
    cursor = conn.cursor()
    query = "INSERT INTO user_feedback (feedback) VALUES (%s)"
    cursor.execute(query, (feedback,))
    conn.commit()
    cursor.close()  

@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\SEM 5\ML\dataset CAT II.csv")
    return df

def split_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def home_page():
    st.title("Diabetes Risk Management using Machine Learning")
    
    st.image(r"D:\SEM 5\ML\CAT II\home.jpg", caption="Manage your diabetes risk effectively", use_column_width=True)

def introduction_page():
    st.title("Introduction to Diabetes")
    
    st.image(r"D:\SEM 5\ML\CAT II\Introduction.jpg", caption="Understanding Diabetes", use_column_width=True)
    
    st.write(""" 
    Diabetes is a chronic medical condition that affects how the body processes blood glucose (or blood sugar), the primary source of energy for cells. In diabetes, either the body does not produce enough insulin, a hormone responsible for regulating blood sugar, or the body cannot effectively use the insulin it produces. This leads to elevated levels of glucose in the blood, which can cause serious health problems over time, including damage to the eyes, kidneys, nerves, and heart.
    
    ### Types of Diabetes:
    1. **Type 1 Diabetes**: An autoimmune condition where the body’s immune system attacks the insulin-producing cells in the pancreas. People with Type 1 diabetes must take insulin daily.
    2. **Type 2 Diabetes**: The most common type, where the body either does not produce enough insulin or becomes resistant to it. It is often associated with lifestyle factors such as obesity and inactivity.
    3. **Gestational Diabetes**: Occurs during pregnancy and usually resolves after childbirth, but it increases the risk of developing Type 2 diabetes later in life.
    
    ### Common Symptoms of Diabetes:
    - Frequent urination (polyuria)
    - Excessive thirst (polydipsia)
    - Unexplained weight loss
    - Increased hunger (polyphagia)
    - Fatigue or feeling very tired
    - Blurred vision
    - Slow healing of wounds or sores
    - Tingling, numbness, or pain in the hands or feet
    - Recurrent infections (such as skin, gum, or bladder infections)
    
    Early detection and proper management are essential for controlling the condition and preventing complications.
    """)

def user_input_page():
    st.write("## Enter your details")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    mobile = st.text_input("Mobile Number")
    location = st.text_input("Location")
    
    if st.button("Submit"):
        if name and mobile:  
            conn = connect_mysql()
            user_info = (name, age, gender, mobile, location)
            store_user_info(conn, user_info)
            st.write("User information saved successfully!")
            conn.close() 
            st.session_state.user_info_submitted = True  
        else:
            st.warning("Please fill in your name and mobile number.")

def prediction_page():
    if not st.session_state.get("user_info_submitted", False):
        st.warning("Please provide your user information first.")
        return

    st.title("Predict Diabetes Risk")
    
    Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=99, value=20)
    Insulin = st.number_input('Insulin', min_value=0, max_value=846, value=100)
    BMI = st.number_input('BMI', min_value=0.0, max_value=67.1, value=32.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    Age = st.number_input('Age', min_value=1, max_value=100, value=30)
    
    if st.button("Predict"):
        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Using Naive Bayes for prediction
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        user_data = pd.DataFrame({
            'Pregnancies': [Pregnancies],
            'Glucose': [Glucose],
            'BloodPressure': [BloodPressure],
            'SkinThickness': [SkinThickness],
            'Insulin': [Insulin],
            'BMI': [BMI],
            'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
            'Age': [Age]
        })
        
        prediction = model.predict(user_data)
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        st.write(f"### Prediction Result: {result}")


def model_evaluation_page():
    if not st.session_state.get("user_info_submitted", False):
        st.warning("Please provide your user information first.")
        return

    st.title("Model Evaluation")
    
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "Bagging": BaggingClassifier(),
        "Boosting": AdaBoostClassifier()
    }
    
    evaluation_results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report["weighted avg"]["f1-score"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        
        evaluation_results[model_name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Predictions": y_pred  # Store predictions for confusion matrix
        }
    
    st.write("### Model Performance Metrics:")
    for model_name, metrics in evaluation_results.items():
        st.write(f"**{model_name}**")
        st.write(f"Accuracy: {metrics['Accuracy']:.2f}")
        st.write(f"F1 Score: {metrics['F1 Score']:.2f}")
        st.write(f"Precision: {metrics['Precision']:.2f}")
        st.write(f"Recall: {metrics['Recall']:.2f}")
        st.write("---")
    
    best_model = max(evaluation_results, key=lambda x: evaluation_results[x]["Accuracy"])
    st.write(f"### Best Model: {best_model}")
    st.write(f"Accuracy: {evaluation_results[best_model]['Accuracy']:.2f}")
    
    # Plot accuracy comparison
    model_names = list(evaluation_results.keys())
    accuracies = [metrics["Accuracy"] for metrics in evaluation_results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color='lightblue')
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    st.pyplot(plt.gcf())
    
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Heatmap")
    st.pyplot(plt.gcf())
    
    # Confusion Matrix
    st.write("### Confusion Matrix")
    confusion_mtx = confusion_matrix(y_test, evaluation_results[best_model]['Predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f"Confusion Matrix for {best_model}")
    st.pyplot(plt.gcf())

def feedback_page():
    if not st.session_state.get("user_info_submitted", False):
        st.warning("Please provide your user information first.")
        return

    st.write("## Provide your feedback")
    
    st.image(r"D:\SEM 5\ML\CAT II\feedback_image.jpg", caption="We value your feedback!", use_column_width=True)

    feedback = st.text_area("Enter your feedback")

    if st.button("Submit Feedback"):
        if feedback:
            conn = connect_mysql()
            store_user_feedback(conn, feedback)
            st.write("Thank you for your feedback!")
            conn.close()
        else:
            st.warning("Please enter your feedback before submitting.")


# Streamlit app routing logic
pages = {
    "Home": home_page,
    "Introduction": introduction_page,
    "User Input": user_input_page,
    "Prediction": prediction_page,
    "Model Evaluation": model_evaluation_page,
    "Feedback": feedback_page
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(pages.keys()))
pages[page]()
