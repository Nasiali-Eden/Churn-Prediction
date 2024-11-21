# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Load Dataset
# Replace 'telco_customer_churn.csv' with the path to your dataset
df = pd.read_csv('telco_customer_churn.csv')

# Step 2: Data Preprocessing
# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Encode the target variable (Churn: Yes=1, No=0)
label_encoder = LabelEncoder()
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
df[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(df[['MonthlyCharges', 'TotalCharges', 'tenure']])

# Define features and target
X = df.drop(columns=['customerID', 'Churn'])  # Features
y = df['Churn']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 4: Evaluate Model
# Predictions and evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 5: User Input for Prediction
def get_user_input():
    print("\nEnter customer details for churn prediction:")
    tenure = float(input("Tenure (number of months): "))
    monthly_charges = float(input("Monthly Charges: "))
    total_charges = float(input("Total Charges: "))
    internet_service = input("Internet Service (options: DSL, Fiber optic, No): ")
    contract = input("Contract Type (options: Month-to-month, One year, Two year): ")
    payment_method = input("Payment Method (options: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)): ")

    # Map inputs to the processed features
    input_data = {
        'tenure': (tenure - df['tenure'].mean()) / df['tenure'].std(),  # Scale input
        'MonthlyCharges': (monthly_charges - df['MonthlyCharges'].mean()) / df['MonthlyCharges'].std(),  # Scale input
        'TotalCharges': (total_charges - df['TotalCharges'].mean()) / df['TotalCharges'].std(),  # Scale input
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
    }

    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data], columns=X.columns)
    return input_df

# Get user input
user_input = get_user_input()

# Predict churn for user input
prediction = rf_model.predict(user_input)
prediction_proba = rf_model.predict_proba(user_input)[:, 1]

# Display prediction result
print("\nPrediction Results:")
if prediction[0] == 1:
    print(f"The customer is likely to churn. (Confidence: {prediction_proba[0]*100:.2f}%)")
else:
    print(f"The customer is not likely to churn. (Confidence: {100 - prediction_proba[0]*100:.2f}%)")
