import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load the data from your local file
file_path = 'telco_customer_churn.csv'  # Update with the correct file path
data = pd.read_csv(file_path)

# Preprocess the data (handling missing values, encoding categorical variables)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Convert to numeric, forcing errors to NaN
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())  # Fixed inplace warning

# Encode categorical variables
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['SeniorCitizen'] = label_encoder.fit_transform(data['SeniorCitizen'])
data['Partner'] = label_encoder.fit_transform(data['Partner'])
data['Dependents'] = label_encoder.fit_transform(data['Dependents'])
data['PhoneService'] = label_encoder.fit_transform(data['PhoneService'])
data['MultipleLines'] = label_encoder.fit_transform(data['MultipleLines'])
data['InternetService'] = label_encoder.fit_transform(data['InternetService'])
data['OnlineSecurity'] = label_encoder.fit_transform(data['OnlineSecurity'])
data['OnlineBackup'] = label_encoder.fit_transform(data['OnlineBackup'])
data['DeviceProtection'] = label_encoder.fit_transform(data['DeviceProtection'])
data['TechSupport'] = label_encoder.fit_transform(data['TechSupport'])
data['StreamingTV'] = label_encoder.fit_transform(data['StreamingTV'])
data['StreamingMovies'] = label_encoder.fit_transform(data['StreamingMovies'])
data['Contract'] = label_encoder.fit_transform(data['Contract'])
data['PaperlessBilling'] = label_encoder.fit_transform(data['PaperlessBilling'])
data['PaymentMethod'] = label_encoder.fit_transform(data['PaymentMethod'])
data['Churn'] = label_encoder.fit_transform(data['Churn'])

# Define features and target
X = data.drop(columns=['Churn', 'customerID'])
y = data['Churn']

# Check the class distribution
print("Class distribution in the dataset:")
print(y.value_counts())

# Split the data into training and testing sets (ensure class balance with stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE (reduce k_neighbors to 1 to avoid errors with small datasets)
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get user input for prediction
def get_user_input():
    print("\nEnter the following details for prediction:")
    gender = int(input("Gender (0 = Female, 1 = Male): "))
    senior_citizen = int(input("SeniorCitizen (0 = No, 1 = Yes): "))
    partner = int(input("Partner (0 = No, 1 = Yes): "))
    dependents = int(input("Dependents (0 = No, 1 = Yes): "))
    phone_service = int(input("PhoneService (0 = No, 1 = Yes): "))
    multiple_lines = int(input("MultipleLines (0 = No, 1 = Yes): "))
    internet_service = int(input("InternetService (0 = DSL, 1 = Fiber optic, 2 = No): "))
    online_security = int(input("OnlineSecurity (0 = No, 1 = Yes): "))
    online_backup = int(input("OnlineBackup (0 = No, 1 = Yes): "))
    device_protection = int(input("DeviceProtection (0 = No, 1 = Yes): "))
    tech_support = int(input("TechSupport (0 = No, 1 = Yes): "))
    streaming_tv = int(input("StreamingTV (0 = No, 1 = Yes): "))
    streaming_movies = int(input("StreamingMovies (0 = No, 1 = Yes): "))
    contract = int(input("Contract (0 = Month-to-month, 1 = One year, 2 = Two year): "))
    paperless_billing = int(input("PaperlessBilling (0 = No, 1 = Yes): "))
    payment_method = int(input("PaymentMethod (0 = Electronic check, 1 = Mailed check, 2 = Bank transfer (automatic), 3 = Credit card (automatic)): "))
    tenure = float(input("Tenure (months): "))
    monthly_charges = float(input("MonthlyCharges: "))
    total_charges = float(input("TotalCharges: "))

    # Creating a user input feature vector
    user_input = np.array([[
        gender, senior_citizen, partner, dependents, phone_service, multiple_lines, internet_service,
        online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method, tenure, monthly_charges, total_charges
    ]])

    return user_input

# Get user input for prediction
user_input = get_user_input()

# Predict the churn value for the user input
user_prediction = rf_model.predict(user_input)

# Output the result
if user_prediction[0] == 0:
    print("\nThe customer is predicted to not churn.")
else:
    print("\nThe customer is predicted to churn.")
