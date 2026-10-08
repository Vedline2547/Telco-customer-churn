# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
# Load dataset
df = pd.read_csv('Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Display first few rows of dataset
print(df.head())
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Remove missing values
df = df.dropna()
# Remove customerID
df = df.drop(columns=['customerID'])
# Convert categorical columns to numbers
df = pd.get_dummies(df, drop_first=True)
# Create X and y
X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']
# Split dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize logistic regression classifier
model = LogisticRegression()
# Train model
model.fit(X_train,y_train)
# Make predictions on test set
y_pred = model.predict(X_test)
# Evaluate model
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
print(f"Accuracy : {accuracy}")
print("\nClassification report:\n",report)