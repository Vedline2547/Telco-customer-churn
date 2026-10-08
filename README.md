# 📊 Telco Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to **churn (leave the company)** using customer demographic, service, contract, and billing information.

The project uses **Logistic Regression** as a binary classification model and demonstrates data preprocessing, categorical encoding, feature scaling, model training, and evaluation.

## 🚀 Project Overview

Customer churn is an important problem for telecommunications companies because retaining existing customers is often more cost-effective than acquiring new ones.

This project builds a machine learning model that learns from historical customer data and predicts whether a customer will:

- `0` → Stay with the company
- `1` → Churn

## 📂 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains information about:

- Customer demographics
- Tenure
- Phone and internet services
- Online security and backup
- Technical support
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Customer churn

### Important Features

| Feature | Description |
|---|---|
| `gender` | Customer gender |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months with the company |
| `InternetService` | Customer's internet service type |
| `Contract` | Customer's contract type |
| `PaymentMethod` | Customer's payment method |
| `MonthlyCharges` | Monthly customer charges |
| `TotalCharges` | Total customer charges |
| `Churn` | Whether the customer churned |

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- StandardScaler
- Train/Test Split
- Classification Report
- Accuracy Score

## 🔄 Machine Learning Workflow

```text
Load Dataset
     ↓
Data Cleaning
     ↓
Convert TotalCharges to Numeric
     ↓
Remove Missing Values
     ↓
Remove Customer ID
     ↓
Encode Categorical Variables
     ↓
Separate Features and Target
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Logistic Regression
     ↓
Predictions
     ↓
Model Evaluation
```

## 🧹 Data Preprocessing

### Convert `TotalCharges`

`TotalCharges` is converted from text to a numerical data type:

```python
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)
```

Invalid values are converted to `NaN` and subsequently removed.

### Remove Customer ID

The `customerID` column is removed because it is an identifier and does not provide useful predictive information.

### Encode Categorical Variables

Categorical features are converted into numerical features using one-hot encoding:

```python
df = pd.get_dummies(df, drop_first=True)
```

## 🤖 Model

The project uses **Logistic Regression** for binary classification.

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

The model predicts whether a customer will churn based on their characteristics and services.

## 📏 Feature Scaling

`StandardScaler` is used to standardize the features before training:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted only on the training data to prevent information from the test set leaking into the training process.

## 📊 Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report

```python
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy}")
print(classification_report(y_test, y_pred))
```

For a churn prediction problem, **recall and F1-score for the churn class are particularly important**, since identifying customers who are likely to leave can help businesses take preventative action.

## 📁 Project Structure

```text
Telco-Customer-Churn/
│
├── main.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Telco-customer-churn.git
```

### 2. Navigate to the project

```bash
cd Telco-Customer-Churn
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn
```

### 4. Run the project

```bash
python main.py
```

## 🎯 Learning Objectives

This project demonstrates practical application of:

- Data cleaning
- Handling missing values
- Categorical encoding
- Feature scaling
- Binary classification
- Logistic Regression
- Train/test splitting
- Model evaluation
- Customer churn prediction

## 🔮 Future Improvements

Possible improvements include:

- Comparing Logistic Regression with Random Forest and XGBoost
- Hyperparameter tuning
- Feature selection
- Handling class imbalance
- ROC-AUC evaluation
- Precision-Recall analysis
- Feature importance analysis
- Deploying the model using Flask or Streamlit

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning & AI Enthusiast | Python Developer

---

⭐ If you find this project useful, feel free to star the repository!
