# Employee-salary-prediction-
A machine learning project that predicts salaries using regression 
# salary_prediction_regression.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset from web
url = 'https://api.slingacademy.com/v1/sample-data/files/salaries.csv'
df = pd.read_csv(url)

# Display sample
print("First 5 rows:\n", df.head())

# Encode categorical feature 'Job'
le_job = LabelEncoder()
df['Job_encoded'] = le_job.fit_transform(df['Job'])

# Features and target
X = df[['Age', 'Job_encoded']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MAE: {mae:.2f}")
print(f"R² Score: {r2:.3f}")
print("\nSample Actual vs Predicted:")
print(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred.round(2)}).head())
