import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('Assets/ME/Expenses.csv')

# Display the first few rows
print(df.head())

#Check column names and data types
print(df.info())

#Identify missing values
print(df.isnull().sum())

# For numerical columns: fill missing values with the median
df['Study_year'] = df['Study_year'].fillna(df['Study_year'].median())
df['Monthly_expenses_$'] = df['Monthly_expenses_$'].fillna(df['Monthly_expenses_$'].median())

# For categorical columns: fill missing values with the mode
categorical_columns = ['Living', 'Transporting', 'Smoking', 'Drinks', 'Cosmetics_&_Self-care', 'Monthly_Subscription', 'Part_time_job']
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

#Convert Categorical Columns to Appropriate Types:
categorical_columns = ['Gender', 'Living', 'Scholarship', 'Part_time_job', 'Transporting', 'Smoking', 'Drinks', 'Games_&_Hobbies', 'Cosmetics_&_Self-care', 'Monthly_Subscription']
df[categorical_columns] = df[categorical_columns].astype('category')

#Living Arrangement Binary
df['Living_Home'] = df['Living'].apply(lambda x: 1 if x == 'Home' else 0)

# One-hot encode all categorical columns and ensure the result is integers (0/1)
df = pd.get_dummies(df, columns=['Gender', 'Living', 'Scholarship', 'Part_time_job', 'Transporting', 'Smoking', 'Drinks', 'Games_&_Hobbies', 'Cosmetics_&_Self-care', 'Monthly_Subscription'], drop_first=True, dtype=int)

# Normalize numerical features
scaler = MinMaxScaler()
df[['Age', 'Study_year', 'Monthly_expenses_$']] = scaler.fit_transform(df[['Age', 'Study_year', 'Monthly_expenses_$']])

# Convert Age to numeric if needed
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

#Part-Time Job and Transportation
df['Part_Time_Transport'] = df['Part_time_job_Yes'] * df['Transporting_Motorcycle']

#Financial Health Indicator
df['Financial_Health'] = df['Monthly_expenses_$'] / df['Monthly_expenses_$'].max()  

df.to_csv('Assets/ME/CleanedExpenses.csv', index=False)