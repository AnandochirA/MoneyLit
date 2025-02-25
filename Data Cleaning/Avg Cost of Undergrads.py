import pandas as pd

# Load the dataset
df_cost = pd.read_csv('Assets/Avg Cost by state/nces330_20.csv')

# Encode categorical columns
cat_cols = ['State', 'Type', 'Length', 'Expense']
df_clean = pd.get_dummies(df_cost, columns=cat_cols, prefix=['state', 'type', 'length', 'expense'], drop_first=True)

# Convert booleans to integers
bool_cols = [col for col in df_clean.columns if col.startswith(('state_', 'type_', 'length_', 'expense_'))]
for col in bool_cols:
    df_clean[col] = df_clean[col].astype(int)

# Ensure numeric columns (redundant but safe)
df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
df_clean['Value'] = pd.to_numeric(df_clean['Value'], errors='coerce')

# Check consistency
negatives = df_clean[df_clean['Value'] < 0]
out_of_range = df_clean[(df_clean['Year'] < 2000) | (df_clean['Year'] > 2020)]

# Drop invalid rows (if any)
df_clean = df_clean[(df_clean['Value'] >= 0) & 
                    (df_clean['Year'].between(2000, 2020))]

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"{column} - Lower: {lower}, Upper: {upper}")
    print(f"Rows outside:", df[(df[column] < lower) | (df[column] > upper)].shape[0])
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Remove outliers
df_clean = remove_outliers(df_clean, 'Value')

# Apply domain cap
df_clean = df_clean[(df_clean['Value'] >= 0) & (df_clean['Value'] <= 75000)]

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Save final dataset
df_clean.to_csv('Assets/Avg Cost by state/cleaned.csv', index=False)