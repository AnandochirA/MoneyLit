import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df_risk = pd.read_csv('Assets/Risk/financial_risk_assessment.csv')

# Step 2: Analyze missingness pattern
missing_cols = ['Income', 'Credit Score', 'Loan Amount', 'Assets Value', 
                'Number of Dependents', 'Previous Defaults']
missing_rows = df_risk[missing_cols].isnull()

# Impute with median (future-proofed)
for col in missing_cols:
    median_value = df_risk[col].median()
    df_risk[col] = df_risk[col].fillna(median_value) 


# List of categorical columns to encode
cat_columns = ['Gender', 'Education Level', 'Marital Status', 'Employment Status', 
               'Payment History', 'Loan Purpose']

# One-hot encode (drop_first=True to avoid multicollinearity)
df_risk = pd.get_dummies(df_risk, columns=cat_columns, drop_first=True)

# Convert boolean encoded columns to integers (0/1)
encoded_cols = [col for col in df_risk.columns if col.startswith(('Gender_', 'Education Level_', 
                                                                 'Marital Status_', 'Employment Status_', 
                                                                 'Payment History_', 'Loan Purpose_'))]
for col in encoded_cols:
    df_risk[col] = df_risk[col].astype(int)

# Ensure numerical columns are proper types
num_columns = ['Age', 'Income', 'Credit Score', 'Loan Amount', 'Years at Current Job', 
               'Debt-to-Income Ratio', 'Assets Value', 'Number of Dependents', 'Previous Defaults', 
               'Marital Status Change']
for col in num_columns:
    df_risk[col] = pd.to_numeric(df_risk[col], errors='coerce')

# Check for inconsistencies: Unemployed with Years at Current Job > 0
inconsistent = df_risk[(df_risk['Employment Status_Unemployed'] == 1) & 
                       (df_risk['Years at Current Job'] > 0)]

# Fix: Set Years at Current Job to 0 for Unemployed
df_risk.loc[df_risk['Employment Status_Unemployed'] == 1, 'Years at Current Job'] = 0

# Function to detect and remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Columns to check for outliers
outlier_cols = ['Income', 'Credit Score', 'Loan Amount', 'Assets Value', 
                'Debt-to-Income Ratio', 'Previous Defaults', 'Age', 
                'Years at Current Job', 'Number of Dependents', 'Marital Status Change']

# Apply outlier removal
df_clean = df_risk.copy()
for col in outlier_cols:
    df_clean = remove_outliers(df_clean, col)

# Add domain知識 constraint for Credit Score (300-850)
df_clean = df_clean[(df_clean['Credit Score'] >= 300) & (df_clean['Credit Score'] <= 850)]

# Remove duplicates
df_clean = df_risk.drop_duplicates()

# Save checkpoint
df_clean.to_csv('Assets/Risk/cleaned_dataset_financial_risk.csv', index=False)