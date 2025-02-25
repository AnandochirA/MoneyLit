import pandas as pd

# Load the dataset
df = pd.read_csv('Assets/Earning and Loan/scorecard.csv')

# Drop the unnamed index column
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Define critical columns
missing_cols = ['pred_degree_awarded_ipeds', 'earnings_med', 'count_not_working', 'count_working']

# Check rows with NAs
missing_rows = df[missing_cols].isnull()

# Drop rows with any NA in critical columns
df_clean = df.dropna(subset=missing_cols, how='any')

# One-hot encode state_abbr (only categorical for modeling)
df_clean = pd.get_dummies(df_clean, columns=['state_abbr'], prefix='state', drop_first=True)

# Convert state columns (booleans) to integers
state_cols = [col for col in df_clean.columns if col.startswith('state_')]
for col in state_cols:
    df_clean[col] = df_clean[col].astype(int)

# Ensure numeric columns are proper types
num_cols = ['unitid', 'pred_degree_awarded_ipeds', 'year', 'earnings_med', 
            'count_not_working', 'count_working']
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Check for inconsistencies
# Negative values
negatives = df_clean[(df_clean['earnings_med'] < 0) | 
                    (df_clean['count_not_working'] < 0) | 
                    (df_clean['count_working'] < 0)]

# Year range
out_of_range = df_clean[(df_clean['year'] < 2000) | (df_clean['year'] > 2020)]

# pred_degree_awarded_ipeds valid values (e.g., 1, 2, 3)
invalid_degrees = df_clean[~df_clean['pred_degree_awarded_ipeds'].isin([1, 2, 3])]

# Drop rows with issues (if any)
df_clean = df_clean[(df_clean['earnings_med'] >= 0) & 
                    (df_clean['count_not_working'] >= 0) & 
                    (df_clean['count_working'] >= 0) & 
                    (df_clean['year'].between(2000, 2020)) & 
                    (df_clean['pred_degree_awarded_ipeds'].isin([1, 2, 3]))]

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"{column} - Lower bound: {lower_bound}, Upper bound: {upper_bound}")
    print(f"Rows outside bounds before: {df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]}")
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Check outliers in key columns
outlier_cols = ['earnings_med', 'count_not_working', 'count_working']
for col in outlier_cols:
    df_clean = remove_outliers(df_clean, col)

# Apply domain caps
df_clean = df_clean[(df_clean['earnings_med'] >= 0) & (df_clean['earnings_med'] <= 150000) &
                    (df_clean['count_not_working'] >= 0) & (df_clean['count_not_working'] <= 10000) &
                    (df_clean['count_working'] >= 0) & (df_clean['count_working'] <= 10000)]

# Check for duplicates
print("Number of duplicate rows:", df_clean.duplicated().sum())

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Check shape
print("\nShape after removing duplicates:", df_clean.shape)
print("\nFirst few rows:")
print(df_clean.head())

# Save checkpoint
df_clean.to_csv('Assets/Earning and Loan/CleanedData.csv', index=False)