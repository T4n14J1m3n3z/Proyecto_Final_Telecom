import pandas as pd
from sklearn.impute import SimpleImputer, IterativeImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer


def load_and_merge_data():
    contract = pd.read_csv(r'datasets\contract.csv')
    personal = pd.read_csv(r'datasets\personal.csv')
    internet = pd.read_csv(r'datasets\internet.csv')
    phone = pd.read_csv(r'datasets\phone.csv')

    df = contract.merge(personal, on='customerID', how='left').merge(
        internet, on='customerID', how='left').merge(phone, on='customerID', how='left')

    return df


def clean_data(df):
    new_columns = {
        'customerID': 'customer_id', 'BeginDate': 'begin_date', 'EndDate': 'end_date', 'Type': 'type',
        'PaperlessBilling': 'paperless_billing', 'PaymentMethod': 'payment_method', 'MonthlyCharges': 'monthly_charges',
        'TotalCharges': 'total_charges', 'SeniorCitizen': 'senior_citizen', 'Partner': 'partner', 'Dependents': 'dependents',
        'InternetService': 'internet_service', 'OnlineSecurity': 'online_security', 'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection', 'TechSupport': 'tech_support', 'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies', 'MultipleLines': 'multiple_lines'
    }

    df.rename(columns=new_columns, inplace=True)

    df['churn'] = df['end_date'].apply(lambda x: 0 if x == 'No' else 1)

    categorical_columns = [
        'internet_service', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines'
    ]

    imputer = SimpleImputer(strategy='most_frequent')
    for col in categorical_columns:
        df[col] = imputer.fit_transform(df[[col]]).ravel()

    df['total_charges'] = df['total_charges'].replace(' ', np.nan)
    df['total_charges'] = pd.to_numeric(df['total_charges'])

    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[['total_charges']] = imputer.fit_transform(df[['total_charges']])

    df.drop(columns=['end_date', 'begin_date'], inplace=True)

    return df


def save_clean_data(df, path=r'datasets\df_intercon.csv'):
    df.to_csv(path, index=False)


def prepare_data():
    df = load_and_merge_data()
    df = clean_data(df)
    save_clean_data(df)
    return df


if __name__ == "__main__":
    prepare_data()
