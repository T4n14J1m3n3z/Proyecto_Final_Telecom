import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_and_process_data():
    # Cargar y combinar dataframes
    contract = pd.read_csv(r'datasets\contract.csv')
    personal = pd.read_csv(r'datasets\personal.csv')
    internet = pd.read_csv(r'datasets\internet.csv')
    phone = pd.read_csv(r'datasets\phone.csv')

    df_intercon = contract.merge(personal, on='customerID', how='left') \
                          .merge(internet, on='customerID', how='left') \
                          .merge(phone, on='customerID', how='left')

    # Renombrar columnas
    new_columns = {'customerID': 'customer_id', 'BeginDate': 'begin_date', 'EndDate': 'end_date', 'Type': 'type', 'PaperlessBilling': 'paperless_billing',
                   'PaymentMethod': 'payment_method', 'MonthlyCharges': 'monthly_charges', 'TotalCharges': 'total_charges',
                   'SeniorCitizen': 'senior_citizen', 'Partner': 'partner', 'Dependents': 'dependents', 'InternetService': 'internet_service',
                   'OnlineSecurity': 'online_security', 'OnlineBackup': 'online_backup', 'DeviceProtection': 'device_protection', 'TechSupport': 'tech_support',
                   'StreamingTV': 'streaming_tv', 'StreamingMovies': 'streaming_movies', 'MultipleLines': 'multiple_lines'}

    df_intercon.rename(columns=new_columns, inplace=True)

    # Crear columna 'churn'
    df_intercon['churn'] = df_intercon['end_date'].apply(
        lambda x: 0 if x == 'No' else 1)

    # Verificar duplicados
    print('Duplicados:', df_intercon['customer_id'].duplicated().sum())

    # Porcentajes de valores faltantes
    total_rows = df_intercon.shape[0]
    null_count = df_intercon.isnull().sum()
    null_values = (null_count / total_rows) * 100

    print("Porcentaje de valores faltantes en cada columna:")
    print(null_values)

    # Imputación de valores faltantes en columnas categóricas
    categorical_columns = ['internet_service', 'online_security', 'online_backup', 'device_protection',
                           'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines']

    imputer = SimpleImputer(strategy='most_frequent')

    for col in categorical_columns:
        df_intercon[col] = imputer.fit_transform(df_intercon[[col]]).ravel()

    # Información final del dataframe
    print(df_intercon.info())

    return df_intercon


def clean_and_export_data(dataframe):
    # Reemplazar valores vacíos en 'total_charges' con NaN
    dataframe['total_charges'] = dataframe['total_charges'].replace(
        ' ', np.nan)

    # Convertir 'total_charges' a tipo numérico
    dataframe['total_charges'] = pd.to_numeric(dataframe['total_charges'])

    # Imputar valores faltantes en 'total_charges' usando IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=42)
    dataframe[['total_charges']] = imputer.fit_transform(
        dataframe[['total_charges']])

    # Eliminar las columnas 'end_date' y 'begin_date'
    dataframe.drop(columns=['end_date', 'begin_date'], inplace=True)

    # Verificar las columnas del DataFrame resultante
    print(dataframe.columns)

    # Exportar el dataframe limpio a un archivo CSV
    dataframe.to_csv(r'datasets\df_intercon.csv', index=False)

    return dataframe  # Devolver el dataframe limpio
