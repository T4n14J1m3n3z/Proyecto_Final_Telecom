# Asignacion de dataframe limpio:
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle


def load_intercon_dataframe(filepath):

    df_intercon = pd.read_csv(filepath)
    print(df_intercon.info())
    return df_intercon

def categorical_dummies(df_intercon):
    cat_columns = df_intercon.drop(
    columns=['customer_id', 'monthly_charges', 'total_charges', 'churn']).columns

    categorical_columns_df = pd.get_dummies(df_intercon[cat_columns], drop_first=True)

    numerical_columns = df_intercon[['monthly_charges', 'total_charges', 'churn']]

    df_final = pd.concat([categorical_columns_df, numerical_columns], axis=1)
    
    return df_final



def balance_classes(df_final, random_state=24680):
    """
    Aplicar SMOTE para balancear las clases en el DataFrame df_final.

    Returns:
    - pd.DataFrame: DataFrame con las clases balanceadas.
    """
    # Dividir el DataFrame en features (X) y target (y)
    features = df_final.drop(columns=['churn'])
    target = df_final['churn']

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=random_state)
    features_resampled, target_resampled = smote.fit_resample(features, target)

    # Combinar features y target en un nuevo DataFrame
    df_resampled = pd.DataFrame(features_resampled, columns=features.columns)
    df_resampled['churn'] = target_resampled

    # Mezclar el DataFrame para eficiencia en modelos y evitar patrones no deseados
    df_final2= shuffle(df_resampled, random_state=random_state)

    # Imprimir conteo de clases balanceadas
    print(df_final2['churn'].value_counts())

    return df_final2

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def feature_selection(df_final2, k=13, threshold=0.8, random_state=24680):

    # Dividir en características (X) y variable objetivo (y)
    features = df_final2.drop(columns=['churn'])
    target = df_final2['churn']

    # Filtrado - Eliminación de características altamente correlacionadas
    correlation_matrix = features.corr()
    plt.figure(figsize=(25, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
    plt.show()

    # Eliminación de características con alta correlación
    high_corr_var = np.where(np.abs(correlation_matrix) > threshold)
    high_corr_var = [(correlation_matrix.columns[x], correlation_matrix.columns[y])
                     for x, y in zip(*high_corr_var) if x != y and x < y]

    # Eliminación de características altamente correlacionadas
    for x, y in high_corr_var:
        features = features.drop(columns=[y])

    # Filtrar con SelectKBest
    select_k_best = SelectKBest(chi2, k=k)
    select_k_best.fit(features, target)

    # Obtener características seleccionadas por SelectKBest
    kbest_features = features.columns[select_k_best.get_support()]

    # Wrapping RFE con RandomForestClassifier
    model = RandomForestClassifier(random_state=random_state)
    rfe = RFE(estimator=model, n_features_to_select=k)
    features_rfe = rfe.fit_transform(features, target)

    # Obtener características seleccionadas por RFE
    rfe_features = features.columns[rfe.support_]

    # Intersección de características seleccionadas por ambos métodos
    intersect_features = list(set(kbest_features) & set(rfe_features))

    # Unión de características seleccionadas por ambos métodos
    union_features = list(set(kbest_features) | set(rfe_features))

    # Creación de DataFrames con características seleccionadas
    df_intersected = df_final2[intersect_features + ['churn']]
    df_union = df_final2[union_features + ['churn']]

    return df_intersected, df_union




    
