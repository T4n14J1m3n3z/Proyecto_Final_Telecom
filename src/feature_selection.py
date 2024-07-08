import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def feature_selection(df):
    # Crear variables dummy
    cat_columns = df.drop(
        columns=['customer_id', 'monthly_charges', 'total_charges', 'churn']).columns
    categorical_columns_df = pd.get_dummies(df[cat_columns], drop_first=True)
    numerical_columns = df[['monthly_charges', 'total_charges', 'churn']]
    df_final = pd.concat([categorical_columns_df, numerical_columns], axis=1)

    # Sobremuestreo con SMOTE
    features = df_final.drop(columns=['churn'])
    target = df_final['churn']
    smote = SMOTE(random_state=24680)
    features_r, target_r = smote.fit_resample(features, target)

    df_resampled = pd.DataFrame(features_r, columns=features.columns)
    df_resampled['churn'] = target_r.reset_index(drop=True)
    df_final2 = shuffle(df_resampled, random_state=24680)

    # Filtrado de características altamente correlacionadas
    correlation_matrix = features.corr()
    plt.figure(figsize=(25, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
    plt.savefig(r'plots\correlation_matrix.png')
    plt.close()

    threshold = 0.8
    high_corr_var = np.where(np.abs(correlation_matrix) > threshold)
    high_corr_var = [(correlation_matrix.columns[x], correlation_matrix.columns[y])
                     for x, y in zip(*high_corr_var) if x != y and x < y]

    for x, y in high_corr_var:
        features = features.drop(columns=[y])

    # Selección de características con SelectKBest
    select_k_best = SelectKBest(chi2, k=13)
    select_k_best.fit(features, target)
    kbest_features = features.columns[select_k_best.get_support()]

    # Selección de características con RFE
    model = RandomForestClassifier(random_state=24680)
    rfe = RFE(estimator=model, n_features_to_select=13)
    features_rfe = rfe.fit_transform(features, target)
    rfe_features = features.columns[rfe.support_]

    # Intersección y unión de características
    intersect_features = list(set(kbest_features) & set(rfe_features))
    union_features = list(set(kbest_features) | set(rfe_features))

    df_intersected = df_final2[intersect_features + ['churn']]
    df_union = df_final2[union_features + ['churn']]

    df_intersected.to_csv(r'datasets\df_intersected.csv', index=False)
    df_union.to_csv(r'datasets\df_union.csv', index=False)

    return df_intersected, df_union
