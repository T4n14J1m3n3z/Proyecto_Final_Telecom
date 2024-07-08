import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from src.data_preparation import prepare_data
from src.eda import plot_eda
from src.feature_selection import feature_selection
from src.modeling import (
    dummy_classifier, get_train_test_data, logistic_regression,
    random_forest, gradient_boosting_classifier, k_neighbors_classifier, catboost_classifier
)
from src.evaluation import evaluate_model


def main():
    # Paso 1: Preparar los datos
    df = prepare_data()

    # Paso 2: Realizar Análisis Exploratorio de Datos (EDA)
    plot_eda(df)

    # Paso 3: Selección de características
    df_intersected, df_union = feature_selection(df)

    # Paso 4: Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = get_train_test_data(df_union)

    # Paso 5: Entrenar y evaluar modelos
    dummy_model = dummy_classifier(X_train, y_train)
    logistic_regression_model = logistic_regression(X_train, y_train)
    random_forest_model = random_forest(X_train, y_train)
    gbc_model = gradient_boosting_classifier(X_train, y_train)
    knn_model = k_neighbors_classifier(X_train, y_train)
    cat_model = catboost_classifier(X_train, y_train)

    # Evaluación de modelos y generación de gráficos ROC
    dummy_results = evaluate_model(
        dummy_model, X_test, y_test, 'dummy_classifier')
    lr_results = evaluate_model(
        logistic_regression_model, X_test, y_test, "Logistic_regression")
    rf_results = evaluate_model(
        random_forest_model, X_test, y_test, "Random_forest")
    gbc_results = evaluate_model(
        gbc_model, X_test, y_test, "Gradient_boosting")
    knn_results = evaluate_model(knn_model, X_test, y_test, "K_neighbors")
    cat_results = evaluate_model(
        cat_model, X_test, y_test, "Catboost_classifier")

    # Paso 6: Mostrar resultados
    print("Dummy Classifier Results:", dummy_results)
    print("Logistic Regression Results:", lr_results)
    print("Random Forest Results:", rf_results)
    print("Gradient Boosting Results:", gbc_results)
    print("K Neighbors Results:", knn_results)
    print("CatBoost Results:", cat_results)


if __name__ == "__main__":
    main()
