
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Funcion para la division de dataframes en entrenamiento y prueba:
def div_test(df):
    x = df.drop(columns=['churn'])
    y = df['churn']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=12345)

    return x_train, x_test, y_train, y_test

# Función para entrenamiento y evaluación de modelos
def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)

    # Predicciones
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    # Probabilidades de la clase positiva
    y_pred_proba_train = model.predict_proba(x_train)[:, 1]
    y_pred_proba_test = model.predict_proba(x_test)[:, 1]

    # Calcular AUC ROC
    auc_roc_train = roc_auc_score(y_train, y_pred_proba_train)
    auc_roc_test = roc_auc_score(y_test, y_pred_proba_test)
    print(f"AUC ROC (entrenamiento): {auc_roc_train:.4f}")
    print(f"AUC ROC (prueba): {auc_roc_test:.4f}")

    # Matriz de confusión
    print("Confusion Matrix (prueba):")
    print(confusion_matrix(y_test, y_pred_test))

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Precision
    precision = precision_score(y_test, y_pred_test)
    print(f"Precision: {precision:.4f}")

    # F1-score
    f1 = f1_score(y_test, y_pred_test)
    print(f"F1-score: {f1:.4f}")

    # Calcular la curva ROC para el conjunto de entrenamiento
    fpr_train, tpr_train, _train = roc_curve(y_train, y_pred_proba_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Calcular la curva ROC para el conjunto de prueba
    fpr_test, tpr_test, _test = roc_curve(y_test, y_pred_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Graficar la curva ROC
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='blue', lw=2,
             label=f'ROC curve (entrenamiento) (area = {roc_auc_train:.4f})')
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2,
             label=f'ROC curve (prueba) (area = {roc_auc_test:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

    return

# Función creada para comprobar que no exista un sobreajuste en los modelos entrenados con validacion cruzada

def evaluate_model_with_cv(model, x_test, y_test, cv=10):
    # Validación cruzada en el conjunto de entrenamiento
    scores = cross_val_score(model, x_test, y_test, cv=cv, scoring='roc_auc')
    print(f"AUC ROC promedio en validación cruzada: {
          scores.mean():.4f} +/- {scores.std():.4f}")

    return
