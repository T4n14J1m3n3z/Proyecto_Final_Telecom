import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier as CBC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier as GBC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.dummy import DummyClassifier as DC


def get_train_test_data(df):
    X = df.drop(columns=['churn'])
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=24680)
    return X_train, X_test, y_train, y_test


def dummy_classifier(X_train, y_train):
    model = DC(strategy='constant', constant=1)
    model.fit(X_train, y_train)
    return model


def logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=1234)
    model.fit(X_train, y_train)
    return model


def random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=3,  min_samples_leaf=2, random_state=1234)
    model.fit(X_train, y_train)
    return model


def gradient_boosting_classifier(X_train, y_train):
    model = GBC(n_estimators=100, learning_rate=0.5, max_depth=2,
                min_samples_split=3, min_samples_leaf=2, subsample=1, random_state=1234)
    model.fit(X_train, y_train)
    return model


def k_neighbors_classifier(X_train, y_train):
    model = KNC()
    model.fit(X_train, y_train)
    return model


def catboost_classifier(X_train, y_train):
    model = CBC(iterations=1300, learning_rate=0.04,
                depth=6, l2_leaf_reg=4, random_state=1234, verbose=0)
    model.fit(X_train, y_train)
    return model
