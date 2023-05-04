import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.app.antifraud.rules import antifraud_rules
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import argparse


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'READ',
    nargs='?',
    default='data/dataset.csv',
    type=str,
    help='path to dataset.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='fitted_models/logreg.pkl',
    type=str,
    help='path for saving model',
)
args = parser.parse_args()


def ret(x):
    """
    Вспомогательная функия.
    Используется для того, чтобы можно было сохранить пайплайн в пикл.
    (если использовать люмбда-функцию, то выдает ошибку)
    """
    return x


def main():
    X = pd.read_csv(args.READ,  index_col='SK_ID_CURR')
    X = antifraud_rules(X)

    # отделяем таргет
    y = X['TARGET']
    X = X.drop('TARGET', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # считаем индексы количественных и бинарных колонок
    bin_cols = [col for col in X if np.isin(X[col].dropna().unique(), [0, 1]).all()]
    num_cols = [col for col in X if not np.isin(X[col].dropna().unique(), [0, 1]).all()]

    # задаем преобразования колонок
    column_transformer = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('bin', FunctionTransformer(ret), bin_cols)  # то есть ничего с этими колонками не делаем

    ])

    # отбор признаков при помощи регрессии
    logreg = LogisticRegression(penalty='l1', C=1, solver='liblinear')
    pipe_logreg = Pipeline([
            ('column_transformer', column_transformer),
            ('model', logreg)
    ])
    pipe_logreg.fit(X_train, y_train)

    # считаем индексы клонок, которые регрессия посчитала незначимыми
    coefficients = np.array(logreg.coef_)
    coefficients = np.where(abs(coefficients[0]) > 0)[0]

    # задаем преобразование,убирающее лишние колонки
    logreg_transformer = ColumnTransformer([
        ('logreg_features', FunctionTransformer(ret), coefficients)
    ])

    # модель и пайплайн
    logreg = LogisticRegression(solver='newton-cholesky')
    pipe_logreg = Pipeline([
        ('column_transformer', column_transformer),
        ('logreg_transformer', logreg_transformer),
        ('model', logreg),
    ])

    # подбор параметров
    grid = {
        'model__C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]
    }
    search = GridSearchCV(pipe_logreg, grid, scoring='roc_auc', verbose=5)
    search.fit(X_train, y_train)

    # скор на тесте
    y_pred_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    logreg_auc = roc_auc_score(y_test, y_pred_proba)
    print('roc_auc_score, test:', logreg_auc)

    # обучим модель на всей выборке
    pipe_logreg.set_params(**search.best_params_)
    pipe_logreg.fit(X, y)

    # сохраним модель
    with open(args.SAVE, "wb") as f:
        pickle.dump(pipe_logreg, f)


if __name__ == '__main__':
    main()
