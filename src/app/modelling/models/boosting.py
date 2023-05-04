import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from src.app.antifraud.rules import antifraud_rules
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
    default='fitted_models/catb.pkl',
    type=str,
    help='path for saving model',
)
args = parser.parse_args()


def main():
    X = pd.read_csv(args.READ, index_col='SK_ID_CURR')
    X = antifraud_rules(X)

    # отделяем таргет
    y = X['TARGET']
    X = X.drop('TARGET', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # модель
    catb = CatBoostClassifier(
        loss_function='Logloss',
        verbose=False
    )

    # подбор параметров
    grid = {
        'iterations': [250, 500, 1000],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [1.e+00, 1.e+01, 1.e+02],
    }
    search = GridSearchCV(catb, grid, scoring='roc_auc', n_jobs=1, verbose=5)
    search.fit(X_train, y_train)

    # скор на тесте
    y_pred_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print('roc_auc_score, test:', auc)

    # обучим модель на всей выборке
    catb.set_params(**search.best_params_)
    catb.fit(X, y)

    # сохраним модель
    with open(args.SAVE, "wb") as f:
        pickle.dump(catb, f)


if __name__ == '__main__':
    main()
