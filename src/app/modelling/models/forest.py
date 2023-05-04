import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    default='fitted_models/rf.pkl',
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
    # RandomForestClassifier говорил, что что-то там не влезает во float32
    X = np.clip(X, -10 ** 37, 10 ** 37)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # модель
    rf = RandomForestClassifier()

    # подбор параметров
    grid = {
        'n_estimators': [1000, 2500, 5000],
        'max_depth': [5, 10, 25],
    }
    search = GridSearchCV(rf, grid, scoring='roc_auc', n_jobs=4, verbose=5)
    search.fit(X_train, y_train)

    # скор на тесте
    y_pred_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print('roc_auc_score, test:', auc)

    # обучим модель на всей выборке
    rf.set_params(**search.best_params_)
    rf.fit(X, y)

    # сохраним модель
    with open(args.SAVE, "wb") as f:
        pickle.dump(rf, f)


if __name__ == '__main__':
    main()


