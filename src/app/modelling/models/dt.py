import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from src.app.antifraud.rules import antifraud_rules
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
    default='fitted_models/dt.pkl',
    type=str,
    help='path for saving model',
)
args = parser.parse_args()

def main():
    X = pd.read_csv(args.READ,  index_col='SK_ID_CURR')
    X = antifraud_rules(X)

    # отделяем таргет
    y = X['TARGET']
    X = X.drop('TARGET', axis=1)
    # DecisionTreeClassifier говорил, что что-то там не влезает во float32
    X = np.clip(X, -10**37, 10**37)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    dt = DecisionTreeClassifier()

    # подбор параметров
    grid = {
        'model__max_depth': [5, 10, 25, 50],
        'model__min_samples_leaf': [100, 250, 500, 750, 1000],
        'model__min_samples_split': [2, 5, 10, 15, 20, 25, 50],
    }
    search = GridSearchCV(dt, grid, scoring='roc_auc', n_jobs=4, verbose=5)
    search.fit(X_train, y_train)

    # скор на тесте
    y_pred_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print('roc_auc_score, test:', auc)

    # обучим модель на всей выборке
    dt.set_params(**search.best_params_)
    dt.fit(X, y)

    # сохраним модель
    with open(args.SAVE, "wb") as f:
        pickle.dump(dt, f)


if __name__ == '__main__':
    main()
