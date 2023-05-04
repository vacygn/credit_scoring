import numpy as np
import pandas as pd
from src.app.antifraud.rules import antifraud_rules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
    'READ_MODELS',
    nargs='?',
    default='fitted_models',
    type=str,
    help='path to folder with pickled models',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='fitted_models/blending.pkl',
    type=str,
    help='path for saving model',
)
args = parser.parse_args()


def ret(x):
    """
    Вспомогательная функия.
    Используется для того, чтобы можно было сохранить пайплайн в пикл, а потом считать.
    (если использовать люмбда-функцию, то при сохранении выдает ошибку)
    """
    return x


def main():
    X = pd.read_csv(args.READ, index_col='SK_ID_CURR')
    X = antifraud_rules(X)

    # отделяем таргет
    y = X['TARGET']
    X = X.drop('TARGET', axis=1)
    X = np.clip(X, -10 ** 37, 10 ** 37)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    X_train, X_train_meta, y_train, y_train_meta = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

    # считываем модели
    models = ['logreg.pkl', 'dt.pkl', 'rf.pkl', 'cb.pkl']
    for i in range(len(models)):
        with open(args.READ_MODELS + '//' + models[i], 'rb') as f:
            models[i] = pickle.load(f)

    meta_train = np.ones((X_train_meta.shape[0], len(models)))  # то на чем будет обучаться мета-алгоритм
    meta_test = np.ones((X_test.shape[0], len(models)))  # тест для мета-алгоритма
    for i, m in enumerate(models):
        m.fit(X_train, y_train)
        meta_train[:, i] = m.predict_proba(X_train_meta)[:, 1]
        meta_test[:, i] = m.predict_proba(X_test)[:, 1]

    # мета-алгоритм
    meta = LogisticRegression()
    grid = {
        'C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]
    }
    search = GridSearchCV(meta, grid, scoring='roc_auc', verbose=5, cv=20)
    search.fit(meta_train, y_train_meta)

    # скор на тесте
    y_pred_proba = search.best_estimator_.predict_proba(meta_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print('roc_auc_score, test:', auc)

    # обучим модель на всей мета-выборке
    meta.set_params(**search.best_params_)
    meta.fit(
        np.concatenate((meta_train, meta_test)),
        np.concatenate((y_train_meta, y_test))
    )

    # сохраним модель
    with open(args.SAVE, "wb") as f:
        pickle.dump(meta, f)


if __name__ == '__main__':
    main()
