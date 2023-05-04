import pandas as pd
import numpy as np
import argparse
from src.app.utils.stats_utils import check_significance_num, check_significance_cat


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'READ_train',
    nargs='?',
    default='data/original_data/application_train.csv',
    type=str,
    help='path to application_train.csv',
)  # тут таргет
parser.add_argument(
    'READ_features',
    nargs='?',
    default='data/features/application_train.csv',
    type=str,
    help='path to train_features.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features_significant/application_train.csv',
    type=str,
    help='path to train_features.csv',
)
args = parser.parse_args()


def main():
    target = pd.read_csv(args.READ_train, usecols=['SK_ID_CURR', 'TARGET'], index_col='SK_ID_CURR')
    application = pd.read_csv(args.READ_features, index_col='SK_ID_CURR')

    # составим списки названий количественных и бинарных признаков
    num_cols = [col for col in application
                if not np.isin(application[col].dropna().unique(), [0, 1]).all()]
    binary_cols = [col for col in application
                   if np.isin(application[col].dropna().unique(), [0, 1]).all()]

    application = pd.merge(
        application,
        target,
        on='SK_ID_CURR'
    )

    # заполняем пропуски
    application['payment_ratio'] = application['payment_ratio'].fillna(application['payment_ratio'].mean())
    application['rate'] = application['rate'].fillna(application['rate'].mean())
    application['avg_child_per_adult'] = application['avg_child_per_adult'].fillna(0)
    application['avg_income_per_child'] = application['avg_income_per_child'].fillna(0)
    application['avg_income_per_adult'] = application['avg_income_per_adult'].fillna(0)

    # проверка признаков
    for col in num_cols:
        application = check_significance_num(application, col)
    for col in binary_cols:
        application = check_significance_cat(application, col)

    application.drop('TARGET', axis=1).to_csv(args.SAVE)


if __name__ == '__main__':
    main()


