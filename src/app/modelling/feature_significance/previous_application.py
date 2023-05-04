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
    default='data/features/previous_application.csv',
    type=str,
    help='path to previous_application_features.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features_significant/previous_application.csv',
    type=str,
    help='path to previous_application_features.csv',
)
args = parser.parse_args()


def main():
    target = pd.read_csv(args.READ_train, usecols=['SK_ID_CURR', 'TARGET'], index_col='SK_ID_CURR')
    previous_application = pd.read_csv(args.READ_features, index_col='SK_ID_CURR')

    # составим списки названий числовых и бинарных признаков
    num_cols = [col for col in previous_application
                if not np.isin(previous_application[col].dropna().unique(), [0, 1]).all()]
    binary_cols = [col for col in previous_application
                   if np.isin(previous_application[col].dropna().unique(), [0, 1]).all()]

    previous_application = pd.merge(
        previous_application,
        target,
        on='SK_ID_CURR'
    )

    # пропуски
    previous_application = previous_application.fillna(0)

    # проверка
    for col in num_cols:
        previous_application = check_significance_num(previous_application, col)
    # проверка
    for col in binary_cols:
        previous_application = check_significance_cat(previous_application, col)

    previous_application.drop('TARGET', axis=1).to_csv(args.SAVE)


if __name__ == '__main__':
    main()
