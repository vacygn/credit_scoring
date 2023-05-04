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
    default='data/features/bureau_balance.csv',
    type=str,
    help='path to bureau_balance_features.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features_significant/bureau_balance.csv',
    type=str,
    help='path to bureau_balance_features.csv',
)
args = parser.parse_args()


def main():
    target = pd.read_csv(args.READ_train, usecols=['SK_ID_CURR', 'TARGET'], index_col='SK_ID_CURR')
    bureau_balance = pd.read_csv(args.READ_features, index_col='SK_ID_CURR')

    # составим списки названий числовых и бинарных признаков
    num_cols = [col for col in bureau_balance
                if not np.isin(bureau_balance[col].dropna().unique(), [0, 1]).all()]
    binary_cols = [col for col in bureau_balance
                   if np.isin(bureau_balance[col].dropna().unique(), [0, 1]).all()]

    bureau_balance = pd.merge(
        bureau_balance,
        target,
        on='SK_ID_CURR'
    )

    # заполняем пропуски
    bureau_balance['gap_now_last_closed'] = bureau_balance['gap_now_last_closed'].fillna(0)

    # проверка
    for col in num_cols:
        bureau_balance = check_significance_num(bureau_balance, col)
    # проверка
    for col in binary_cols:
        bureau_balance = check_significance_cat(bureau_balance, col)

    bureau_balance.drop('TARGET', axis=1).to_csv(args.SAVE)


if __name__ == '__main__':
    main()
