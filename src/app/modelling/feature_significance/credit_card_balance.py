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
    'READ_cards',
    nargs='?',
    default='data/original_data/credit_card_balance.csv',
    type=str,
    help='path to credit_card_balance.csv',
)
parser.add_argument(
    'READ_features',
    nargs='?',
    default='data/features/credit_card_balance.csv',
    type=str,
    help='path to credit_card_balance_features.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features_significant/credit_card_balance.csv',
    type=str,
    help='path to credit_card_balance_features.csv',
)
args = parser.parse_args()


def main():
    target = pd.read_csv(args.READ_train, usecols=['SK_ID_CURR', 'TARGET'], index_col='SK_ID_CURR')
    decoy = pd.read_csv(args.READ_cards, usecols=['SK_ID_CURR', 'SK_ID_PREV'])
    credit_card_balance = pd.read_csv(args.READ_features, index_col='SK_ID_PREV')

    credit_card_balance = pd.merge(
        credit_card_balance,
        decoy,
        on='SK_ID_PREV'
    )
    del decoy

    credit_card_balance = credit_card_balance.groupby('SK_ID_CURR').mean().drop('SK_ID_PREV', axis=1)

    # составим списки названий числовых и бинарных признаков
    num_cols = [col for col in credit_card_balance
                if not np.isin(credit_card_balance[col].dropna().unique(), [0, 1]).all()]
    binary_cols = [col for col in credit_card_balance
                   if np.isin(credit_card_balance[col].dropna().unique(), [0, 1]).all()]

    credit_card_balance = pd.merge(
        credit_card_balance,
        target,
        on='SK_ID_CURR'
    )

    # заполняем пропуски
    credit_card_balance = credit_card_balance.replace([np.inf, -np.inf], np.nan)
    credit_card_balance = credit_card_balance.fillna(0)

    # проверка
    for col in num_cols:
        credit_card_balance = check_significance_num(credit_card_balance, col)
    # проверка
    for col in binary_cols:
        credit_card_balance = check_significance_cat(credit_card_balance, col)

    credit_card_balance.drop('TARGET', axis=1).to_csv(args.SAVE)


if __name__ == '__main__':
    main()
