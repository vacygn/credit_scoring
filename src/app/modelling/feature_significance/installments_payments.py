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
    default='data/features/installments_payments.csv',
    type=str,
    help='path to installments_payments_features.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features_significant/installments_payments.csv',
    type=str,
    help='path to installments_payments_features.csv',
)
args = parser.parse_args()


def main():
    target = pd.read_csv(args.READ_train, usecols=['SK_ID_CURR', 'TARGET'], index_col='SK_ID_CURR')
    installments_payments = pd.read_csv(args.READ_features, index_col='SK_ID_CURR')

    # составим списки названий числовых и бинарных признаков
    num_cols = [col for col in installments_payments
                if not np.isin(installments_payments[col].dropna().unique(), [0, 1]).all()]
    binary_cols = [col for col in installments_payments
                   if np.isin(installments_payments[col].dropna().unique(), [0, 1]).all()]

    installments_payments = pd.merge(
        installments_payments,
        target,
        on='SK_ID_CURR'
    )

    # заполняем пропуски
    installments_payments = installments_payments.replace([np.inf, -np.inf], np.nan)
    installments_payments = installments_payments.fillna(0)

    # проверка
    for col in num_cols:
        installments_payments = check_significance_num(installments_payments, col)
    # проверка
    for col in binary_cols:
        installments_payments = check_significance_cat(installments_payments, col)

    installments_payments.drop('TARGET', axis=1).to_csv(args.SAVE)


if __name__ == '__main__':
    main()
