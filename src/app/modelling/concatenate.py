import argparse

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'READ_FEATURES',
    nargs='?',
    default='data/features_significant',
    type=str,
    help='path to folder with features and application_train.csv',
)
parser.add_argument(
    'READ_APPL',
    nargs='?',
    default='data/original_data/application_train.csv',
    type=str,
    help='path to application_train.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/dataset.csv',
    type=str,
    help='path to folder for saving dataset')
args = parser.parse_args()


def main():
    target = pd.read_csv(args.READ_APPL, index_col='SK_ID_CURR')['TARGET']

    # считываем файлы с фичами
    train = pd.read_csv(args.READ_FEATURES + '//application_train.csv', index_col='SK_ID_CURR')
    bureau_balance = pd.read_csv(args.READ_FEATURES + '//bureau_balance.csv', index_col='SK_ID_CURR')
    bureau = pd.read_csv(args.READ_FEATURES + '//bureau.csv', index_col='SK_ID_CURR')
    credit_card_balance = pd.read_csv(args.READ_FEATURES + '//credit_card_balance.csv', index_col='SK_ID_CURR')
    installments_payments = pd.read_csv(args.READ_FEATURES + '//installments_payments.csv', index_col='SK_ID_CURR')
    previous_application = pd.read_csv(args.READ_FEATURES + '//previous_application.csv', index_col='SK_ID_CURR')

    X = pd.merge(target, train, on='SK_ID_CURR')
    X = pd.merge(X, bureau_balance, on='SK_ID_CURR')
    X = pd.merge(X, bureau, on='SK_ID_CURR')
    X = pd.merge(X, credit_card_balance, on='SK_ID_CURR')
    X = pd.merge(X, installments_payments, on='SK_ID_CURR')
    X = pd.merge(X, previous_application, on='SK_ID_CURR')

    X.to_csv(args.SAVE)


if __name__ == '__main__':
    main()
