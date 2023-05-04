import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'READ',
    nargs='?',
    default='data/original_data/installments_payments.csv',
    type=str,
    help='path for installments_payments.csv',)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features/installments_payments.csv',
    type=str,
    help='path for installments_payments_features.csv',
)
args = parser.parse_args()

installments_payments = pd.read_csv(args.READ)
features = pd.DataFrame()

# процент выплаты
installments_payments['payment_ratio'] = installments_payments['AMT_PAYMENT'] / installments_payments['AMT_INSTALMENT']
# сколько осталось
installments_payments['payment_left'] = installments_payments['AMT_INSTALMENT'] - installments_payments['AMT_PAYMENT']
# разность между тем, когда была совершена выплата, и тем, когда предполагалось ее совершение
installments_payments['diff'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']

cols = ['payment_ratio', 'payment_left', 'diff',
        'AMT_INSTALMENT', 'AMT_PAYMENT', 'DAYS_ENTRY_PAYMENT']
aggs = ['min', 'max', 'mean']

features = installments_payments.groupby('SK_ID_CURR')[cols].agg(aggs)
features.columns = ['_'.join(col) for col in features.columns.values]

features.to_csv(args.SAVE)
