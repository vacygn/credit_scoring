import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'READ',
    nargs='?',
    default='data/original_data/credit_card_balance.csv',
    type=str,
    help='path for credit_card_balance.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features/credit_card_balance.csv',
    type=str,
    help='path for credit_card_balance_features.csv',
)
args = parser.parse_args()

credit_card_balance = pd.read_csv(args.READ)
features = pd.DataFrame()

# посчитайте все возможные аггрегаты по картам.
cols = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT',
        'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL',
        'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']
aggs = ['min', 'max', 'first', 'last', 'count', 'var', 'mean']
aggregates = credit_card_balance.groupby(['SK_ID_PREV'])[cols].agg(aggs)
features = aggregates.copy()
features.columns = ['_'.join(col) for col in features.columns.values]

# отношение к агрегатам за 3 месяца
decoy = credit_card_balance[credit_card_balance['MONTHS_BALANCE'] >= -3].groupby(['SK_ID_PREV'])[cols].agg(aggs)
decoy = (aggregates / decoy).replace(np.inf, np.nan)
decoy.columns = ['_'.join(col)+'_div3' for col in decoy.columns.values]
features = pd.merge(
    features,
    decoy,
    how='left',
    on='SK_ID_PREV'
)

# разность с агрегатами за 3 месяца
decoy = credit_card_balance[credit_card_balance['MONTHS_BALANCE'] >= -3].groupby(['SK_ID_PREV'])[cols].agg(aggs)
decoy = aggregates - decoy
decoy.columns = ['_'.join(col)+'_sub3' for col in decoy.columns.values]
features = pd.merge(
    features,
    decoy,
    how='left',
    on='SK_ID_PREV'
)

features.to_csv(args.SAVE)
