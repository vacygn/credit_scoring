import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'READ',
    nargs='?',
    default='data/original_data/previous_application.csv',
    type=str,
    help='path for previous_application.csv')
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features/previous_application.csv',
    type=str,
    help='path for previous_application.csv')
args = parser.parse_args()

previous_application = pd.read_csv(args.READ)
features = pd.DataFrame()

cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
        'AMT_GOODS_PRICE', 'RATE_DOWN_PAYMENT', 'CNT_PAYMENT']
aggs = ['min', 'max', 'count', 'var', 'mean']

# агрегаты по всем
aggregates = previous_application.groupby('SK_ID_CURR')[cols].agg(aggs)
aggregates.columns = ['_'.join(col) for col in aggregates.columns.values]
features = aggregates

# агрегаты по одобренным
approved = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Approved']
approved = approved.groupby('SK_ID_CURR')[cols].agg(aggs)
approved.columns = ['_'.join(col)+'approved_' for col in approved.columns.values]
features = pd.merge(
    features,
    approved,
    how='left',
    on='SK_ID_CURR'
)

# агрегаты по отказам
refused = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Refused']
refused = refused.groupby('SK_ID_CURR')[cols].agg(aggs)
refused.columns = ['_'.join(col)+'refused_' for col in refused.columns.values]
features = pd.merge(
    features,
    refused,
    how='left',
    on='SK_ID_CURR'
)

features.to_csv(args.SAVE)
