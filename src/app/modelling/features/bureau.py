import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'READ',
    nargs='?',
    default='data/original_data/bureau.csv',
    type=str,
    help='path for bureau.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features/bureau.csv',
    type=str,
    help='path for bureau_features.csv',
)
args = parser.parse_args()

bureau = pd.read_csv(args.READ)
features = pd.DataFrame()
features['SK_ID_CURR'] = bureau['SK_ID_CURR'].unique()
features = features.set_index('SK_ID_CURR')

# максимальная сумма просрочки
features['max_overdue'] = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].max()

# минимальная сумма просрочки
features['min_overdue'] = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].min()

# какую долю суммы от открытого займа просрочил
active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
features['share_overdue'] = active['AMT_CREDIT_SUM_OVERDUE'] / active['AMT_CREDIT_SUM']
del active

# кол-во кредитов определенного типа
pivot_types = pd.pivot_table(
    bureau[['SK_ID_CURR', 'CREDIT_TYPE', 'SK_ID_BUREAU']],
    index='SK_ID_CURR',
    columns='CREDIT_TYPE',
    values='SK_ID_BUREAU',
    aggfunc='count',
    fill_value=0,
)
pivot_types.columns = ['cnt_'+'_'.join(col.split()).lower() for col in pivot_types.columns]
features = pd.merge(
    features,
    pivot_types,
    how='left',
    on='SK_ID_CURR'
)
del pivot_types

# Кол-во просрочек кредитов определенного типа
pivot_overs = pd.pivot_table(
    bureau[bureau['AMT_CREDIT_SUM_OVERDUE'] > 0][['SK_ID_CURR', 'CREDIT_TYPE', 'SK_ID_BUREAU']],
    index='SK_ID_CURR',
    columns='CREDIT_TYPE',
    values='SK_ID_BUREAU',
    aggfunc='count',
    fill_value=0,
)
pivot_overs.columns = ['cnt_overdue_'+'_'.join(col.split()).lower() for col in pivot_overs.columns]
bureau_features = pd.merge(
    features,
    pivot_overs,
    how='left',
    on='SK_ID_CURR'
)
del pivot_overs

# Кол-во закрытых кредитов определенного типа
pivot_closed = pd.pivot_table(
    bureau[bureau['CREDIT_ACTIVE'] == 'Closed'][['SK_ID_CURR', 'CREDIT_TYPE', 'SK_ID_BUREAU']],
    index='SK_ID_CURR',
    columns='CREDIT_TYPE',
    values='SK_ID_BUREAU',
    aggfunc='count',
    fill_value=0,
)
pivot_closed.columns = ['cnt_closed_'+'_'.join(col.split()).lower() for col in pivot_closed.columns]
features = pd.merge(
    features,
    pivot_closed,
    how='left',
    on='SK_ID_CURR'
)
del pivot_closed

features.to_csv(args.SAVE)
