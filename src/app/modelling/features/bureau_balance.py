import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'BUREAU_BALANCE_READ',
    nargs='?',
    default='data/original_data/bureau_balance.csv',
    type=str,
    help='path for bureau_balance.csv',
)
parser.add_argument(
    'BUREAU_READ',
    nargs='?',
    default='data/original_data/bureau.csv',
    type=str,
    help='path for bureau.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features/bureau_balance.csv',
    type=str,
    help='path for bureau_balance_features.csv',
)
args = parser.parse_args()

bureau_balance = pd.read_csv(args.BUREAU_BALANCE_READ)
bureau_ids = pd.read_csv(args.BUREAU_READ)[['SK_ID_CURR', 'SK_ID_BUREAU']]
bureau_balance = pd.merge(
    bureau_balance,
    bureau_ids,
    on='SK_ID_BUREAU'
)
features = pd.DataFrame()

pivot_status = pd.pivot_table(
    pd.DataFrame(bureau_balance.groupby(['SK_ID_CURR', 'SK_ID_BUREAU'])['STATUS'].first()).reset_index(),
    index='SK_ID_CURR',
    columns='STATUS',
    values='SK_ID_BUREAU',
    aggfunc='count',
    fill_value=0,
)

# кол-во открытых кредитов
cols = ['0', '1', '2', '3', '4', '5', 'X']
features['cnt_open'] = pivot_status[cols].sum(axis=1)

# кол-во закрытых кредитов
features['cnt_closed'] = pivot_status['C']

# кол-во просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
cols = ['0', '1', '2', '3', '4', '5']
features[['cnt_dpd_' + i for i in cols]] = pivot_status[cols]

# кол-во кредитов
features['cnt_total'] = pivot_status.sum(axis=1)

# доля закрытых кредитов
features['ratio_closed'] = features['cnt_closed'] / features['cnt_total']

# доля открытых кредитов
features['ratio_open'] = features['cnt_open'] / features['cnt_total']

# доля просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
cols = ['0', '1', '2', '3', '4', '5']
features[['ratio_dpd_' + i for i in cols]] = pivot_status[cols].divide(features['cnt_total'], axis=0)

del pivot_status

# интервал между последним закрытым кредитом и текущей заявкой
closed = bureau_balance[bureau_balance['STATUS'] == 'C']
last_closed = closed.groupby(['SK_ID_CURR', 'SK_ID_BUREAU']).last()
features['gap_now_last_closed'] = last_closed.groupby('SK_ID_CURR')['MONTHS_BALANCE'].max()
del closed, last_closed

# Интервал между взятием последнего активного займа и текущей заявкой
active = bureau_balance[bureau_balance['STATUS'] != 'C']
last_active = active.groupby(['SK_ID_CURR', 'SK_ID_BUREAU']).last()
features['gap_now_last_closed'] = last_active.groupby('SK_ID_CURR')['MONTHS_BALANCE'].max()
del active, last_active

features.to_csv(args.SAVE)
