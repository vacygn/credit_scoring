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
    default='data/features/bureau.csv',
    type=str,
    help='path to bureau_features.csv',
)
parser.add_argument(
    'SAVE',
    nargs='?',
    default='data/features_significant/bureau.csv',
    type=str,
    help='path to bureau_features.csv',
)
args = parser.parse_args()


def main():
    target = pd.read_csv(args.READ_train, usecols=['SK_ID_CURR', 'TARGET'], index_col='SK_ID_CURR')
    bureau = pd.read_csv(args.READ_features, index_col='SK_ID_CURR')

    # составим списки названий числовых и бинарных признаков
    num_cols = [col for col in bureau
                if not np.isin(bureau[col].dropna().unique(), [0, 1]).all()]
    binary_cols = [col for col in bureau
                   if np.isin(bureau[col].dropna().unique(), [0, 1]).all()]

    bureau = pd.merge(
        bureau,
        target,
        on='SK_ID_CURR'
    )

    # заполняем пропуски
    #bureau['share_overdue_active'] = bureau['share_overdue_active'].replace(np.inf, np.nan).fillna(0)
    bureau['share_overdue'] = bureau['share_overdue'].replace(np.inf, np.nan)
    bureau = bureau.fillna(0)

    # проверка признаков
    for col in num_cols:
        bureau = check_significance_num(bureau, col)
    for col in binary_cols:
        bureau = check_significance_cat(bureau, col)

    bureau.drop('TARGET', axis=1).to_csv(args.SAVE)


if __name__ == '__main__':
    main()
