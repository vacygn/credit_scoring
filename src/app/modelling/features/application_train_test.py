import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='paths')
parser.add_argument(
    'TRAIN_READ',
    nargs='?',
    default='data/original_data/application_train.csv',
    type=str,
    help='path for application_train.csv',
)
parser.add_argument(
    'TEST_READ',
    nargs='?',
    default='data/original_data/application_test.csv',
    type=str,
    help='path for application_test.csv',
)
parser.add_argument(
    'TRAIN_SAVE',
    nargs='?',
    default='data/features/application_train.csv',
    type=str,
    help='path for train_features.csv',
)
parser.add_argument(
    'TEST_SAVE',
    nargs='?',
    default='data/features/application_test.csv',
    type=str,
    help='path for test_features.csv',
)
args = parser.parse_args()


def make_features(path_read, path_save):
    """
    Считывает файл, подсчитывает признаки и сохраняет в файл.

    path_read - путь для считывания
    path_save - путь для сохранения
    """
    df = pd.read_csv(path_read, index_col='SK_ID_CURR')
    features = pd.DataFrame(index=df.index)

    # кол-во документов
    document_columns = ['FLAG_DOCUMENT_' + str(i) for i in range(2, 22)]
    features['docs_count'] = df[document_columns].sum(axis=1)

    # есть ли полная информация о доме
    apartment_columns = df.columns[43:90]

    features['flag_apartment_info'] = df[apartment_columns].isna().sum(axis=1)
    features.loc[features['flag_apartment_info'] < 30, 'flag_apartment_info'] = 1
    features.loc[features['flag_apartment_info'] >= 30, 'flag_apartment_info'] = 0

    # кол-во полных лет
    features['age'] = (df['DAYS_BIRTH'] / -365).astype(int)

    # возраст смены документа
    features['doc_age'] = (df['DAYS_ID_PUBLISH'] / -365).astype(int)

    # разница во времени между сменой документа и возрастом на момент смены документы
    features['doc_diff'] = features['age'] - features['doc_age']

    # Признак задержки смены документа. Документ выдается или меняется в 14, 20 и 45 лет
    decoy1 = ((14 <= features['age']) &
              (features['age'] < 20) &
              (features['age'] - features['doc_age'] == 14))
    decoy2 = ((20 <= features['age']) &
              (features['age'] < 45) &
              (features['age'] - features['doc_age'] == 20))
    decoy3 = ((45 <= features['age']) &
              (features['age'] - features['doc_age'] == 45))
    features['doc_change_in_time'] = (decoy1 | decoy2 | decoy3).astype(int)
    del decoy1, decoy2, decoy3

    # доля денег которые клиент отдает на займ за год
    features['payment_ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # кол-во детей в семье на одного взрослого
    features['avg_child_per_adult'] = (df['CNT_CHILDREN'] /
                                       (df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']))

    # доход на ребенка
    features['avg_income_per_child'] = (df['CNT_CHILDREN'].apply(lambda x: 1 if x > 0 else 0) *
                                        df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'])

    # доход на взрослого
    features['avg_income_per_adult'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # процентная ставка
    features['rate'] = (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'] - 1) * 100

    # взвешенный скор внеешних источников (отнормированный процент пропусков)
    features['ext_score_weighed'] = (0.33 * df['EXT_SOURCE_1'].fillna(0)
                                     + 0.73 * df['EXT_SOURCE_2'].fillna(0) + 0.59 * df['EXT_SOURCE_3'].fillna(0))

    # поделим людей на группы в зависимости от пола и образования.
    # в каждой группе посчитаем средний доход.
    # сделаем признак разница емжду средним доходом в группе и доходом заявителя
    features['mean_income_gender_ed'] = (df.groupby(['CODE_GENDER', 'NAME_EDUCATION_TYPE'])['AMT_INCOME_TOTAL']
                                         .transform('mean') - df['AMT_INCOME_TOTAL'])

    features.to_csv(path_save)


make_features(args.TRAIN_READ, args.TRAIN_SAVE)
make_features(args.TEST_READ, args.TEST_SAVE)
