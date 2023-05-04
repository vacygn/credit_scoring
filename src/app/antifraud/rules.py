def antifraud_rules(df):
    """Применяет правила и возвращает измененный датафрейм"""
    # 22 процента дефолта
    mask1 = (df['ext_score_weighed'] < 0.25)
    # 15 процентов дефолта
    mask2 = (df['CNT_DRAWINGS_ATM_CURRENT_mean'] < 0) | (df['CNT_DRAWINGS_ATM_CURRENT_mean'] > 0.63)
    # 15 процентов дефолта
    mask3 = df['AMT_BALANCE_min'] > 0
    df = df[~(mask1 | mask2 | mask3)]
    return df


