import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from scipy.stats import chi2
import numpy as np


def check_significance_num(df, col):
    """
    Проверяет количественный признак по критерию Манна-Уитни;
    если признак не значим, то он удаляется.
    """
    _, p_mw = mannwhitneyu(df[df['TARGET'] == 0][col], df[df['TARGET'] == 1][col])
    if p_mw >= 0.05:
        df = df.drop(col, axis=1)
    return df


def check_significance_cat(df, col):
    """
    Проверяет категориальный признак по критерию хи-квадрат;
    если признак не значим, то он удаляется.
    """
    cross_tab = pd.concat([
        pd.crosstab(df[col], df['TARGET'], margins=False),
        df.groupby(col)['TARGET'].agg(['count', 'mean']).round(4)
    ], axis=1).rename(columns={0: f"target=0", 1: f"target=1", "mean": 'probability_of_default'})

    cross_tab['probability_of_default'] = np.round(cross_tab['probability_of_default'] * 100, 2)

    chi2_stat, p, dof, expected = chi2_contingency(cross_tab.values)
    prob = 0.95
    critical = chi2.ppf(prob, dof)

    if abs(chi2_stat) < critical:
        df = df.drop(col, axis=1)

    return df
