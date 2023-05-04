from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd


class ScoringDecision(Enum):
    """Возможные решения модели."""

    ACCEPTED = auto()
    DECLINED = auto()


@dataclass
class ScoringResult(object):
    """Класс, содержащий результаты скоринга."""

    decision: ScoringDecision
    amount: int
    threshold: float
    proba: float


@dataclass
class Features(object):
    """Фичи для принятия решения об одобрении."""

    series: pd.Series

    # фичи для калькулятора
    avg_income_per_child: float
    ext_score_weighed: float
    ratio_open: float
