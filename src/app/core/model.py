import pickle

from src.app.core.api import Features, ScoringDecision, ScoringResult
from src.app.core.calculator import Calculator


class Model(object):
    """Класс для моделей, подсчитывающий вероятность дефолта и одобренную сумму."""

    _threshold = 0.15  # пояснение в ноутбуке

    def __init__(self, model_path: str):
        """Создает объект класса."""
        with open(model_path, 'rb') as pickled_model:
            self._model = pickle.load(pickled_model)
        self._calculator = Calculator()

    def get_scoring_result(self, features: Features):
        """Вычисляет одобренную сумму."""
        proba = self._predict_proba(features)

        decision = ScoringDecision.declined
        amount = 0
        if proba < self._threshold:
            decision = ScoringDecision.accepted
            amount = self._calculator.calc_amount(
                proba,
                features.avg_income_per_child,
                features.ext_score_weighed,
                features.ratio_open,
            )

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )

    def _predict_proba(self, features: Features) -> float:
        """Определяет вероятность невозврата займа."""
        return self._model.predict_proba(features.series)[1]
