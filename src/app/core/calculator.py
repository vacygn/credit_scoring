class Calculator(object):
    """Класс, подсчитывающий одобренную сумму."""

    _thresholds = [0.02, 0.05, 0.09, 0.12]
    _ratio_cond = 0.5
    _child_income_cond = 0
    _ext_score_cond = 0.72
    _amounts = [150, 100, 75, 50, 25, 15, 10]

    def calc_amount(
        self,
        proba: float,
        avg_income_per_child: float,
        ext_score_weighed: float,
        ratio_open: float,
    ) -> int:
        """Функция вычисляет одобренную сумму."""
        if proba < self._thresholds[0]:
            return self._amounts[0]
        if ext_score_weighed >= self._ext_score_cond:
            return self._calc_amount_ext_score(proba, ratio_open)
        if avg_income_per_child == self._child_income_cond:
            return self._amounts[5]
        return self._amounts[6]

    def _calc_amount_ext_score(
        self,
        proba: float,
        ratio_open: float,
    ) -> int:
        """Функция подсчитывает выдаваемую сумму для случая ext_score_weighed > 0.72."""
        if proba < self._thresholds[1]:
            return self._amounts[1]
        if proba < self._thresholds[2]:
            return self._amounts[2]
        if proba < self._thresholds[3]:
            if ratio_open < self._ratio_cond:
                return self._amounts[3]
            return self._amounts[4]
        return self._amounts[5]
