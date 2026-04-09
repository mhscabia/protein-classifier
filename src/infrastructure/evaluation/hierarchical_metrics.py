from src.domain.interfaces.evaluator import HierarchicalEvaluator


class HierarchicalMetricsEvaluator(HierarchicalEvaluator):
    def evaluate(self, y_true: list[str], y_pred: list[str]) -> dict:
        raise NotImplementedError
