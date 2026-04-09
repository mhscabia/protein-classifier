from abc import ABC, abstractmethod


class HierarchicalEvaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true: list[str], y_pred: list[str]) -> dict:
        """Retorna: {'hP': float, 'hR': float, 'hF': float}"""
        ...
