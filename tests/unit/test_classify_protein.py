import pytest

from src.application.use_cases.classify_protein import ClassifyProteinUseCase
from src.infrastructure.prediction.inference_pipeline import InferencePipeline
from src.domain.interfaces.classifier import HierarchicalClassifier


class FakeClassifier(HierarchicalClassifier):
    def train(self, X, y, hierarchy):
        pass

    def predict(self, X):
        return ["GO:0005488;GO:0003674"]


class TestClassifyProteinUseCase:
    def test_returns_predicted_terms(self):
        clf = FakeClassifier()
        pipeline = InferencePipeline(classifier=clf)
        use_case = ClassifyProteinUseCase(pipeline)

        result = use_case.execute("ACDEFGHIKLMNPQRSTVWY")

        assert isinstance(result, set)
        assert len(result) > 0

    def test_contains_expected_terms(self):
        clf = FakeClassifier()
        pipeline = InferencePipeline(classifier=clf)
        use_case = ClassifyProteinUseCase(pipeline)

        result = use_case.execute("ACDEFGHIKLMNPQRSTVWY")

        assert "GO:0005488" in result
        assert "GO:0003674" in result

    def test_delegates_to_pipeline(self):
        clf = FakeClassifier()
        pipeline = InferencePipeline(classifier=clf)
        use_case = ClassifyProteinUseCase(pipeline)

        result1 = use_case.execute("AAA")
        result2 = use_case.execute("CCC")

        # Ambas devem retornar o mesmo resultado (mesmo fake)
        assert result1 == result2
