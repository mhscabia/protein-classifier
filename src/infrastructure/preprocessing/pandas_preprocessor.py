import pandas as pd

from src.domain.interfaces.preprocessor import ProteinPreprocessor


class PandasPreprocessor(ProteinPreprocessor):
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
