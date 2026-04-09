from abc import ABC, abstractmethod

import pandas as pd


class ProteinPreprocessor(ABC):
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame: ...
