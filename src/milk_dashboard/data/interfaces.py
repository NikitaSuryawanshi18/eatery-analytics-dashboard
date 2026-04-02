"""Data source interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Contract for pluggable data sources."""

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique source id used in logs and API metadata."""

    @abstractmethod
    def load_sales(self) -> pd.DataFrame:
        """Load raw sales table."""

    @abstractmethod
    def load_ingredients(self) -> pd.DataFrame:
        """Load ingredient mapping table."""

