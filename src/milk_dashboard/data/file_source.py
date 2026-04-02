"""File-backed data source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .interfaces import DataSource


def _read_table(path: Path, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding=encoding)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


@dataclass
class FileDataSource(DataSource):
    """Current v1 adapter for CSV/XLSX files."""

    sales_path: str
    ingredients_path: str
    sales_encoding: str = "ISO-8859-1"
    _source_id: str = "default_files"

    @property
    def source_id(self) -> str:
        return self._source_id

    def load_sales(self) -> pd.DataFrame:
        return _read_table(Path(self.sales_path), encoding=self.sales_encoding)

    def load_ingredients(self) -> pd.DataFrame:
        return _read_table(Path(self.ingredients_path))

