from typing import Any, Dict

import pandas as pd
from dagster import MetadataValue


def markdown_table(df: pd.DataFrame, n: int | None = None) -> str:
    """Render a markdown table that is safe for DataFrames containing `pd.NA`."""
    if df is None or df.empty:
        return "*(empty)*"

    preview = df.head(n).copy() if n is not None else df.copy()
    preview = preview.astype(object).where(pd.notna(preview), "")
    return preview.to_markdown(index=False)


def preview_metadata(df: pd.DataFrame, n: int = 10) -> Dict[str, Any]:
    """
    Generate standard metadata for a DataFrame asset, including row/col counts
    and a markdown preview of the first n rows.
    """
    if df is None or df.empty:
        return {
            "n_rows": 0,
            "n_columns": 0,
            "preview": MetadataValue.md("*(empty)*"),
        }

    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "preview": MetadataValue.md(markdown_table(df, n=n)),
    }
