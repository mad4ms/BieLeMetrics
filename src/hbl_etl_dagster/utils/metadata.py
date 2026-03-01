from typing import Any, Dict

import pandas as pd
from dagster import MetadataValue


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
        "preview": MetadataValue.md(df.head(n).to_markdown(index=False)),
    }
