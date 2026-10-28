# preprocessing/eda/eda_stats.py

from collections import defaultdict
import pandas as pd

from configs.config_eda import (
    LABEL_ORDER,
    LABEL_FIELD,
    TEXT_FIELDS,
    DATE_FIELD,
    TIME_GRAIN,
    DATE_FORMAT,
    QUANTILES,
)

from .eda_utils import (
    numeric_summary,
    parse_datetime,
)


# =========================================================
# (1) DATA QUALITY
# =========================================================

def compute_data_quality(df: pd.DataFrame) -> dict:
    quality = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "missing_ratio": {},
        "duplicate_ratio": {},
    }

    for col in df.columns:
        quality["missing_ratio"][col] = float(df[col].isna().mean())

    if "text" in df.columns:
        quality["duplicate_ratio"]["text"] = float(
            df.duplicated(subset=["text"]).mean()
        )

    return quality


# =========================================================
# (2) LABEL DISTRIBUTION
# =========================================================

def compute_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[LABEL_FIELD].value_counts().reindex(LABEL_ORDER, fill_value=0)
    ratios = counts / counts.sum()

    return pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "ratio": ratios.values,
    })


# =========================================================
# (3) TEXT LENGTH STATS (FOR CNN / SVM)
# =========================================================

def compute_text_length_stats(df: pd.DataFrame) -> dict:
    results = defaultdict(dict)

    for field in TEXT_FIELDS:
        if field not in df.columns:
            continue

        lengths = (
            df[field]
            .fillna("")
            .astype(str)
            .apply(lambda x: len(x.split()))
        )

        for label in LABEL_ORDER:
            sub_lengths = lengths[df[LABEL_FIELD] == label]
            results[label][f"{field}_word_count"] = numeric_summary(
                sub_lengths,
                QUANTILES
            )

    return results


# =========================================================
# (4) TEMPORAL STATS (EDA ONLY – NOT FOR TRAINING)
# =========================================================

def compute_temporal_stats(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_FIELD not in df.columns:
        return pd.DataFrame()

    dt = parse_datetime(df[DATE_FIELD], DATE_FORMAT)

    if TIME_GRAIN == "year":
        time_key = dt.dt.year
    elif TIME_GRAIN == "month":
        time_key = dt.dt.to_period("M").astype(str)
    else:
        raise ValueError(f"Unsupported TIME_GRAIN: {TIME_GRAIN}")

    return (
        pd.DataFrame({
            "time": time_key,
            LABEL_FIELD: df[LABEL_FIELD]
        })
        .dropna()
        .groupby(["time", LABEL_FIELD])
        .size()
        .reset_index(name="count")
        .sort_values("time")
    )