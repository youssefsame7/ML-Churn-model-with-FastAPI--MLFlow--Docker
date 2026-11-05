import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.

    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.

    """

    # get unique values and remove nulls
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # ====== DETERMINISTIC MAPPINGS ======

    # yes/no mapping

    if valset == {"Yes", "No"}:
        return s.map({"Yes": 1, "No": 0}).astype("Int64")

    # male/female mapping
    if valset == {"Male", "Female"}:
        return s.map({"Male": 1, "Female": 0}).astype("Int64")

    # any other 2-category feature -> use stabel alphabitical ordering
    if len(valset) == 2:
        # sort values to ensure consistent mapping
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}

        return s.astype(str).map(mapping).astype("Int64")

    # non-binary features will be handeld later on
    return s


def build_features(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """
    apply all feature engineering pipline
    that transforms the raw customer data into
    ml-ready data
    """

    df = df.copy()
    print(f" feature engineering on {df.shape[1]} columns...")

    # feature types
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target]
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f" Found {len(cat_cols)} categorical and {len(num_cols)} numeric columns")

    # split catgegorical features by cardinality

    binary_cols = [c for c in cat_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in cat_cols if df[c].dropna().nunique() > 2]

    print(
        f"   Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}"
    )
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")

    # apply binary encoding
    for col in binary_cols:
        org_dtype = df[col].dtype
        df[col] = _map_binary_series(df[col].astype(str))
        print(f"     {col}: {org_dtype} -> binary (0/1)")

    # boolean conversion
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # one-hot encoding for multi-category features
    if multi_cols:
        print(f" One_hot encoding {len(multi_cols)} multi-category features...")
        org_shape = df.shape

        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

        print(f" added {df.shape[1] - org_shape[1]} new columns from one-hot encoding")

    # data type cleanup
    # changing nullable int columns to regular int

    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f" Feature engineering complete: {df.shape[1]} final features")
    return df
