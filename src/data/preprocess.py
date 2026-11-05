import pandas as pd


def preprocess_data(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """
    Basic data preprocessing

    -trime column names
    -drop id column
    -fix total charges column
    -map churn column to 0 and 1
    -handle NA

    Args:
    data frame and target column

    Returns:
        clean data frame
    """

    # trim column names
    df.columns = df.columns.str.strip()

    # drop id column
    for col in ["CustomerID", "customerID", "customer_id"]:
        if col in df.columns:
            df.drop(columns=[col])

    # fix total charges column
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Senior citizen column to int if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # missing values
    # numeric columns -> fill with 0
    # other columns leave them to get dummified later

    numeric = df.select_dtypes(include=["number"]).columns
    df[numeric] = df[numeric].fillna(0)

    return df
