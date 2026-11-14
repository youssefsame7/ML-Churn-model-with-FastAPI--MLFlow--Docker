import great_expectations as ge
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    This function implements critical data validation checks that data
    must pass before going into the ml,
    It validates data integrity, business logic, and statistical properties of the data,
    that the model expects.

    """

    print("Validating data...")

    # convert pandas dataframe to great expectations dataframe
    ge_df = ge.dataset.PandasDataset(df)

    # ===== Validating Columns ======

    print("Validating columns...")

    # customerID must exist
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")

    # Core features
    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")

    # Service features (essintial for churn analysis)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")

    # Financial features (key churn predictors)
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    ge_df.expect_column_to_exist("tenure")

    # Contract types must be valid (business constraint)
    ge_df.expect_column_values_to_be_in_set(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "InternetService", ["DSL", "Fiber optic", "No"]
    )

    # === NUMERIC RANGE VALIDATION ===

    print("Validating numeric properties...")

    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_between(
        "MonthlyCharges", min_value=0, max_value=200
    )

    # no nulls in critical numeric features
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    # === DATA CONSISTENCY CHECKS ===

    print("Validating data consistency...")

    # Total charges should generally be >= Monthly charges (except for very new customers)
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95,  # allow some exceptions for new customers with low tenure
    )

    # === RUN VALIDATION SUITE ===

    print("Running complete validation suite...")
    results = ge_df.validate()

    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting

    failed_expectaions = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectaions.append(expectation_type)

    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"Data validation passed: {passed_checks}/{total_checks} checks passed.")
    else:
        print(f"Data validation failed: {failed_checks}/{total_checks} checks failed.")
        print(f"Failed expectations: {failed_expectaions}")

    return results["success"], failed_expectaions
