import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("ignore")


def prepare_data_for_rule_mining(data):
    """
    Prepare data for association rule mining by discretizing numerical features
    and formatting categorical features appropriately.

    Creates a value_to_column mapping to track which column each value came from,
    with improved handling of numeric values to avoid ambiguity.

    Args:
        data (DataFrame): Data to prepare

    Returns:
        tuple: (list of transactions, value_to_column_mapping)
    """
    prepared_data = data.copy()

    # Create a mapping dictionary to track which column each value belongs to
    value_to_column = {}

    # Process columns in a deterministic order to ensure consistent mapping
    sorted_columns = sorted(prepared_data.columns)

    # First pass: Process all categorical columns
    for col in sorted_columns:
        if (
            prepared_data[col].dtype == "object"
            or prepared_data[col].dtype.name == "category"
        ):
            # Get unique values
            unique_values = prepared_data[col].dropna().unique()

            # Add mappings for both formats (with and without column prefix)
            for val in unique_values:
                item_name = f"{col}_{val}"
                # Store with column prefix
                value_to_column[item_name] = col

                # Only store raw value if it's not a simple number
                # This avoids ambiguity with bin numbers
                if not str(val).isdigit() and not isinstance(val, (int, float)):
                    value_to_column[str(val)] = col

            # Apply transformation
            prepared_data[col] = prepared_data[col].apply(
                lambda x: f"{col}_{x}" if pd.notnull(x) else np.nan
            )

    # Second pass: Process all numerical columns
    for col in sorted_columns:
        if (
            prepared_data[col].dtype in ["int64", "float64"]
            and col in prepared_data.columns
        ):
            # Skip columns with too few unique values or all same value
            if prepared_data[col].nunique() <= 1 or prepared_data[col].std() == 0:
                # For single-value columns, use a special format
                unique_values = prepared_data[col].unique()
                for val in unique_values:
                    item_name = f"{col}_{val}"
                    value_to_column[item_name] = col
                    # DO NOT add the raw value mapping to avoid ambiguity

                prepared_data[col] = prepared_data[col].apply(lambda x: f"{col}_{x}")
                continue

            try:
                # Discretize using quantiles - store the bin mapping
                binned_values = pd.qcut(
                    prepared_data[col], q=4, labels=False, duplicates="drop"
                )

                # Create a special format for binned values to avoid ambiguity
                for bin_label in sorted(binned_values.dropna().unique()):
                    bin_name = f"{col}_{bin_label}"
                    # Only store with column prefix format
                    value_to_column[bin_name] = col

                    # Special format for numeric bins to avoid ambiguity
                    numeric_bin_name = f"{col}_bin_{bin_label}"
                    value_to_column[numeric_bin_name] = col

                # Apply transformation with special format
                prepared_data[col] = binned_values.apply(
                    lambda x: f"{col}_{x}" if pd.notnull(x) else np.nan
                )

            except ValueError:
                # Fall back to equal-width bins if qcut fails
                binned_values = pd.cut(
                    prepared_data[col], bins=4, labels=False, duplicates="drop"
                )

                # Create a special format for binned values to avoid ambiguity
                for bin_label in sorted(binned_values.dropna().unique()):
                    bin_name = f"{col}_{bin_label}"
                    # Only store with column prefix format
                    value_to_column[bin_name] = col

                    # Special format for numeric bins to avoid ambiguity
                    numeric_bin_name = f"{col}_bin_{bin_label}"
                    value_to_column[numeric_bin_name] = col

                # Apply transformation
                prepared_data[col] = binned_values.apply(
                    lambda x: f"{col}_{x}" if pd.notnull(x) else np.nan
                )

    # Convert to transactions
    transactions = prepared_data.values.tolist()
    transactions = [
        [str(item) for item in transaction if pd.notnull(item) and str(item) != "nan"]
        for transaction in transactions
    ]

    # Debug: print a sample of the mapping
    print("\nSample of value to column mapping:")
    sample_items = list(value_to_column.items())[:20]  # First 20 items
    for value, column in sample_items:
        print(f"  {value} -> {column}")

    return transactions, value_to_column


def mine_association_rules(
    transactions,
    min_support=0.3,
    min_confidence=0.5,
    min_lift=1.0,
    value_to_column=None,
):
    """
    Mine association rules from transaction data.

    Now accepts and returns the value_to_column mapping.

    Args:
        transactions (list): List of transactions
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold
        value_to_column (dict): Mapping from values to their original columns

    Returns:
        tuple: (frequent_itemsets DataFrame, rules DataFrame, value_to_column mapping)
    """
    # Convert transactions to a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Update the value_to_column dictionary with any new unmapped items
    if value_to_column is not None:
        # Extract all unique items from the transactions
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)

        # Check for unmapped items and try to infer their column
        for item in all_items:
            if item not in value_to_column and "_" in item:
                # Try to infer column from the prefix
                parts = item.split("_", 1)
                if len(parts) == 2:
                    col, val = parts
                    value_to_column[item] = col

    # Find frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_confidence
        )

        # Filter rules by lift
        rules = rules[rules["lift"] >= min_lift]
        return frequent_itemsets, rules, value_to_column
    else:
        # If no frequent itemsets found, return empty DataFrames
        return pd.DataFrame(), pd.DataFrame(), value_to_column


def prepare_rules_for_application(rules, value_to_column=None):
    """
    Convert association rules from mlxtend to our dictionary format.
    Now uses value_to_column mapping to resolve columns for both antecedent and consequent items.

    Args:
        rules: DataFrame of association rules from mlxtend
        value_to_column: Dictionary mapping values to their original columns

    Returns:
        List: List of dictionaries where each dictionary represents a rule
              with feature-value pairs extracted from both antecedents and consequents
    """
    rule_dicts = []

    for index, rule in rules.iterrows():
        rule_dict = {}

        # The antecedents and consequents in mlxtend format are stored as frozensets
        antecedents = rule["antecedents"]
        consequents = rule["consequents"]

        print(
            f"Processing rule {index}, antecedents: {antecedents}, consequents: {consequents}"
        )

        # Helper function to process items (for both antecedents and consequents)
        def process_item(item, result_dict):
            # First option: Check if the item contains an equals sign (our custom format)
            if "=" in item:
                feature, value = item.split("=", 1)
                # Try to convert numeric values
                try:
                    if value.isdigit():
                        value = int(value)
                    elif "." in value and all(
                        part.isdigit() for part in value.split(".", 1)
                    ):
                        value = float(value)
                except (ValueError, AttributeError):
                    pass  # Keep as string if conversion fails
                result_dict[feature] = value

            # Second option: Check if the item has column_value format
            elif "_" in item:
                parts = item.split("_", 1)
                if len(parts) == 2:
                    feature, value = parts
                    # Try to convert numeric values
                    try:
                        if value.isdigit():
                            value = int(value)
                        elif "." in value and all(
                            part.isdigit() for part in value.split(".", 1)
                        ):
                            value = float(value)
                    except (ValueError, AttributeError):
                        pass  # Keep as string if conversion fails
                    result_dict[feature] = value

            # Third option: Use the value_to_column mapping to find the column
            elif value_to_column is not None and item in value_to_column:
                feature = value_to_column[item]
                value = item  # Keep the original value

                # If the value contains the column name as a prefix, strip it
                if value.startswith(feature + "_"):
                    value = value[len(feature) + 1 :]

                # Try to convert numeric values
                try:
                    if value.isdigit():
                        value = int(value)
                    elif "." in value and all(
                        part.isdigit() for part in value.split(".", 1)
                    ):
                        value = float(value)
                except (ValueError, AttributeError):
                    pass  # Keep as string if conversion fails

                result_dict[feature] = value

            else:
                # For simple values without a column hint, we need help from value_to_column
                if value_to_column is not None and str(item) in value_to_column:
                    feature = value_to_column[str(item)]
                    result_dict[feature] = item
                else:
                    # Last resort: Use the item itself as both feature and value
                    print(f"Warning: Could not determine column for item '{item}'")
                    result_dict[item] = 1

        # Create a temporary consequent dictionary to track consequent features separately
        consequent_dict = {}

        # Process each item in the antecedent
        for item in antecedents:
            process_item(item, rule_dict)

        # Process each item in the consequent
        for item in consequents:
            process_item(item, consequent_dict)

        # Add consequent features to main rule dictionary
        # This puts consequents at the same level as antecedents
        rule_dict.update(consequent_dict)

        # Include metadata from the rule
        rule_dict["confidence"] = rule["confidence"]
        rule_dict["lift"] = rule["lift"]
        rule_dict["support"] = rule["support"]

        # Include the original rule patterns for reference
        rule_dict["original_antecedents"] = list(antecedents)
        rule_dict["original_consequents"] = list(consequents)

        # Mark the consequent fields for reference if needed
        rule_dict["consequent_fields"] = list(consequent_dict.keys())

        rule_dicts.append(rule_dict)

    # Print sample rules
    if rule_dicts:
        print("\nSample rules in new format:")
        for i, rule in enumerate(rule_dicts[: min(3, len(rule_dicts))]):
            # Format a more readable representation of the rule
            # Get antecedent and consequent fields
            antecedent_fields = [
                k
                for k in rule.keys()
                if k
                not in [
                    "confidence",
                    "lift",
                    "support",
                    "original_antecedents",
                    "original_consequents",
                    "consequent_fields",
                ]
                + rule.get("consequent_fields", [])
            ]

            consequent_fields = rule.get("consequent_fields", [])

            # Create formatted strings
            antecedent_str = ", ".join([f"{k}={rule[k]}" for k in antecedent_fields])
            consequent_str = ", ".join([f"{k}={rule[k]}" for k in consequent_fields])

            print(
                f"Rule {i+1}: IF {antecedent_str} THEN {consequent_str} "
                + f"(conf={rule['confidence']:.2f}, lift={rule['lift']:.2f})"
            )

    return rule_dicts


def simple_error_rule_mining(dataset_name):
    """
    Simple implementation of the error rule mining approach.
    Now with improved column mapping.

    Args:
        dataset_name: Name of the dataset to use

    Returns:
        dict: Results of the experiment
    """
    print(f"\n{'='*50}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*50}")

    try:
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        dtf = read_data(dataset_name)
        main_feature = feature_by_dataset(dataset_name)
        print(
            f"Dataset loaded with {len(dtf)} samples and {len(dtf.columns)} features."
        )
        print(f"Target feature: {main_feature}")

        # 2. Train initial model
        print("\n2. Training initial model...")
        X_train, X_test, y_train, y_test, model = process_data(dtf, dataset_name)
        print(
            f"Data split into {len(X_train)} training and {len(X_test)} test samples."
        )

        # 3. Detect and analyze errors
        print("\n3. Detecting and analyzing errors...")
        errors_df, error_transactions, value_to_column = detect_errors(
            X_train, y_train, model, dataset=dataset_name
        )
        error_count = errors_df["error"].sum()
        error_rate = error_count / len(errors_df) * 100
        print(f"Found {error_count} errors in training data ({error_rate:.2f}%).")

        # 4. Mine error patterns with association rules
        print("\n4. Mining error patterns with association rules...")
        min_support = 0.1
        frequent_itemsets, rules, value_to_column = mine_association_rules(
            error_transactions,
            min_support=min_support,
            min_confidence=0.4,
            min_lift=1.0,
            value_to_column=value_to_column,
        )

        if not rules.empty:
            print(f"Found {len(rules)} rules with min_support={min_support}")

            # Sort rules by lift and confidence
            sorted_rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

            top_n = 10
            top_rules = sorted_rules.head(top_n)

            # Convert to our clearer dictionary format with column mapping
            rule_dicts = prepare_rules_for_application(top_rules, value_to_column)

            # 5. Display top rules
            print(f"\n5. Top {top_n} error patterns discovered:")
            for i, rule in enumerate(rule_dicts):
                # Get confidence and lift
                confidence = rule["confidence"]
                lift = rule["lift"]

                # Get antecedent and consequent fields
                consequent_fields = rule.get("consequent_fields", [])
                antecedent_fields = [
                    k
                    for k in rule.keys()
                    if k
                    not in [
                        "confidence",
                        "lift",
                        "support",
                        "original_antecedents",
                        "original_consequents",
                        "consequent_fields",
                    ]
                    + consequent_fields
                ]

                # For display only
                if antecedent_fields:
                    condition_str = ", ".join(
                        [f"{feat}={rule[feat]}" for feat in antecedent_fields]
                    )
                    if consequent_fields:
                        consequent_str = ", ".join(
                            [f"{feat}={rule[feat]}" for feat in consequent_fields]
                        )
                        print(
                            f"Rule {i+1}: IF {condition_str} THEN {consequent_str} (conf={confidence:.2f}, lift={lift:.2f})"
                        )
                    else:
                        print(
                            f"Rule {i+1}: IF {condition_str} THEN Error (conf={confidence:.2f}, lift={lift:.2f})"
                        )
                else:
                    # If we couldn't parse into feature-value pairs, show original
                    original_ant = rule.get("original_antecedents", [])
                    condition_str = ", ".join([str(item) for item in original_ant])

                    if consequent_fields:
                        consequent_str = ", ".join(
                            [f"{feat}={rule[feat]}" for feat in consequent_fields]
                        )
                        print(
                            f"Rule {i+1}: IF {condition_str} THEN {consequent_str} (conf={confidence:.2f}, lift={lift:.2f})"
                        )
                    else:
                        print(
                            f"Rule {i+1}: IF {condition_str} THEN Error (conf={confidence:.2f}, lift={lift:.2f})"
                        )

            # 6. Get original model predictions on test data
            y_pred_orig = model.predict(X_test)
            orig_accuracy = accuracy_score(y_test, y_pred_orig)
            orig_f1 = f1_score(y_test, y_pred_orig, average="weighted")

            print(f"\n6. Original model performance:")
            print(f"Accuracy: {orig_accuracy:.4f}")
            print(f"F1 score: {orig_f1:.4f}")

            # 7. Apply rules to flip predictions on test data
            print("\n7. Applying top 5 rules to test data...")
            y_pred_corrected = predict_with_rules(model, X_test, rule_dicts)

            # 8. Evaluate corrected predictions
            corrected_accuracy = accuracy_score(y_test, y_pred_corrected)
            corrected_f1 = f1_score(y_test, y_pred_corrected, average="weighted")

            print("\n8. Performance after applying error correction rules:")
            print(
                f"Accuracy: {corrected_accuracy:.4f} ({(corrected_accuracy-orig_accuracy)*100:+.2f}%)"
            )
            print(f"F1 score: {corrected_f1:.4f} ({(corrected_f1-orig_f1)*100:+.2f}%)")

            # 9. Count errors fixed vs introduced
            errors_fixed = sum((y_pred_orig != y_test) & (y_pred_corrected == y_test))
            errors_introduced = sum(
                (y_pred_orig == y_test) & (y_pred_corrected != y_test)
            )
            print(f"\n9. Error analysis:")
            print(f"Errors fixed: {errors_fixed}")
            print(f"Errors introduced: {errors_introduced}")
            print(f"Net improvement: {errors_fixed - errors_introduced} samples")

            return {
                "dataset": dataset_name,
                "original_accuracy": orig_accuracy,
                "corrected_accuracy": corrected_accuracy,
                "original_f1": orig_f1,
                "corrected_f1": corrected_f1,
                "errors_fixed": errors_fixed,
                "errors_introduced": errors_introduced,
                "top_rules": rule_dicts,
            }

        else:
            print("No association rules found after multiple attempts.")
            return {"dataset": dataset_name, "status": "No rules found"}

    except Exception as e:
        import traceback

        print(f"Error processing {dataset_name} dataset: {e}")
        print(traceback.format_exc())
        return {"dataset": dataset_name, "status": f"Error: {str(e)}"}


def detect_errors(X, y, model, threshold=0.2, dataset="marketing"):
    """
    Detect errors in model predictions and prepare them for association rule mining.
    For regression, errors are predictions with relative error above threshold.
    For classification, errors are misclassifications.
    Now returns value_to_column mapping.

    Args:
        X (DataFrame): Feature data
        y (Series): True labels
        model (Pipeline): Trained model pipeline
        threshold (float): Relative error threshold for regression
        dataset (str): Dataset name to determine if classification or regression

    Returns:
        tuple: (DataFrame with error column, Error data for rule mining, value_to_column mapping)
    """
    # Make predictions
    y_pred = model.predict(X)

    # Determine if this is a classification or regression task
    # For classification, errors are simply misclassifications
    errors = y_pred != y

    # Create a dataframe with error flag
    data_with_errors = X.copy()
    data_with_errors["error"] = errors.astype(int)

    # Filter to only error cases
    error_data = data_with_errors[data_with_errors["error"] == 1].drop("error", axis=1)

    # Prepare error data for association rule mining
    error_transactions, value_to_column = prepare_data_for_rule_mining(error_data)

    return data_with_errors, error_transactions, value_to_column


def check_if_row_matches_rule(row, rule_items):
    """
    Check if a single data row matches all items in a rule.
    Designed to handle rules where items are just values without column information.

    Args:
        row: A pandas Series representing a single data row
        rule_items: List of items (values) from a rule

    Returns:
        bool: True if the row matches all rule items, False otherwise
    """
    for item in rule_items:
        item_matched = False

        # First, try direct value matching (check if the item appears in any column)
        for col, value in row.items():
            # Convert both to strings for comparison to handle numeric vs string types
            if str(value).lower() == str(item).lower():
                print(
                    f"Match found: '{item}' appears in column '{col}' with value '{value}'"
                )
                item_matched = True
                break

        # Handle special cases for compound items with underscores
        if not item_matched and "_" in item:
            parts = item.split("_")

            # Case 1: Item could be "column_value" format
            if len(parts) == 2:
                col_name, expected_value = parts
                if (
                    col_name in row.index
                    and str(row[col_name]).lower() == str(expected_value).lower()
                ):
                    print(
                        f"Match found: Column '{col_name}' has expected value '{expected_value}'"
                    )
                    item_matched = True

            # Case 2: Item could be a value with underscores (like "United-States")
            # Try to match it against any column
            if not item_matched:
                for col, value in row.items():
                    if str(value).lower() == item.lower():
                        print(f"Match found: '{item}' appears in column '{col}'")
                        item_matched = True
                        break

        # If we still haven't found a match, check if the item appears as part of any column value
        if not item_matched:
            for col, value in row.items():
                if isinstance(value, str) and item.lower() in value.lower():
                    print(
                        f"Partial match found: '{item}' is contained in column '{col}' with value '{value}'"
                    )
                    item_matched = True
                    break

        if not item_matched:
            print(f"No match found for rule item: '{item}'")

            # For debugging, print some row values
            print(f"Row values sample: {dict(list(row.items())[:5])}")
            return False

    # If we get here, all items matched
    print(f"SUCCESS: Row matched all rule items!")
    return True


def predict_with_rules(model, X, rules):
    """
    Apply the model first, then flip predictions for rows matching rules.
    Rules are applied in order of their index (highest lift/confidence first).
    Each row is only flipped once by the highest-priority matching rule.

    Args:
        model: Trained model
        X: Input features DataFrame
        rules: List of rule dictionaries with feature-value pairs

    Returns:
        y_pred: Corrected predictions
    """
    # Get original predictions
    y_pred = model.predict(X)

    # Convert to DataFrame if not already
    X_df = X.copy()

    # Keep track of matches
    matches_count = 0
    rules_matched = {i: 0 for i in range(len(rules))}

    # Track which rows have already been flipped
    flipped_rows = set()

    # For each row, check if it matches any rule
    for row_idx, (i, row) in enumerate(X_df.iterrows()):
        # Skip if this row has already been flipped
        if row_idx in flipped_rows:
            continue

        # Check each rule in order (rules should already be sorted by lift/confidence)
        for rule_idx, rule in enumerate(rules):
            # Skip metadata fields when matching
            rule_conditions = {
                k: v
                for k, v in rule.items()
                if k
                not in [
                    "confidence",
                    "lift",
                    "support",
                    "original_antecedents",
                    "original_consequents",
                    "consequent_fields",
                ]
            }

            # Standard feature-value matching
            matches_rule = True
            for feature, expected_value in rule_conditions.items():
                if feature not in row:
                    matches_rule = False
                    break

                # Handle different types properly
                actual_value = row[feature]

                # Convert types for comparison if needed
                if isinstance(expected_value, (int, float)) and isinstance(
                    actual_value, str
                ):
                    try:
                        actual_value = (
                            float(actual_value)
                            if "." in actual_value
                            else int(actual_value)
                        )
                    except ValueError:
                        pass
                elif isinstance(expected_value, str) and isinstance(
                    actual_value, (int, float)
                ):
                    expected_value = (
                        float(expected_value)
                        if "." in expected_value
                        else int(expected_value)
                    )

                # Compare values
                if actual_value != expected_value:
                    matches_rule = False
                    break

            # If row matches the rule, flip the prediction
            if matches_rule:
                old_pred = y_pred[row_idx]
                y_pred[row_idx] = 1 - y_pred[row_idx]
                matches_count += 1
                rules_matched[rule_idx] += 1
                flipped_rows.add(row_idx)  # Mark this row as flipped

                if i < 10:  # Show more matches for debugging
                    print(
                        f"MATCH! Rule {rule_idx+1} flipped prediction for row {row_idx} from {old_pred} to {y_pred[row_idx]}"
                    )

                # Break out of rules loop, but continue with next row
                break

    # Print summary of matches
    print(f"\nTotal rows where rules were applied: {matches_count} out of {len(X_df)}")
    for rule_idx, count in rules_matched.items():
        if count > 0:
            print(f"Rule {rule_idx+1} matched and flipped {count} predictions")

    return y_pred


def main():
    """
    Run the error rule mining approach on multiple datasets.
    """
    datasets = ["heart", "adult"]  # Add more datasets as needed
    all_results = {}

    for dataset in datasets:
        results = simple_error_rule_mining(dataset)
        all_results[dataset] = results

    # Print summary of results
    print("\n\n" + "=" * 50)
    print("SUMMARY OF RESULTS")
    print("=" * 50)

    for dataset, result in all_results.items():
        print(f"\n{dataset.upper()}")
        if "status" in result:
            print(f"Status: {result['status']}")
        else:
            print(
                f"Accuracy: {result['original_accuracy']:.4f} -> {result['corrected_accuracy']:.4f} ({(result['corrected_accuracy']-result['original_accuracy'])*100:+.2f}%)"
            )
            print(
                f"F1 Score: {result['original_f1']:.4f} -> {result['corrected_f1']:.4f} ({(result['corrected_f1']-result['original_f1'])*100:+.2f}%)"
            )
            print(
                f"Errors fixed: {result['errors_fixed']}, Errors introduced: {result['errors_introduced']}"
            )
            print(
                f"Net improvement: {result['errors_fixed'] - result['errors_introduced']} samples"
            )

    return all_results


def read_breast_cancer_data():
    """
    Loads the Wisconsin Breast Cancer dataset using scikit-learn.
    Returns a pandas DataFrame with feature columns and a binary target column 'target'
    where 0 = malignant, 1 = benign.
    """
    # Load the dataset using scikit-learn
    data = load_breast_cancer()

    # Create DataFrame with feature names
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add target column (0 = malignant, 1 = benign)
    df["target"] = data.target

    return df


def read_heart_disease_data():
    """
    Loads the Heart Disease dataset from the UCI ML Repository.
    Returns a pandas DataFrame with the processed data and a binary target column 'target'
    where 0 = no heart disease, 1 = has heart disease.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    # Read the data
    df = pd.read_csv(url, names=column_names, na_values="?")

    # Handle missing values
    df = df.dropna()

    # Convert categorical columns to strings to ensure proper handling
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # Convert target to binary (0 = no disease, 1 = disease)
    df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

    return df


def read_credit_card_fraud_data():
    """
    Loads the Credit Card Fraud Detection dataset.
    Note: This dataset is large (>100MB) and may be slow to download.
    Returns a pandas DataFrame with the processed data and a binary target column 'Class'
    where 0 = normal transaction, 1 = fraudulent transaction.
    """

    # If you have downloaded the CSV file locally, use:
    df = pd.read_csv("data/creditcard.csv")

    # Note that this is just a sample. In practice, you would download the full dataset
    print("Created a small sample dataset with", len(df), "rows for demonstration.")

    return df


def read_spam_email_data():
    """
    Loads the Spam Email Classification dataset from UCI ML Repository.
    Returns a pandas DataFrame with feature columns and a binary target column 'spam'
    where 0 = ham (legitimate email), 1 = spam.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

    # Feature names for the spambase dataset
    n_features = 57  # There are 57 features in total

    # Generate feature names for word frequencies
    word_features = [f"word_freq_{i}" for i in range(48)]

    # Generate feature names for char frequencies
    char_features = [f"char_freq_{i}" for i in range(6)]

    # Additional features
    capital_features = [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ]

    # Combine all feature names
    column_names = word_features + char_features + capital_features + ["spam"]

    # Read the data
    df = pd.read_csv(url, names=column_names)
    print(
        f"Successfully loaded Spam Email dataset with {len(df)} rows and {len(column_names)} columns"
    )

    return df


def read_adult_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    # Read the data
    df = pd.read_csv(url, names=column_names, sep=",\s*", engine="python")

    df["income"] = df["income"].str.strip().apply(lambda x: 1 if x == ">50K" else 0)

    return df


def read_data(dataset: str):
    if dataset == "cancer":
        return read_breast_cancer_data()
    elif dataset == "adult":
        return read_adult_data()
    elif dataset == "heart":
        return read_heart_disease_data()
    elif dataset == "fraud":
        return read_credit_card_fraud_data()
    elif dataset == "email":
        return read_spam_email_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def feature_by_dataset(dataset):
    if dataset == "email":
        return "spam"
    elif dataset == "cancer":
        return "target"
    elif dataset == "adult":
        return "income"
    elif dataset == "heart":
        return "target"
    elif dataset == "fraud":
        return "Class"

    return "target"


def get_categorical_cols(dataset):
    if dataset == "adult":
        return [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
    elif dataset == "heart":
        return [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ]

    elif dataset == "email":
        return []
    elif dataset == "cancer":
        return []
    elif dataset == "fraud":
        return []

    return []


def process_data(dtf, dataset, use_error_rules=False, rules=None):
    main_feature = feature_by_dataset(dataset)
    X = dtf.drop([main_feature], axis=1)
    y = dtf[main_feature]

    categorical_cols = get_categorical_cols(dataset)

    # Handle missing values in categorical columns for heart dataset
    if dataset == "heart":
        for col in categorical_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mode()[0])

    # Explicitly convert categorical columns to category dtype
    if dataset == "adult":
        for col in categorical_cols:
            X[col] = X[col].astype("category")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if dataset == "adult":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
        )

        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
        )

    else:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
        )

        # Use standard approach with OneHotEncoder for other datasets
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ],
            remainder="passthrough",
        )
    # Define model pipeline
    if use_error_rules and rules is not None:
        try:
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("error_rules", ErrorRuleFeatureGenerator(rules)),
                    ("model", model),
                ]
            )
        except Exception as e:
            print(f"Error creating pipeline with error rules: {e}")
            print("Falling back to regular model")
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
    else:
        print("Regular model")
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    try:
        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred_test = pipeline.predict(X_test)

        print("Accuracy:", "{:.3f}".format(accuracy_score(y_test, y_pred_test)))
        print(
            "F1 Score:",
            "{:.3f}".format(f1_score(y_test, y_pred_test, average="weighted")),
        )

        # For binary classification, we can also compute ROC AUC
        if len(np.unique(y_test)) == 2:
            try:
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                print("ROC AUC:", "{:.3f}".format(roc_auc_score(y_test, y_pred_proba)))
            except:
                pass
    except Exception as e:
        print(f"Error during model training: {e}")
        # If we fail with error rules, try again without them
        if use_error_rules:
            print("Retrying without error rules...")
            return process_data(dtf, dataset, use_error_rules=False)
        else:
            # If even the basic model fails, raise the exception
            raise

    return X_train, X_test, y_train, y_test, pipeline


if __name__ == "__main__":
    main()
