import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import traceback
import warnings

warnings.filterwarnings("ignore")


def prepare_data_for_rule_mining(data):
    """
    Prepare data for association rule mining by discretizing numerical features
    and formatting categorical features appropriately.

    Creates a value_to_column mapping to track which column each value came from.

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

    Returns:
        tuple: (rules, value_to_column mapping)
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
                    col, _ = parts
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
        return rules, value_to_column
    else:
        raise ValueError("No frequent itemsets found.")


def prepare_rules_for_application(rules, value_to_column):
    """
    convert association rules from mlxtend to dictionary format.

    Returns:
        List: List of dictionaries where each dictionary represents a rule
        from both antecedents and consequents
    """
    rule_dicts = []

    for index, rule in rules.iterrows():
        rule_dict = {}

        # The antecedents and consequents in mlxtend format are stored as frozensets
        antecedents = rule["antecedents"]
        consequents = rule["consequents"]

        # Helper function to process items (for both antecedents and consequents)
        def process_item(item, result_dict):
            if "_" in item:
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
            else:
                raise ValueError("Item without _ sign found")

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

    return rule_dicts


def detect_errors(X, y, model):
    """
    Detect errors in model predictions and prepare them for association rule mining.
    For regression, errors are predictions with relative error above threshold.
    For classification, errors are misclassifications.
    Now returns value_to_column mapping.

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


def predict_with_rules(model, X, rules):
    """
    Apply the model first, then flip predictions for rows matching rules.
    Each row is only flipped once by the highest-priority matching rule.

    Returns:
        y_pred: predictions after applying rules
    """
    # Get original predictions
    y_pred = model.predict(X)

    # Convert to DataFrame if not already
    X_df = X.copy()

    # Create binned version of X for matching with rules
    X_binned, _ = prepare_data_for_rule_mining(X_df)

    # Convert X_binned from transactions (list of lists) to DataFrame for easier matching
    # First, get all unique items across all transactions
    all_items = set()
    for transaction in X_binned:
        all_items.update(transaction)

    # Create a DataFrame with one-hot encoding for each item
    binned_df = pd.DataFrame(index=X_df.index)
    for i, transaction in enumerate(X_binned):
        for item in transaction:
            binned_df.loc[X_df.index[i], item] = 1

    # Fill NaN with 0 (items not present in a transaction)
    binned_df = binned_df.fillna(0)

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

        # Get the binned representation of this row
        binned_row = binned_df.loc[i]

        # Check each rule in order (rules should already be sorted by lift/confidence)
        for rule_idx, rule in enumerate(rules):
            # Get the original antecedents (these are the binned feature values)
            antecedents = rule.get("original_antecedents", [])

            # Check if all antecedents are present in the binned row
            rule_match = True
            for item in antecedents:
                if item not in binned_row or binned_row[item] != 1:
                    rule_match = False
                    break

            # If row matches the rule, flip the prediction
            if rule_match:
                y_pred[row_idx] = 1 - y_pred[row_idx]
                matches_count += 1
                rules_matched[rule_idx] += 1
                flipped_rows.add(row_idx)  # Mark this row as flipped

                # Break out of rules loop, but continue with next row
                break

    # Print summary of matches
    print(f"\nTotal rows where rules were applied: {matches_count} out of {len(X_df)}")
    for rule_idx, count in rules_matched.items():
        if count > 0:
            print(f"Rule {rule_idx+1} matched and flipped {count} predictions")

    return y_pred


def read_bank_marketing_data():
    """Read the Bank Marketing dataset"""

    df = pd.read_csv("data/bank-additional-full.csv", sep=";")

    # Convert target to binary
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    return df


def read_diabetes_data():
    """Read the Pima Indians Diabetes dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree",
        "age",
        "outcome",
    ]

    # Read the data
    df = pd.read_csv(url, names=column_names)

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


def read_data(name):
    """Get a dataset by name"""
    datasets = {
        "adult": read_adult_data,
        "bank_marketing": read_bank_marketing_data,
        "diabetes": read_diabetes_data,
        "heart": read_heart_disease_data,
    }

    if name not in datasets:
        raise ValueError(f"Dataset {name} not found. ")

    return datasets[name]()


def get_target_feature(dataset_name):
    """
    Return the target feature name for a specific dataset.
    """
    target_features = {
        "adult": "income",
        "bank_marketing": "y",
        "diabetes": "outcome",
        "heart": "target",
    }

    if dataset_name not in target_features:
        raise ValueError(
            f"Dataset {dataset_name} not found. Available datasets: {list(target_features.keys())}"
        )

    return target_features[dataset_name]


def get_categorical_cols(df):
    """
    identify categorical columns in a dataframe.

    Returns:
        List of all categorical column names in the dataset
    """
    # Get columns with object or category dtype
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Also check for columns that are numeric but have few unique values (likely categorical)
    for col in df.select_dtypes(include=["int64"]).columns:
        # If column has less than 10 unique values, it's probably categorical
        if df[col].nunique() < 10 and col not in cat_cols:
            cat_cols.append(col)

    return cat_cols


def process_data(dtf, dataset):
    """
    Process the data for training a model.
    """
    main_feature = get_target_feature(dataset)
    X = dtf.drop([main_feature], axis=1)
    y = dtf[main_feature]

    categorical_cols = get_categorical_cols(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, pipeline


def error_rule_mining(dataset_name):
    """
    main function to run the error rule mining approach on a single dataset.
    returns all the results.
    """
    print(f"\n{'='*50}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*50}")

    try:
        # 1. Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        dtf = read_data(dataset_name)

        # 2. Train initial model
        print("\n2. Training initial model...")
        X_train, X_test, y_train, y_test, model = process_data(dtf, dataset_name)

        # 3. find errors
        print("\n3. Finding errors...")
        errors_df, error_transactions, value_to_column = detect_errors(
            X_train, y_train, model
        )
        error_count = errors_df["error"].sum()
        error_rate = error_count / len(errors_df) * 100
        print(f"Found {error_count} errors in training data ({error_rate:.2f}%).")

        # 4. Mine error patterns with association rules
        print("\n4. Mining error patterns with association rules...")
        min_support = 0.05
        rules, value_to_column = mine_association_rules(
            error_transactions,
            min_support=min_support,
            min_confidence=0.4,
            min_lift=1.0,
            value_to_column=value_to_column,
        )

        if not rules.empty:
            print(f"Found {len(rules)} rules with min_support={min_support}")

            # Sort rules by lift and minus confidence
            sorted_rules = rules.sort_values(
                by=["lift", "confidence"], ascending=[False, True]
            )

            top_n = 5
            top_rules = sorted_rules.head(top_n)

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

            # 6. Get original model predictions on test data
            y_pred_orig = model.predict(X_test)
            orig_accuracy = accuracy_score(y_test, y_pred_orig)
            orig_f1 = f1_score(y_test, y_pred_orig, average="weighted")
            orig_roc_auc = roc_auc_score(y_test, y_pred_orig)

            print(f"\n6. Original model performance:")
            print(f"Accuracy: {orig_accuracy:.4f}")
            print(f"F1 score: {orig_f1:.4f}")
            print(f"ROC AUC: {orig_roc_auc:.4f}")

            # 7. Apply rules to flip predictions on test data
            print("\n7. Applying top rules to test data...")
            y_pred_corrected = predict_with_rules(model, X_test, rule_dicts)

            # 8. Evaluate corrected predictions
            corrected_accuracy = accuracy_score(y_test, y_pred_corrected)
            corrected_f1 = f1_score(y_test, y_pred_corrected, average="weighted")
            corrected_roc_auc = roc_auc_score(y_test, y_pred_corrected)

            print("\n8. Performance after applying error correction rules:")
            print(
                f"Accuracy: {corrected_accuracy:.4f} ({(corrected_accuracy-orig_accuracy)*100:+.2f}%)"
            )
            print(f"F1 score: {corrected_f1:.4f} ({(corrected_f1-orig_f1)*100:+.2f}%)")
            print(
                f"ROC AUC: {corrected_roc_auc:.4f} ({(corrected_roc_auc-orig_roc_auc)*100:+.2f}%)"
            )

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
                "original_roc_auc": orig_roc_auc,
                "corrected_f1": corrected_f1,
                "corrected_roc_auc": corrected_roc_auc,
                "errors_fixed": errors_fixed,
                "errors_introduced": errors_introduced,
                "top_rules": rule_dicts,
            }

        else:
            print("No association rules found after multiple attempts.")
            return {"dataset": dataset_name, "status": "No rules found"}

    except Exception as e:

        print(f"Error processing {dataset_name} dataset: {e}")
        print(traceback.format_exc())
        return {"dataset": dataset_name, "status": f"Error: {str(e)}"}


def main():
    """
    Run the error rule mining approach on multiple datasets.
    """
    datasets = [
        "bank_marketing",
        "adult",
        "diabetes",
        "heart",
    ]
    all_results = {}

    for dataset in datasets:
        results = error_rule_mining(dataset)
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
                f"ROC AUC: {result['original_roc_auc']:.4f} -> {result['corrected_roc_auc']:.4f} ({(result['corrected_roc_auc']-result['original_roc_auc'])*100:+.2f}%)"
            )
            print(
                f"Errors fixed: {result['errors_fixed']}, Errors introduced: {result['errors_introduced']}"
            )
            print(
                f"Net improvement: {result['errors_fixed'] - result['errors_introduced']} samples"
            )
            print(f"Top rules:")
            for i, rule in enumerate(result["top_rules"]):
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

    return all_results


if __name__ == "__main__":
    main()
