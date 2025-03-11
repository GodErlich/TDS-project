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
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer


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


warnings.filterwarnings("ignore")


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


def predict_with_rule_corrections(pipeline, X, rules=None, correction_map=None):
    """
    Make predictions and apply manual corrections based on rule matches.
    Handles both string and integer rule IDs consistently.

    Args:
        pipeline: Trained model pipeline
        X: Input features
        rules: Association rules DataFrame

    Returns:
        y_pred: Corrected predictions
        rule_matches: DataFrame showing which rules matched each sample
    """
    final_predictions = None
    rule_matches = None

    return final_predictions, rule_matches


def get_rule_matches(pipeline, X, rules):
    """
    Helper function to just get the rule matches without applying corrections.
    Useful for analysis and debugging.
    """
    # Create rule features
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
        X_preprocessed = preprocessor.transform(X)
    else:
        X_preprocessed = X

    rule_generator = ErrorRuleFeatureGenerator(rules)
    rule_generator.fit(X_preprocessed)
    rule_matches = rule_generator.transform(X_preprocessed)

    # Add mapping information for reference
    rule_id_map = {v: k for k, v in rule_generator.rule_column_map.items()}

    return rule_matches, rule_id_map


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


def detect_errors(X, y, model, threshold=0.2, dataset="marketing"):
    """
    Detect errors in model predictions and prepare them for association rule mining.
    For regression, errors are predictions with relative error above threshold.
    For classification, errors are misclassifications.

    Args:
        X (DataFrame): Feature data
        y (Series): True labels
        model (Pipeline): Trained model pipeline
        threshold (float): Relative error threshold for regression
        dataset (str): Dataset name to determine if classification or regression

    Returns:
        DataFrame: Original data with error column
        DataFrame: Error data in a format suitable for rule mining
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
    error_data_prepared = prepare_data_for_rule_mining(error_data)

    return data_with_errors, error_data_prepared


def prepare_data_for_rule_mining(data):
    """
    Prepare data for association rule mining by discretizing numerical features
    and formatting categorical features appropriately.

    Args:
        data (DataFrame): Data to prepare

    Returns:
        list: List of transactions for association rule mining
    """
    prepared_data = data.copy()

    # Discretize numerical columns using quantiles
    for col in prepared_data.select_dtypes(include=["int64", "float64"]).columns:
        # Skip columns with too few unique values or all same value
        if prepared_data[col].nunique() <= 1 or prepared_data[col].std() == 0:
            prepared_data[col] = prepared_data[col].apply(lambda x: f"{col}_{x}")
            continue

        try:
            prepared_data[col] = pd.qcut(
                prepared_data[col], q=4, labels=False, duplicates="drop"
            )
            prepared_data[col] = prepared_data[col].apply(
                lambda x: f"{col}_{x}" if pd.notnull(x) else np.nan
            )
        except ValueError:
            # Fall back to equal-width bins if qcut fails
            prepared_data[col] = pd.cut(
                prepared_data[col], bins=4, labels=False, duplicates="drop"
            )
            prepared_data[col] = prepared_data[col].apply(
                lambda x: f"{col}_{x}" if pd.notnull(x) else np.nan
            )

    # Format categorical columns
    for col in prepared_data.select_dtypes(include=["object"]).columns:
        prepared_data[col] = prepared_data[col].apply(
            lambda x: f"{col}_{x}" if pd.notnull(x) else np.nan
        )

    # Convert to transactions
    transactions = prepared_data.values.tolist()
    transactions = [
        [str(item) for item in transaction if pd.notnull(item) and str(item) != "nan"]
        for transaction in transactions
    ]

    return transactions


def mine_association_rules(
    transactions, min_support=0.3, min_confidence=0.5, min_lift=1.0
):
    """
    Mine association rules from transaction data.

    Args:
        transactions (list): List of transactions
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold

    Returns:
        DataFrame: Frequent itemsets
        DataFrame: Association rules
    """
    # Convert transactions to a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_confidence
        )

        # Filter rules by lift
        rules = rules[rules["lift"] >= min_lift]
        return frequent_itemsets, rules
    else:
        # If no frequent itemsets found, return empty DataFrames
        return pd.DataFrame(), pd.DataFrame()


def apply_rules_to_improve_model(X_train, X_test, y_train, y_test, rules, dataset):
    """
    Apply discovered error patterns to improve the model using various strategies
    """
    print("\n=== Improved Model Performance ===")

    # Strategy 1: Use error rules as features
    original_X = pd.concat([X_train, X_test], axis=0)
    original_y = pd.concat([y_train, y_test], axis=0)

    # Create a DataFrame with features and target
    combined_df = pd.concat([original_X, original_y], axis=1)

    try:
        _, _, _, _, improved_model = process_data(
            combined_df,
            dataset,
            use_error_rules=True,
            rules=rules,
        )
        return improved_model
    except ValueError as e:
        print(f"Warning: Error when applying rules to improve model: {e}")
        print("Falling back to standard model without error rules...")

        return None


def apply_rules_to_predictions(X, y_pred, rules, correction_map=None):
    """
    Apply association rules to modify model predictions for matching data rows.

    Args:
        X: Input features (pandas DataFrame)
        y_pred: Original model predictions (numpy array or pandas Series)
        rules: DataFrame of association rules from Apriori algorithm with columns:
               - antecedents: list of feature conditions that form the rule
               - consequents: list of target conditions
               - rule_id: unique identifier for each rule
        correction_map: Dictionary mapping rule IDs to new prediction values
                       {rule_id: new_prediction_value}

    Returns:
        corrected_predictions: Model predictions after applying rule corrections
        rule_matches: DataFrame showing which rules matched each sample
    """

    # If no correction map provided, create a default one
    if correction_map is None:
        correction_map = {}
        # Default: flip the prediction (0->1, 1->0)
        for rule_id in rules["rule_id"].unique():
            # Get the consequent (error class) for this rule
            if "consequent_values" in rules.columns:
                error_class = rules.loc[
                    rules["rule_id"] == rule_id, "consequent_values"
                ].iloc[0]
                # Set the correction to the opposite of the error class
                correction_map[rule_id] = 1 if error_class == 0 else 0
            else:
                # If consequent values not provided, default to flipping the prediction
                correction_map[rule_id] = None  # Will be determined during correction

    # Create a DataFrame to track rule matches
    rule_matches = pd.DataFrame(index=X.index)

    # Create a copy of the predictions to modify
    corrected_predictions = y_pred.copy()

    # Apply each rule
    for _, rule in rules.iterrows():
        rule_id = rule["rule_id"]
        # Get the features and values that define this rule (antecedents)
        antecedents = rule["antecedents"]

        # Initialize mask for this rule (all True)
        rule_mask = pd.Series(True, index=X.index)

        # Apply each condition in the antecedent
        for feature, value in antecedents:
            if feature in X.columns:
                # Check if the feature exists and matches the value
                if isinstance(value, (list, tuple)):
                    # For categorical features with multiple possible values
                    rule_mask &= X[feature].isin(value)
                else:
                    # For numeric or binary features
                    rule_mask &= X[feature] == value
            else:
                print(
                    f"Warning: Feature '{feature}' from rule {rule_id} not found in data"
                )

        # Store which rows match this rule
        rule_column = f"rule_{rule_id}"
        rule_matches[rule_column] = rule_mask.astype(int)

        # Apply correction if specified
        if rule_id in correction_map:
            matches = rule_mask
            if any(matches):
                correction = correction_map[rule_id]
                # If correction is None, flip the prediction
                if correction is None:
                    corrected_predictions[matches] = 1 - corrected_predictions[matches]
                else:
                    corrected_predictions[matches] = correction
                print(
                    f"Applied correction for rule {rule_id} to {matches.sum()} samples"
                )

    return corrected_predictions, rule_matches


class ErrorRuleApplier:
    """
    A class to apply error correction rules to model predictions
    """

    def __init__(self, rules, correction_map=None):
        """
        Initialize the rule applier with a set of rules

        Args:
            rules: DataFrame of association rules
            correction_map: Dictionary mapping rule IDs to corrected predictions
        """
        self.rules = rules
        self.correction_map = correction_map

    def fit(self, X, y_true, y_pred):
        """
        Analyze the model errors and create a correction map if not provided

        Args:
            X: Feature DataFrame
            y_true: True labels
            y_pred: Model predictions
        """
        if self.correction_map is None:
            self.correction_map = {}
            errors = y_true != y_pred

            # For each rule, determine the best correction
            for rule_id in self.rules["rule_id"].unique():
                rule_column = f"rule_{rule_id}"

                # Apply the rule to find matches
                _, rule_matches = apply_rules_to_predictions(
                    X, y_pred, self.rules[self.rules["rule_id"] == rule_id], None
                )

                if rule_column in rule_matches.columns:
                    # Find samples where the rule matches
                    matches = rule_matches[rule_column] == 1
                    if any(matches):
                        # Find errors within the matches
                        match_errors = errors[matches]

                        # If the rule is associated with errors, determine the correction
                        if match_errors.sum() > 0:
                            # Count errors by predicted class
                            error_0 = (
                                (y_pred[matches] == 0) & (y_true[matches] == 1)
                            ).sum()
                            error_1 = (
                                (y_pred[matches] == 1) & (y_true[matches] == 0)
                            ).sum()

                            # Choose correction based on most common error type
                            if error_0 > error_1:
                                # More errors where model predicted 0 but should be 1
                                self.correction_map[rule_id] = 1
                            elif error_1 > error_0:
                                # More errors where model predicted 1 but should be 0
                                self.correction_map[rule_id] = 0

        return self

    def transform(self, X, y_pred):
        """
        Apply rules to correct predictions

        Args:
            X: Feature DataFrame
            y_pred: Model predictions

        Returns:
            corrected_predictions: Predictions after applying rules
            rule_matches: DataFrame showing which rules matched
        """
        return apply_rules_to_predictions(X, y_pred, self.rules, self.correction_map)


def evaluate_with_rule_corrections(pipeline, X, y_true, rules, correction_map=None):
    """
    Evaluate model with manual rule-based corrections.

    Args:
        pipeline: Trained model pipeline
        X: Test features
        y_true: True target values
        rules: Association rules DataFrame
        correction_map: Dictionary mapping rule IDs to corrected predictions

    Returns:
        metrics: Dictionary of evaluation metrics
        rule_matches: DataFrame showing which rules matched each sample
    """
    # Make predictions with corrections
    y_pred, rule_matches = predict_with_rule_corrections(
        pipeline, X, rules, correction_map
    )

    # Add true values and original predictions for comparison
    rule_matches["true_value"] = y_true
    rule_matches["original_prediction"] = pipeline.predict(X)
    rule_matches["corrected_prediction"] = y_pred

    # Calculate evaluation metrics
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted")

    # For binary classification, calculate ROC AUC
    if len(np.unique(y_true)) == 2:
        try:
            proba_method = getattr(pipeline, "predict_proba", None)
            if callable(proba_method):
                y_pred_proba = pipeline.predict_proba(X)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except:
            pass

    return metrics, rule_matches


# Example of how to use these functions in your main code:
def main_with_rule_corrections():
    datasets = ["adult", "heart", "fraud", "email"]

    for dataset in datasets:
        print(f"\n=== Running for {dataset} dataset with rule corrections ===")

        try:
            # Load and preprocess data
            print("=== Loading and preprocessing data ===")
            dtf = read_data(dataset)

            # Train initial model
            print("\n=== Training initial model ===")
            X_train, X_test, y_train, y_test, model = process_data(dtf, dataset)

            # Detect and analyze errors
            print("\n=== Detecting and analyzing errors ===")
            errors, error_data = detect_errors(X_test, y_test, model, dataset=dataset)
            print(
                f"Number of errors detected: {errors['error'].sum()} out of {len(errors)} samples"
            )

            # Mine error patterns
            print("\n=== Mining error patterns ===")
            frequent_itemsets, rules = mine_association_rules(
                error_data, min_support=0.3, min_confidence=0.5, min_lift=1.2
            )

            if not rules.empty:
                # Sort rules by lift for better insights
                top_n = 10
                sorted_rules = rules.sort_values(
                    by=["confidence", "lift"], ascending=False
                ).head(top_n)

                # Evaluate the model without corrections
                print("\n=== Original Model Performance ===")
                orig_metrics, _ = evaluate_with_rule_corrections(
                    model,
                    X_test,
                    y_test,
                    rules=sorted_rules,
                )
                for metric, value in orig_metrics.items():
                    print(f"{metric}: {value:.3f}")

                for i, (idx, rule) in enumerate(sorted_rules.head(3).iterrows()):
                    # In a real scenario, you would analyze each rule and decide the appropriate correction
                    # For demonstration, we're flipping 0 to 1 and 1 to 0
                    print(
                        f"Rule {idx}: {rule['antecedents']} -> {rule['consequents']} (lift: {rule['lift']:.2f})"
                    )

            else:
                print("No association rules found.")

        except Exception as e:
            import traceback

            print(f"Error processing {dataset} dataset: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main_with_rule_corrections()
    # another idea, to find rules to correct the model we would also compare the prediction.
    # we will check the predictions that the rule made. if most of the predictions are 0 we w
    # ill only fix rule + preditiction 0.
