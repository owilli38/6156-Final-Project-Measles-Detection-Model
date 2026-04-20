import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from imblearn.ensemble import BalancedRandomForestClassifier

from data_merging import get_complete_dataset


# ==============================
# 1. HELPERS
# ==============================
def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip(),
        errors='coerce'
    )


def impute_by_state_median(df, cols, state_col="State_1"):
    df = df.copy()

    if state_col not in df.columns:
        state_col = "STNAME"

    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(
                df.groupby(state_col)[col].transform("median")
            )
            df[col] = df[col].fillna(df[col].median())

    return df


def build_model_dataset(target_col="Target_At_Least_2", run_model=True):
    """
    Build training/validation/test splits for the measles model.

    Parameters
    ----------
    target_col : str
        Either 'Target_At_Least_1' or 'Target_At_Least_2'
    run_model : bool
        If True, fits a simple Balanced Random Forest and prints validation AUC

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    valid_targets = ["Target_At_Least_1", "Target_At_Least_2"]
    if target_col not in valid_targets:
        raise ValueError(f"target_col must be one of {valid_targets}")

    # ==============================
    # 2. LOAD DATA
    # ==============================
    df = get_complete_dataset().copy()

    if target_col not in df.columns:
        raise KeyError(f"{target_col} not found in dataset columns.")

    # ==============================
    # 3. CLEAN / STANDARDIZE RATES
    # ==============================
    rate_cols = [
        "Estimate (%)",
        "Percent adults fully vaccinated against COVID-19 (as of 6/10/21)",
        "Estimated hesitant"
    ]

    for col in rate_cols:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
            if df[col].max(skipna=True) > 1:
                df[col] = df[col] / 100.0

    # ==============================
    # 4. IMPUTE IMPORTANT BASE COLS
    # ==============================
    impute_cols = [
        "Estimate (%)",
        "Estimated hesitant",
        "UNDER5_TOT",
        "AGE513_TOT",
        "AGE1417_TOT",
        "POPESTIMATE",
        "Social Vulnerability Index (SVI)",
        "Percent adults fully vaccinated against COVID-19 (as of 6/10/21)"
    ]

    df = impute_by_state_median(df, impute_cols)

    # ==============================
    # 5. FEATURE ENGINEERING
    # ==============================
    if "Estimate (%)" in df.columns:
        df["unvaccinated_rate"] = 1 - df["Estimate (%)"]
        df["low_vax_flag"] = (df["Estimate (%)"] < 0.9).astype(int)

    if (
        "Social Vulnerability Index (SVI)" in df.columns
        and "Estimated hesitant" in df.columns
    ):
        df["risk_interaction_2"] = (
            df["Social Vulnerability Index (SVI)"] * df["Estimated hesitant"]
        )

    if (
        "Percent adults fully vaccinated against COVID-19 (as of 6/10/21)" in df.columns
        and "Estimated hesitant" in df.columns
    ):
        df["vax_hesitancy_interaction"] = (
            df["Percent adults fully vaccinated against COVID-19 (as of 6/10/21)"]
            * df["Estimated hesitant"]
        )

        df["not_covid_vaccinated"] = (
            1 - df["Percent adults fully vaccinated against COVID-19 (as of 6/10/21)"]
        )

    # ------------------------------
    # Group age features
    # ------------------------------
    age_18_44_cols = [
        "AGE1824_TOT",
        "AGE2544_TOT",
        "AGE3034_TOT",
        "AGE3539_TOT"
    ]

    age_45_plus_cols = [
        "AGE4564_TOT",
        "AGE6569_TOT",
        "AGE7074_TOT",
        "AGE7579_TOT",
        "AGE8084_TOT",
        "AGE85PLUS_TOT"
    ]

    age_child_cols = [
        "UNDER5_TOT",
        "AGE513_TOT",
        "AGE1417_TOT"
    ]

    age_18_44_cols = [c for c in age_18_44_cols if c in df.columns]
    age_45_plus_cols = [c for c in age_45_plus_cols if c in df.columns]
    age_child_cols = [c for c in age_child_cols if c in df.columns]

    df["AGE_18_44"] = df[age_18_44_cols].sum(axis=1, skipna=True)
    df["AGE_45_PLUS"] = df[age_45_plus_cols].sum(axis=1, skipna=True)
    df["AGE_CHILD"] = df[age_child_cols].sum(axis=1, skipna=True)

    if "POPESTIMATE" in df.columns:
        pop = df["POPESTIMATE"].replace(0, np.nan)

        df["AGE_18_44_RATIO"] = df["AGE_18_44"] / pop
        df["AGE_45_PLUS_RATIO"] = df["AGE_45_PLUS"] / pop
        df["AGE_CHILD_RATIO"] = df["AGE_CHILD"] / pop

    if (
        "Estimated hesitant" in df.columns
        and "unvaccinated_rate" in df.columns
    ):
        df["hesitancy_vax_interaction"] = (
            df["Estimated hesitant"] * df["unvaccinated_rate"]
        )

    if (
        "AGE_CHILD_RATIO" in df.columns
        and "Estimated hesitant" in df.columns
    ):
        df["risk_interaction"] = (
            df["AGE_CHILD_RATIO"] * df["Estimated hesitant"]
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    # Optional second imputation pass after engineered features
    engineered_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = impute_by_state_median(df, engineered_numeric_cols)

    # ------------------------------
    # Drop original redundant age cols
    # ------------------------------
    cols_to_drop = age_18_44_cols + age_45_plus_cols + age_child_cols
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # ==============================
    # 6. DROP LEAKY / ID-LIKE COLUMNS
    # ==============================
    drop_cols = [
        "Unnamed: 0",
        "FIPS",
        "FIPS Code",
        "COUNTY",
        "STATE",
        "CTYNAME",
        "STNAME",
        "Geography",
        "Geography Type",
        "Footnotes",
        "State",
        "Vaccine/Exemption",
        "total_measles_cases",
        "Population Size",
        "POPESTIMATE",
        "State Code",
        "location_id"
    ]

    # Drop both target columns from X later, but keep chosen target in y
    other_targets = [c for c in ["Target_At_Least_1", "Target_At_Least_2"] if c != target_col]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    geo_keywords = ["MULTIPOLYGON", "Boundary", "Polygon", "Point", "Coordinate"]
    race_keywords = ["White", "Black", "Hispanic", "Asian", "Native", "Pacific", "Ethnicity"]
    gender_keywords = ["MALE", "FEMALE", "FEM"]
    state_keywords = ["Survey", "State Code"]
    age_keywords = ["AGE", "UNDER", "ADULT", "CHILD"]

    keep_age_features = ["AGE_18_44_RATIO", "AGE_CHILD_RATIO", "AGE_45_PLUS_RATIO"]

    drop_all_filters = geo_keywords + race_keywords + gender_keywords + state_keywords + age_keywords

    final_drop_list = [
        c for c in df.columns
        if any(k.lower() in c.lower() for k in drop_all_filters)
        and c not in keep_age_features
    ]

    df = df.drop(columns=final_drop_list, errors="ignore")

    # ==============================
    # 7. SPLIT DATA
    # ==============================
    X = df.drop(columns=[target_col] + other_targets, errors="ignore")
    y = df[target_col].copy()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
    )

    # ==============================
    # 8. ENCODE CATEGORICALS SAFELY
    # ==============================
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

    high_card = [c for c in cat_cols if X_train[c].nunique() > 50]

    X_train = X_train.drop(columns=high_card, errors="ignore")
    X_val = X_val.drop(columns=high_card, errors="ignore")
    X_test = X_test.drop(columns=high_card, errors="ignore")

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_val = pd.get_dummies(X_val, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_test = pd.get_dummies(X_test, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    # ==============================
    # 9. FEATURE SELECTION
    # ==============================
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=200, random_state=42),
        threshold="median"
    )

    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.get_support()]

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    # Final safety net
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    # ==============================
    # 10. OPTIONAL CHECK MODEL
    # ==============================
    if run_model:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", BalancedRandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]

        print(f"\n=== RESULTS FOR {target_col} ===")
        print("Features used:", X_train.shape[1])
        print("Validation ROC-AUC:", roc_auc_score(y_val, val_probs))

        importances = pd.Series(
            model.named_steps["clf"].feature_importances_,
            index=X_train.columns
        )

        print("\nTop features:")
        print(importances.sort_values(ascending=False).head(10))

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================
# DEFAULT OUTPUT FOR IMPORTS
# ==============================
DEFAULT_TARGET = "Target_At_Least_2"

X_train, X_val, X_test, y_train, y_val, y_test = build_model_dataset(
    target_col=DEFAULT_TARGET,
    run_model=True
)


def get_data(target_col=DEFAULT_TARGET):
    return build_model_dataset(target_col=target_col, run_model=False)
