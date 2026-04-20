# import necessary packages 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# get prepared data from data_prep .py file 
from data_prep import get_data

# =========================
# HELPER FUNCTIONS
# =========================
def get_metrics(y_true, y_pred, y_prob):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-10
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Actual 0', 'Actual 1']
    )
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_brf_feature_importance(model, X_train, title):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df.head(20),
        hue='Feature',
        palette='viridis',
        legend=False
    )
    plt.title(title)
    plt.xlabel('Gini Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    print("\n--- Feature Importance Ranking ---")
    print(feat_imp_df.head(20))

    return feat_imp_df


def run_shap(best_xgb, X_test, target_name):
    print(f"\n=== SHAP ANALYSIS: {target_name} ===")

    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary Plot ({target_name})")
    plt.tight_layout()
    plt.show()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance (Bar) - {target_name}")
    plt.tight_layout()
    plt.show()

    # Dependence plots for top features if present
    top_features = [
        "not_covid_vaccinated",
        "AGE_18_44_RATIO",
        "vax_hesitancy_interaction"
    ]

    for feat in top_features:
        if feat in X_test.columns:
            plt.figure()
            shap.dependence_plot(feat, shap_values, X_test, show=False)
            plt.title(f"SHAP Dependence Plot: {feat} ({target_name})")
            plt.tight_layout()
            plt.show()

    # Waterfall plot for first test example
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test.iloc[0],
            feature_names=X_test.columns.tolist()
        )
    )
    plt.show()

    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df.to_csv(f"shap_values_{target_name}.csv", index=False)
    print(f"Saved SHAP values to shap_values_{target_name}.csv")


def plot_precision_recall_curve(y_true, y_probs, label):
    precisions, recalls, _ = precision_recall_curve(y_true, y_probs)

    plt.plot(recalls, precisions, label=label)

# =========================
# MAIN MODELING FUNCTION
# =========================
def run_model_pipeline(target_col):
    print("\n" + "=" * 80)
    print(f"RUNNING MODELS FOR: {target_col}")
    print("=" * 80)

    X_train, X_val, X_test, y_train, y_val, y_test = get_data(target_col=target_col)

    # ==========================================
    # 1. LOGISTIC REGRESSION
    # ==========================================
    print("\n--- Tuning Logistic Regression ---")
    log_reg_model = LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        random_state=42
    )

    log_reg_param_dist = {
        'C': uniform(loc=0.001, scale=10),
        'solver': ['saga'],
        'l1_ratio': uniform(loc=0, scale=1),
        'penalty': ['elasticnet']
    }

    log_reg_search = RandomizedSearchCV(
        log_reg_model,
        param_distributions=log_reg_param_dist,
        n_iter=20,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    log_reg_search.fit(X_train, y_train)

    best_log_reg_model = log_reg_search.best_estimator_
    y_val_prob_log_reg = best_log_reg_model.predict_proba(X_val)[:, 1]
    y_val_pred_log_reg = (y_val_prob_log_reg >= 0.3).astype(int)

    log_reg_metrics = get_metrics(y_val, y_val_pred_log_reg, y_val_prob_log_reg)

    print("Best Logistic Regression Params:", log_reg_search.best_params_)
    print("Logistic Regression Validation Metrics:", log_reg_metrics)

    # ==========================================
    # 2. BALANCED RANDOM FOREST
    # ==========================================
    print("\n--- Tuning Balanced Random Forest ---")
    brf_param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }

    brf_model = BalancedRandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        sampling_strategy='auto'
    )

    brf_search = RandomizedSearchCV(
        brf_model,
        param_distributions=brf_param_dist,
        n_iter=20,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    brf_search.fit(X_train, y_train)

    best_brf_model = brf_search.best_estimator_
    y_val_prob_brf = best_brf_model.predict_proba(X_val)[:, 1]
    y_val_pred_brf = (y_val_prob_brf >= 0.3).astype(int)

    brf_metrics = get_metrics(y_val, y_val_pred_brf, y_val_prob_brf)

    print("Best Balanced Random Forest Params:", brf_search.best_params_)
    print("Balanced Random Forest Validation Metrics:", brf_metrics)

        # ==========================================
    # 3. XGBOOST
    # ==========================================
    print("\n--- Tuning XGBoost ---")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    xgb_param_dist = {
        'n_estimators': randint(200, 600),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.1),
        'max_delta_step': randint(1, 5),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }

    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        ),
        param_distributions=xgb_param_dist,
        n_iter=25,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)

    best_xgb = xgb_search.best_estimator_

    # ------------------------------------------
    # VALIDATION: find optimal threshold
    # ------------------------------------------
    y_val_probs_xgb = best_xgb.predict_proba(X_val)[:, 1]
    opt_threshold, best_f1 = find_best_threshold(y_val, y_val_probs_xgb)
    y_val_pred_xgb = (y_val_probs_xgb >= opt_threshold).astype(int)

    xgb_metrics = get_metrics(y_val, y_val_pred_xgb, y_val_probs_xgb)

    print("Best XGBoost Params:", xgb_search.best_params_)
    print(f"Optimal XGBoost Threshold: {opt_threshold:.4f}")
    print(f"Best Validation F1 at Optimal Threshold: {best_f1:.4f}")
    print("XGBoost Validation Metrics:", xgb_metrics)

    # ==========================================
    # COMPARISON TABLE
    # ==========================================
    results_df_tuned = pd.DataFrame(
        [log_reg_metrics, brf_metrics, xgb_metrics],
        index=[
            f"Logistic Regression ({target_col})",
            f"Balanced Random Forest ({target_col})",
            f"XGBoost ({target_col})"
        ]
    )

    print("\n=== Tuned Model Comparison (Validation Set) ===")
    print(results_df_tuned.round(4))

    # ==========================================
    # TEST SET EVALUATION FOR BEST XGBOOST
    # ==========================================
    y_test_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
    y_test_pred_xgb = (y_test_prob_xgb >= opt_threshold).astype(int)

    print(f"\n--- XGBoost Results (Test Set) | {target_col} ---")
    print(classification_report(y_test, y_test_pred_xgb, zero_division=0))
    print("ROC-AUC (Test Set):", roc_auc_score(y_test, y_test_prob_xgb))

    plot_conf_matrix(
        y_test,
        y_test_pred_xgb,
        f'Confusion Matrix for XGBoost ({target_col})'
    )

    feat_imp_df = plot_brf_feature_importance(
        best_brf_model,
        X_train,
        f'Feature Importance: Balanced Random Forest ({target_col})'
    )

    run_shap(best_xgb, X_test, target_col)

    return {
        "target": target_col,
        "results_df": results_df_tuned,
        "best_xgb_model": best_xgb,
        "best_xgb_threshold": opt_threshold,
        "feature_importance_df": feat_imp_df,
        "y_test": y_test,
        "y_test_probs": y_test_prob_xgb
    }


# =========================
# RUN BOTH TARGETS
# =========================
all_results = {}

for target in ["Target_At_Least_1", "Target_At_Least_2"]:
    all_results[target] = run_model_pipeline(target)

# =========================
# FINAL SUMMARY TABLE
# =========================
summary_rows = []

for target, result in all_results.items():
    temp_df = result["results_df"].copy()
    temp_df["Target"] = target
    temp_df["Model"] = temp_df.index
    summary_rows.append(temp_df.reset_index(drop=True))

final_summary = pd.concat(summary_rows, ignore_index=True)

print("\n" + "=" * 80)
print("FINAL SUMMARY ACROSS BOTH TARGETS")
print("=" * 80)
print(final_summary[["Target", "Model", "Precision", "Recall", "F1", "ROC-AUC"]].round(4))

final_summary.to_csv("model_comparison_both_targets.csv", index=False)
print("\nSaved comparison table to model_comparison_both_targets.csv")

# =========================
# PRECISION-RECALL CURVES
# =========================
plt.figure(figsize=(8, 6))

for target, result in all_results.items():
    y_test = result["y_test"]
    y_probs = result["y_test_probs"]

    plot_precision_recall_curve(
        y_test,
        y_probs,
        label=target
    )

    # Add baseline (class balance)
    baseline = y_test.mean()
    plt.hlines(
        baseline,
        0,
        1,
        linestyles='dashed',
        label=f"{target} baseline"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
