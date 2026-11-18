import os
import json
import requests
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from pathlib import Path

load_dotenv()

# 1. Connect to Remote MLflow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("NIFTY50_CrossMarket_Direction")

# 2. Load & Clean Preprocessed Features
# Handles running either from project root or from inside Forecasting/code/

BASE_DIR = Path(__file__).resolve().parent.parent  # Points to C:\Users\muhda\Desktop\Forecasting
DATA_PATH = BASE_DIR / "data" / "preprocess" / "features" / "nifty50_premarket_features.csv"

df = pd.read_csv(DATA_PATH)
date_col = [c for c in df.columns if "date" in c.lower()][0]
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

target_col = "target_up"
feature_cols = [c for c in df.columns if c not in [date_col, target_col]]

clean_df = df[[date_col] + feature_cols + [target_col]].dropna().reset_index(drop=True)
X = clean_df[feature_cols]
y = clean_df[target_col].astype(int)

# Chronological 80/20 Split
split_idx = int(len(clean_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
tscv = TimeSeriesSplit(n_splits=5)

# 3. Candidate Models (Regularized for 143 correlated cross-market features)
models = {
    "Baseline_LogReg_L2": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.5, random_state=42))
    ]),
    "Sparse_LogReg_ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            C=0.05, max_iter=2000, random_state=42
        ))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=15,
        max_features="sqrt", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.5, min_child_weight=15,
        reg_alpha=1.0, reg_lambda=5.0, eval_metric="logloss", random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=250, max_depth=3, num_leaves=7, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.5, min_child_samples=20,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300, depth=4, learning_rate=0.02,
        l2_leaf_reg=10.0, random_seed=42, verbose=0
    )
}

best_auc = -1.0
best_model_name = None
best_model_obj = None
best_run_id = None
leaderboard = []

# 4. Train, Cross-Validate, Evaluate & Log Every Model to MLflow
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        # Walk-forward CV on training window
        cv_aucs = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc")
        cv_auc_mean = float(np.mean(cv_aucs))

        # Fit on full training window
        model.fit(X_train, y_train)

        # Train metrics
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        train_acc = accuracy_score(y_train, train_preds)
        train_auc = roc_auc_score(y_train, train_probs)
        train_loss = log_loss(y_train, train_probs)

        # Holdout Test metrics
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_auc = roc_auc_score(y_test, test_probs)
        test_loss = log_loss(y_test, test_probs)

        # Log parameters & metrics to MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        mlflow.log_metric("cv_roc_auc", cv_auc_mean)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_roc_auc", train_auc)
        mlflow.log_metric("train_log_loss", train_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", test_auc)
        mlflow.log_metric("test_log_loss", test_loss)

        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

        leaderboard.append({
            "Model": name,
            "CV_AUC": round(cv_auc_mean, 4),
            "Train_Acc": round(train_acc, 4),
            "Test_Acc": round(test_acc, 4),
            "Test_F1": round(test_f1, 4),
            "Test_AUC": round(test_auc, 4)
        })

        if test_auc > best_auc:
            best_auc = test_auc
            best_model_name = name
            best_model_obj = model
            best_run_id = run.info.run_id

# 5. Cross-Market Signal Ranking & SHAP Explainability on Champion Model
MARKET_GROUPS = {
    "US Markets (S&P 500 / NASDAQ / Dow)": ["sp500_", "nasdaq_", "dow_jones_"],
    "European Markets (FTSE 100 / DAX / CAC 40)": ["ftse100_", "dax_", "cac40_"],
    "Asian Markets (Nikkei / Hang Seng / Shanghai)": ["nikkei225_", "hang_seng_", "shanghai_"],
    "Indian Indices (NIFTY 50 / SENSEX)": ["nifty50_", "sensex_"],
    "Macro & Volatility (India VIX / USD-INR)": ["india_vix_", "usd_inr_"]
}

if best_model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
    explainer = shap.TreeExplainer(best_model_obj)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
else:
    bg = shap.sample(X_train, min(100, len(X_train)), random_state=42)
    explainer = shap.Explainer(best_model_obj.predict_proba, bg)
    shap_vals = explainer(X_test).values[:, :, 1]

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

market_signal_scores = {}
for group_name, prefixes in MARKET_GROUPS.items():
    mask = feature_importance_df["feature"].apply(lambda f: any(f.startswith(p) for p in prefixes))
    market_signal_scores[group_name] = float(feature_importance_df.loc[mask, "mean_abs_shap"].sum())

market_ranking = sorted(market_signal_scores.items(), key=lambda x: x[1], reverse=True)

# Explain the most recent row's prediction
latest_X = X.iloc[[-1]]
latest_date = str(clean_df[date_col].iloc[-1].date())
latest_prob_up = float(best_model_obj.predict_proba(latest_X)[0, 1])
latest_direction = "UP 🟢" if latest_prob_up >= 0.5 else "DOWN 🔴"

if best_model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
    latest_shap = explainer.shap_values(latest_X)
    if isinstance(latest_shap, list):
        latest_shap = latest_shap[1][0]
    elif latest_shap.ndim == 3:
        latest_shap = latest_shap[0, :, 1]
    else:
        latest_shap = latest_shap[0]
else:
    latest_shap = explainer(latest_X).values[0, :, 1]

latest_drivers = pd.DataFrame({
    "feature": feature_cols,
    "value": latest_X.values[0],
    "shap_impact": latest_shap
}).reindex(np.argsort(np.abs(latest_shap))[::-1])

# Log explainability artifacts to the Champion MLflow Run
with mlflow.start_run(run_id=best_run_id):
    feature_importance_df.to_csv("shap_feature_importance.csv", index=False)
    latest_drivers.to_csv("latest_prediction_explanation.csv", index=False)
    with open("market_signal_ranking.json", "w") as f:
        json.dump(dict(market_ranking), f, indent=2)
    mlflow.log_artifact("shap_feature_importance.csv", artifact_path="explainability")
    mlflow.log_artifact("latest_prediction_explanation.csv", artifact_path="explainability")
    mlflow.log_artifact("market_signal_ranking.json", artifact_path="explainability")

# Clean up temporary local files
for tmp in ["shap_feature_importance.csv", "latest_prediction_explanation.csv", "market_signal_ranking.json"]:
    if os.path.exists(tmp):
        os.remove(tmp)

# 6. Send Telegram Notification
top_market = market_ranking[0][0]
top_3_drivers_text = "\n".join([
    f"  • `{row.feature}` = {row.value:.4f} (SHAP: {row.shap_impact:+.4f})"
    for _, row in latest_drivers.head(3).iterrows()
])

best_row = [r for r in leaderboard if r["Model"] == best_model_name][0]
msg = (
    f"📊 *NIFTY 50 Pre-Market Forecast ({latest_date})*\n"
    f"• *Predicted Direction:* {latest_direction} (Prob UP: `{latest_prob_up:.1%}`)\n"
    f"• *Champion Model:* `{best_model_name}` (Test Acc: `{best_row['Test_Acc']:.1%}`, AUC: `{best_row['Test_AUC']:.3f}`)\n"
    f"• *Strongest Global Signal:* {top_market}\n\n"
    f"🔍 *Top Factors Driving Signal:*\n{top_3_drivers_text}"
)

resp = requests.post(
    f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
    json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": msg, "parse_mode": "Markdown"}
)

print("\n=== MODEL COMPARISON LEADERBOARD ===")
print(pd.DataFrame(leaderboard).sort_values("Test_AUC", ascending=False).to_string(index=False))
print(f"\nChampion Model: {best_model_name} (Test ROC-AUC: {best_auc:.4f})")
print("\n=== INTERNATIONAL MARKET PREDICTIVE RANKING (SHAP) ===")
for mkt, score in market_ranking:
    print(f"  {mkt}: {score:.4f}")
print("\nTelegram Status:", resp.status_code)