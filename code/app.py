import os
import json
import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="NIFTY 50 Cross-Market AI", layout="wide")
st.title("NIFTY 50 Direction Predictor & Cross-Market Signal Explainability")

# Connect to DagsHub MLflow (works locally via .env and on Streamlit Cloud via st.secrets)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or st.secrets.get("MLFLOW_TRACKING_URI")
if os.getenv("MLFLOW_TRACKING_USERNAME"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
elif "MLFLOW_TRACKING_USERNAME" in st.secrets:
    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]

mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.get_experiment_by_name("NIFTY50_CrossMarket_Direction")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if runs.empty:
    st.warning("No MLflow runs found yet. Run train_pipeline.py first.")
    st.stop()

# Keep latest run per model name so the leaderboard stays clean across daily runs
latest_runs = runs.sort_values("start_time", ascending=False).drop_duplicates(subset=["tags.mlflow.runName"])

# 1. Model Comparison Leaderboard
st.subheader("1. Model Comparison Leaderboard (MLflow Tracked)")
cols = [
    "tags.mlflow.runName",
    "metrics.cv_roc_auc",
    "metrics.train_accuracy",
    "metrics.test_accuracy",
    "metrics.test_f1",
    "metrics.test_roc_auc",
    "start_time"
]
leaderboard = latest_runs[cols].sort_values("metrics.test_roc_auc", ascending=False)
st.dataframe(leaderboard, use_container_width=True)

best_row = leaderboard.iloc[0]
best_run_id = latest_runs.loc[leaderboard.index[0], "run_id"]
best_model_name = best_row["tags.mlflow.runName"]

st.success(
    f"Champion Model: **{best_model_name}** | "
    f"Out-of-Sample Accuracy: **{best_row['metrics.test_accuracy']:.2%}** | "
    f"Test ROC-AUC: **{best_row['metrics.test_roc_auc']:.4f}**"
)

# Download SHAP & Market Signal artifacts from Champion Run
client = mlflow.tracking.MlflowClient()
local_dir = client.download_artifacts(best_run_id, "explainability")

with open(os.path.join(local_dir, "market_signal_ranking.json")) as f:
    market_signals = json.load(f)

feat_imp_df = pd.read_csv(os.path.join(local_dir, "shap_feature_importance.csv"))
latest_exp_df = pd.read_csv(os.path.join(local_dir, "latest_prediction_explanation.csv"))

# 2 & 3. International Market Predictive Power & Top Global Features
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Strongest International Market Signals")
    mkt_df = pd.DataFrame({
        "Market Region": list(market_signals.keys()),
        "Cumulative SHAP Predictive Power": list(market_signals.values())
    }).sort_values("Cumulative SHAP Predictive Power", ascending=True)
    fig_mkt = px.bar(mkt_df, x="Cumulative SHAP Predictive Power", y="Market Region", orientation="h")
    st.plotly_chart(fig_mkt, use_container_width=True)

with col2:
    st.subheader("3. Top Cross-Market Features (Global SHAP)")
    top_feats = feat_imp_df.head(12).sort_values("mean_abs_shap", ascending=True)
    fig_feat = px.bar(top_feats, x="mean_abs_shap", y="feature", orientation="h")
    st.plotly_chart(fig_feat, use_container_width=True)

# 4. Individual Prediction Factor Breakdown
st.subheader("4. Factors Driving the Latest NIFTY 50 Prediction")
top_local = latest_exp_df.head(10).copy()
top_local["Direction Push"] = top_local["shap_impact"].apply(lambda v: "Bullish (UP)" if v > 0 else "Bearish (DOWN)")
fig_local = px.bar(
    top_local.sort_values("shap_impact"),
    x="shap_impact",
    y="feature",
    color="Direction Push",
    color_discrete_map={"Bullish (UP)": "#00CC96", "Bearish (DOWN)": "#EF553B"},
    orientation="h"
)
st.plotly_chart(fig_local, use_container_width=True)