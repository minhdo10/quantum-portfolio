"""
regime_detector.py
──────────────────
Hidden Markov Model (HMM) market regime detection.
Classifies each trading day as Bull / Sideways / Bear based on the
portfolio's rolling return and volatility features.
"""

import numpy as np
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


REGIME_LABELS = {0: "Bear", 1: "Sideways", 2: "Bull"}
REGIME_COLORS = {"Bull": "#10B981", "Sideways": "#F59E0B", "Bear": "#EF4444"}


@st.cache_data(show_spinner=False, ttl=600)
def detect_regimes(returns_df: pd.DataFrame, weights: dict, n_states: int = 3) -> pd.DataFrame:
    """
    Fit a Gaussian HMM on the weighted portfolio's daily return & rolling-vol
    feature vector and return a DataFrame with columns:
        date, portfolio_return, regime_label, regime_id
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        # Graceful fallback: label everything Sideways
        port = _portfolio_series(returns_df, weights)
        df = pd.DataFrame({"date": port.index, "portfolio_return": port.values})
        df["regime_label"] = "Sideways"
        df["regime_id"] = 1
        return df

    port = _portfolio_series(returns_df, weights)
    if len(port) < 60:
        df = pd.DataFrame({"date": port.index, "portfolio_return": port.values})
        df["regime_label"] = "Sideways"
        df["regime_id"] = 1
        return df

    # Build feature matrix: [daily_return, rolling_21d_vol]
    roll_vol = port.rolling(21).std().fillna(method="bfill")
    X = np.column_stack([port.values, roll_vol.values])

    # Fit HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    try:
        model.fit(X)
        hidden_states = model.predict(X)
    except Exception:
        df = pd.DataFrame({"date": port.index, "portfolio_return": port.values})
        df["regime_label"] = "Sideways"
        df["regime_id"] = 1
        return df

    # Map HMM state IDs → semantic labels by mean return of each state
    state_means = {}
    for s in range(n_states):
        mask = hidden_states == s
        state_means[s] = port.values[mask].mean() if mask.sum() > 0 else 0.0

    # Sort states by mean return: lowest → Bear, middle → Sideways, highest → Bull
    sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])
    label_map = {}
    semantic = ["Bear", "Sideways", "Bull"]
    for rank, state_id in enumerate(sorted_states):
        label_map[state_id] = semantic[rank]

    df = pd.DataFrame({
        "date": port.index,
        "portfolio_return": port.values,
        "regime_id": hidden_states,
    })
    df["regime_label"] = df["regime_id"].map(label_map)
    return df


def _portfolio_series(returns_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted portfolio daily return series."""
    tickers = [t for t in weights if weights[t] > 0 and t in returns_df.columns]
    if not tickers:
        return returns_df.mean(axis=1)
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    return returns_df[tickers].dot(w)


def regime_summary(regime_df: pd.DataFrame) -> dict:
    """Return count, avg return, and avg duration for each regime."""
    summary = {}
    for label in ["Bull", "Sideways", "Bear"]:
        mask = regime_df["regime_label"] == label
        sub = regime_df[mask]
        summary[label] = {
            "days": int(mask.sum()),
            "avg_daily_return": float(sub["portfolio_return"].mean()) if len(sub) else 0.0,
            "pct_of_time": float(mask.sum() / max(len(regime_df), 1)),
        }
    return summary
