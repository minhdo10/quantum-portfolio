import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data(show_spinner=False)
def fetch_data(tickers, period="1y"):
    try:
        # yfinance returns a multindex columns DataFrame when there's more than 1 ticker
        data = yf.download(tickers, period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        returns = data.pct_change().dropna()
        if returns.empty:
            return None, "No data found for the provided tickers."
        return returns, None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def fetch_benchmark(period="1y"):
    try:
        data = yf.download("SPY", period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        returns = data.pct_change().dropna()
        return returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]
    except:
        return pd.Series(dtype=float)

def compute_metrics(weights_dict, returns, risk_free_rate=0.01):
    tickers = list(weights_dict.keys())
    weights_array = np.array([weights_dict[t] for t in tickers])
    
    mean_returns = returns[tickers].mean().values
    cov_matrix = returns[tickers].cov().values

    # Annualize
    portfolio_return = np.dot(weights_array, mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix * 252, weights_array)))

    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0

    return portfolio_return, portfolio_volatility, sharpe_ratio

@st.cache_data(show_spinner=False)
def compute_risk_metrics(weights_dict, returns, benchmark_returns=None, risk_free_rate=0.01, confidence_level=0.05):
    tickers = list(weights_dict.keys())
    weights_array = np.array([weights_dict[t] for t in tickers])
    
    port_returns = returns[tickers].dot(weights_array)
    clean_returns = port_returns.dropna()
    
    # VaR & CVaR
    var_95 = np.percentile(clean_returns, confidence_level * 100)
    cvar_95 = clean_returns[clean_returns <= var_95].mean()
    if np.isnan(cvar_95):
        cvar_95 = var_95
        
    # Sortino Ratio
    downside_returns = clean_returns[clean_returns < 0]
    downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    ann_return = clean_returns.mean() * 252
    sortino = (ann_return - risk_free_rate) / downside_dev if downside_dev != 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + clean_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    # Alpha & Beta
    alpha, beta = 0.0, 1.0
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align dates
        aligned = pd.concat([clean_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 1:
            p_ret = aligned.iloc[:, 0]
            b_ret = aligned.iloc[:, 1]
            cov = np.cov(p_ret, b_ret)[0, 1]
            var_b = np.var(b_ret)
            if var_b != 0:
                beta = cov / var_b
                ann_b_ret = b_ret.mean() * 252
                alpha = ann_return - (risk_free_rate + beta * (ann_b_ret - risk_free_rate))
                
    return {
        "VaR": var_95,
        "CVaR": cvar_95,
        "Sortino": sortino,
        "Max_DD": max_dd,
        "Alpha": alpha,
        "Beta": beta
    }

# ── Watchlist helpers ─────────────────────────────────────────────────────────

def fmt_price(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"${v:,.2f}"

def fmt_change(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—", "neutral"
    sign = "+" if v >= 0 else ""
    color = "up" if v >= 0 else "down"
    return f"{sign}{v:.2f}%", color

def fmt_volume(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    return f"{v:,.0f}"

def fmt_mktcap(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_watchlist_ticker(ticker: str, period: str = "3mo"):
    """Fetch price history + snapshot stats for a single ticker. Cached 5 min."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return None
        info = t.fast_info
        close = hist["Close"]
        prev_close = close.iloc[-2] if len(close) >= 2 else close.iloc[-1]
        last_price = float(close.iloc[-1])
        change_1d_pct = (last_price - float(prev_close)) / float(prev_close) * 100
        period_start = float(close.iloc[0])
        change_period_pct = (last_price - period_start) / period_start * 100
        volume = float(hist["Volume"].iloc[-1])
        try:
            mkt_cap = float(info.market_cap)
        except Exception:
            mkt_cap = float("nan")
        try:
            high_52w = float(info.year_high)
            low_52w  = float(info.year_low)
        except Exception:
            high_52w = float(close.max())
            low_52w  = float(close.min())
        return {
            "ticker": ticker,
            "price": last_price,
            "change_1d": change_1d_pct,
            "change_period": change_period_pct,
            "volume": volume,
            "mkt_cap": mkt_cap,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "history": close,
        }
    except Exception:
        return None

def add_ticker(ticker: str) -> tuple[bool, str]:
    """Add ticker to watchlist. Returns (success, message)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return False, "Empty ticker."
    wl = st.session_state.get("watchlist", [])
    if ticker in wl:
        return False, f"{ticker} is already in your watchlist."
    if len(wl) >= 20:
        return False, "Watchlist is full (20 ticker limit)."
    data = fetch_watchlist_ticker(ticker)
    if data is None:
        return False, f"Could not find data for {ticker}. Check the symbol."
    st.session_state.watchlist = wl + [ticker]
    return True, f"Added {ticker}."

def remove_ticker(ticker: str):
    """Remove ticker from watchlist."""
    wl = st.session_state.get("watchlist", [])
    st.session_state.watchlist = [t for t in wl if t != ticker]
