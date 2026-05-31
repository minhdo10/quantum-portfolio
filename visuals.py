import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def get_theme_colors():
    theme = st.session_state.get("theme", "Dark")
    if theme == "Light":
        return "#374151", "#111827", "#E5E7EB" # secondary, primary, grid
    else:
        return "#9CA3AF", "#F9FAFB", "rgba(255,255,255,0.1)"

def plot_correlation_heatmap(corr_matrix):
    tc_sec, tc_pri, grid_col = get_theme_colors()
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale="Mint",
        aspect="auto",
        title="Asset Correlation Heatmap"
    )
    fig.update_traces(textfont=dict(family="'Inter', sans-serif", size=14, color=tc_sec))
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=20, color=tc_pri, family="'Inter', sans-serif")
    )
    return fig

def plot_allocation_donut(weights_dict):
    df = pd.DataFrame(list(weights_dict.items()), columns=["Ticker", "Weight"])
    df = df[df["Weight"] > 0]
    
    if df.empty:
        # Edge case: all zero
        df = pd.DataFrame([{"Ticker": "Cash/Null", "Weight": 1.0}])

    alpha_colors = ["#8B5CF6", "#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#14B8A6"]

    tc_sec, tc_pri, grid_col = get_theme_colors()
    
    fig = px.pie(
        df, 
        values="Weight", 
        names="Ticker", 
        hole=0.7,
        title="Optimal Asset Allocation",
        color_discrete_sequence=alpha_colors
    )
    fig.update_traces(
        textinfo='percent+label',
        textfont_size=15,
        textfont_color=tc_sec,
        hoverinfo='label+percent',
        marker=dict(line=dict(color='rgba(0,0,0,0)', width=2))
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=20, color=tc_pri, family="'Inter', sans-serif")
    )
    return fig

import numpy as np
import plotly.graph_objects as go

def plot_efficient_frontier(returns_df, optimal_weights_dict):
    np.random.seed(42)
    num_portfolios = 500
    all_weights = np.zeros((num_portfolios, len(returns_df.columns)))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    
    mean_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252
    
    for i in range(num_portfolios):
        w = np.random.random(len(returns_df.columns))
        w /= np.sum(w)
        all_weights[i,:] = w
        ret_arr[i] = np.sum(mean_returns * w)
        vol_arr[i] = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        
    opt_w = np.array([optimal_weights_dict.get(t, 0) for t in returns_df.columns])
    opt_ret = np.sum(mean_returns * opt_w)
    opt_vol = np.sqrt(np.dot(opt_w.T, np.dot(cov_matrix, opt_w)))
    
    tc_sec, tc_pri, grid_col = get_theme_colors()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vol_arr, y=ret_arr, mode='markers',
        marker=dict(
            color=ret_arr/vol_arr, 
            colorscale='Viridis', 
            showscale=True, 
            colorbar=dict(title=dict(text="Sharpe", font=dict(color=tc_sec, family="'Inter', sans-serif")), tickfont=dict(color=tc_sec, family="'Inter', sans-serif")),
            size=7, 
            opacity=0.5,
            line=dict(width=0)
        ),
        name="Random Portfolios"
    ))
    
    fig.add_trace(go.Scatter(
        x=[opt_vol], y=[opt_ret], mode='markers+text',
        marker=dict(color='#8B5CF6', size=24, symbol='circle', line=dict(width=3, color='rgba(0,0,0,0)')),
        text=["Axiom Optimum"],
        textposition="top center",
        textfont=dict(color='#8B5CF6', size=16, family="'Inter', sans-serif", weight='bold'),
        name="Optimum"
    ))
    
    fig.update_layout(
        title="Axiom Edge: The Efficient Frontier",
        xaxis_title="Risk (Volatility)",
        yaxis_title="Expected Return",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=20, color=tc_pri, family="'Inter', sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        margin=dict(r=20)
    )
    return fig

def plot_cumulative_performance(returns_df, optimal_weights_dict):
    tickers = list(optimal_weights_dict.keys())
    # Filter returns_df down to only the tickers present in the weights dict
    returns_df = returns_df[tickers]
    
    # Check if returns_df is empty or missing data
    if returns_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Not enough data for Backtesting.")
        return fig
    
    w_opt = np.array([optimal_weights_dict[t] for t in tickers])
    w_eq = np.ones(len(tickers)) / len(tickers)
    
    # Calculate daily returns
    port_opt_daily = returns_df.dot(w_opt)
    port_eq_daily = returns_df.dot(w_eq)
    
    # Calculate cumulative returns
    cum_opt = (1 + port_opt_daily).cumprod()
    cum_eq = (1 + port_eq_daily).cumprod()
    
    # Normalize to start at 1
    if len(cum_opt) > 0:
        cum_opt = cum_opt / cum_opt.iloc[0]
        cum_eq = cum_eq / cum_eq.iloc[0]
    
    tc_sec, tc_pri, grid_col = get_theme_colors()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cum_eq.index, y=cum_eq, 
        mode='lines', 
        name='Equal Weight Benchmark', 
        line=dict(color=tc_sec, width=2)
    ))
    fig.add_trace(go.Scatter(
        x=cum_opt.index, y=cum_opt, 
        mode='lines', 
        name='Quantum Optimized', 
        line=dict(color='#10B981', width=3)
    ))
    
    max_val = max(cum_opt.max(), cum_eq.max()) if not cum_opt.empty else 1.5
    
    fig.update_layout(
        title="Historical Cumulative Performance",
        xaxis_title="",
        yaxis_title="Growth of initial investment",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02, 
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=tc_sec)
        ),
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=18, color=tc_pri, family="'Inter', sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False, tickformat=".2f", tickprefix="$"),
        margin=dict(t=50, l=20, r=20, b=20)
    )
    return fig

def plot_correlation_network(corr_matrix, threshold=0.3):
    tc_sec, tc_pri, grid_col = get_theme_colors()
    G = nx.Graph()
    tickers = corr_matrix.columns
    for t in tickers:
        G.add_node(t)
        
    pos_edges = []
    neg_edges = []
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(tickers[i], tickers[j], weight=corr)
                if corr > 0:
                    pos_edges.append((tickers[i], tickers[j]))
                else:
                    neg_edges.append((tickers[i], tickers[j]))
                    
    pos = nx.spring_layout(G, seed=42)
    
    fig = go.Figure()
    
    def add_edge_trace(edges, color, name):
        ex, ey = [], []
        for e in edges:
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            ex.extend([x0, x1, None])
            ey.extend([y0, y1, None])
        if ex:
            fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines', line=dict(width=2, color=color), name=name, hoverinfo='none'))
            
    add_edge_trace(pos_edges, '#10B981', 'Positive Corr')
    add_edge_trace(neg_edges, '#EF4444', 'Negative Corr')
    
    nx_vals = [pos[t][0] for t in G.nodes()]
    ny_vals = [pos[t][1] for t in G.nodes()]
    
    fig.add_trace(go.Scatter(
        x=nx_vals, y=ny_vals, mode='markers+text',
        text=list(G.nodes()), textposition="top center",
        marker=dict(size=20, color='#6366F1', line=dict(width=2, color='rgba(0,0,0,0)')),
        textfont=dict(color=tc_pri, size=14, family="'Inter', sans-serif", weight='bold'),
        name='Assets'
    ))
    
    fig.update_layout(
        title="Asset Correlation Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=20, color=tc_pri, family="'Inter', sans-serif")
    )
    return fig

def draw_quantum_circuit(circuit):
    if circuit is None:
        return None
    theme = st.session_state.get("theme", "Dark")
    text_color = "white" if theme == "Dark" else "black"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # Pylatexenc needed for mpl backend
        circuit.draw(output='mpl', ax=ax, style='clifford')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        return fig
    except Exception as e:
        return None

def plot_sparkline(price_series: pd.Series, positive: bool = True) -> go.Figure:
    """Compact inline sparkline — no axes, no hover, color-coded fill."""
    y = price_series.values
    x = list(range(len(y)))
    line_color = "#10B981" if positive else "#EF4444"
    fill_color = "rgba(16,185,129,0.10)" if positive else "rgba(239,68,68,0.10)"
    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=line_color, width=1.8),
        fill="tozeroy",
        fillcolor=fill_color,
        hoverinfo="none",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=64,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig

def plot_watchlist_detail(price_series: pd.Series, ticker: str, positive: bool = True) -> go.Figure:
    """Full price chart with date axis and hover tooltips for the watchlist expander."""
    tc_sec, tc_pri, grid_col = get_theme_colors()
    line_color = "#10B981" if positive else "#EF4444"
    fill_color = "rgba(16,185,129,0.08)" if positive else "rgba(239,68,68,0.08)"
    fig = go.Figure(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode="lines",
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=fill_color,
        name=ticker,
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{ticker} — Price History",
        margin=dict(l=10, r=10, t=40, b=10),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=14, color=tc_pri, family="'Inter', sans-serif"),
        xaxis=dict(showgrid=False, zeroline=False, color=tc_sec, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False, tickprefix="$",
                   color=tc_sec, tickfont=dict(size=10)),
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def plot_comparison_overlay(runs):
    """Overlay cumulative returns for multiple optimization runs."""
    tc_sec, tc_pri, grid_col = get_theme_colors()
    colors = ["#6366F1", "#10B981", "#F59E0B", "#EC4899"]
    fig = go.Figure()
    for i, run in enumerate(runs):
        w = run["weights"]
        rdf = run["returns_df"]
        tickers = [t for t in w if w[t] > 0]
        if not tickers:
            continue
        w_arr = np.array([w[t] for t in tickers])
        daily = rdf[tickers].dot(w_arr)
        cum = (1 + daily).cumprod()
        if len(cum) > 0:
            cum = cum / cum.iloc[0]
        fig.add_trace(go.Scatter(
            x=cum.index,
            y=cum,
            mode="lines",
            name=run.get("label", f"Run {i + 1}"),
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))
    fig.update_layout(
        title="Cumulative Return Comparison",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=18, color=tc_pri, family="'Inter', sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        hovermode="x unified",
        margin=dict(r=20),
    )
    return fig


def plot_comparison_overlay(runs):
    """Overlay cumulative returns for multiple optimization runs."""
    tc_sec, tc_pri, grid_col = get_theme_colors()
    colors = ["#6366F1", "#10B981", "#F59E0B", "#EC4899"]
    fig = go.Figure()
    for i, run in enumerate(runs):
        w = run["weights"]
        rdf = run["returns_df"]
        tickers = [t for t in w if w[t] > 0]
        if not tickers:
            continue
        w_arr = np.array([w[t] for t in tickers])
        daily = rdf[tickers].dot(w_arr)
        cum = (1 + daily).cumprod()
        if len(cum) > 0:
            cum = cum / cum.iloc[0]
        fig.add_trace(go.Scatter(
            x=cum.index,
            y=cum,
            mode="lines",
            name=run.get("label", f"Run {i + 1}"),
            line=dict(color=colors[i % len(colors)], width=2.5),
        ))
    fig.update_layout(
        title="Cumulative Return Comparison",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=18, color=tc_pri, family="'Inter', sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        hovermode="x unified",
        margin=dict(r=20),
    )
    return fig



def plot_regime_chart(regime_df) -> "go.Figure":
    """
    Area chart showing portfolio cumulative growth colour-coded by
    HMM-detected market regime (Bull / Sideways / Bear).
    """
    import plotly.graph_objects as go
    tc_sec, tc_pri, grid_col = get_theme_colors()
    regime_colors = {"Bull": "#10B981", "Sideways": "#F59E0B", "Bear": "#EF4444"}

    fig = go.Figure()

    prev_regime = None
    band_start = None
    bands = []
    for _, row in regime_df.iterrows():
        if row["regime_label"] != prev_regime:
            if prev_regime is not None:
                bands.append((band_start, row["date"], prev_regime))
            band_start = row["date"]
            prev_regime = row["regime_label"]
    if prev_regime is not None and band_start is not None:
        bands.append((band_start, regime_df["date"].iloc[-1], prev_regime))

    for start, end, label in bands:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=regime_colors.get(label, "#9CA3AF"),
            opacity=0.12, layer="below", line_width=0,
        )

    cum = (1 + regime_df["portfolio_return"]).cumprod()
    if len(cum) > 0:
        cum = cum / cum.iloc[0]

    fig.add_trace(go.Scatter(
        x=regime_df["date"], y=cum, mode="lines", name="Portfolio",
        line=dict(color="#6366F1", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        hovertemplate="<b>%{x|%b %d %Y}</b><br>Growth: %{y:.3f}<extra></extra>",
    ))

    for label, color in regime_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=color, symbol="square"),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        title="Market Regime Detection (HMM)", xaxis_title="", yaxis_title="Portfolio Growth",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=20, color=tc_pri, family="'Inter', sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=tc_sec)),
        xaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=False, tickformat=".2f"),
        hovermode="x unified", margin=dict(t=60, l=20, r=20, b=20),
    )
    return fig


def plot_benchmark_comparison(quantum_metrics: dict, classical_metrics: dict) -> "go.Figure":
    """
    Grouped bar chart comparing Quantum vs Classical optimizer metrics.
    """
    import plotly.graph_objects as go
    tc_sec, tc_pri, grid_col = get_theme_colors()

    metrics = ["Ann. Return (%)", "Volatility (%)", "Sharpe Ratio"]
    q_vals = [
        quantum_metrics.get("return", 0) * 100,
        quantum_metrics.get("vol", 0) * 100,
        quantum_metrics.get("sharpe", 0),
    ]
    c_vals = [
        classical_metrics.get("return", 0) * 100,
        classical_metrics.get("vol", 0) * 100,
        classical_metrics.get("sharpe", 0),
    ]

    fig = go.Figure(data=[
        go.Bar(
            name="Quantum (QAOA/VQE)", x=metrics, y=q_vals,
            marker_color="#6366F1", marker_line_width=0,
            text=[f"{v:.2f}" for v in q_vals], textposition="outside",
            textfont=dict(color=tc_pri, family="'Inter', sans-serif", size=13),
        ),
        go.Bar(
            name="Classical (Max Sharpe)", x=metrics, y=c_vals,
            marker_color="#10B981", marker_line_width=0,
            text=[f"{v:.2f}" for v in c_vals], textposition="outside",
            textfont=dict(color=tc_pri, family="'Inter', sans-serif", size=13),
        ),
    ])

    fig.update_layout(
        title="Classical vs Quantum Benchmarking",
        barmode="group", bargap=0.25, bargroupgap=0.08,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=tc_sec, family="'Inter', sans-serif"),
        title_font=dict(size=20, color=tc_pri, family="'Inter', sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=tc_sec)),
        xaxis=dict(showgrid=False, zeroline=False, color=tc_sec),
        yaxis=dict(showgrid=True, gridcolor=grid_col, zeroline=True, color=tc_sec),
        margin=dict(t=60, l=20, r=20, b=20),
    )
    return fig
