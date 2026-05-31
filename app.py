import html
import streamlit as st
import pandas as pd
import base64
import os
from datetime import datetime

# ── Logo loader ──────────────────────────────────────────────────────────────
def _logo_b64() -> str:
    logo_path = os.path.join(os.path.dirname(__file__), "axiom_logo.png")
    with open(logo_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_B64 = _logo_b64()

from data_fetcher import (
    fetch_data, compute_metrics, compute_risk_metrics, fetch_benchmark,
    fetch_watchlist_ticker, add_ticker, remove_ticker,
    fmt_price, fmt_change, fmt_volume, fmt_mktcap,
)
from quantum_engine import build_weighted_qp, solve_qaoa, solve_vqe, interpret_weights, IBM_AVAILABLE
from visuals import (
    plot_correlation_heatmap, plot_allocation_donut, plot_efficient_frontier,
    plot_cumulative_performance, plot_correlation_network,
    plot_sparkline, plot_watchlist_detail, plot_comparison_overlay,
    plot_regime_chart, plot_benchmark_comparison,
)
from regime_detector import detect_regimes, regime_summary

# ── Top 20 curated stocks ─────────────────────────────────────────────────────
TOP_20_STOCKS = {
    "NVDA":  "NVIDIA Corp",
    "AAPL":  "Apple Inc",
    "MSFT":  "Microsoft Corp",
    "GOOGL": "Alphabet Inc",
    "AMZN":  "Amazon.com Inc",
    "META":  "Meta Platforms",
    "TSLA":  "Tesla Inc",
    "AVGO":  "Broadcom Inc",
    "LLY":   "Eli Lilly & Co",
    "JPM":   "JPMorgan Chase",
    "V":     "Visa Inc",
    "UNH":   "UnitedHealth Group",
    "XOM":   "ExxonMobil Corp",
    "NFLX":  "Netflix Inc",
    "AMD":   "Advanced Micro Devices",
    "COST":  "Costco Wholesale",
    "ORCL":  "Oracle Corp",
    "WMT":   "Walmart Inc",
    "MA":    "Mastercard Inc",
    "CRM":   "Salesforce Inc",
}

STATIC_SECTOR_MAP = {
    "NVDA": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication Services",
    "AMZN": "Consumer Cyclical",
    "META": "Communication Services",
    "TSLA": "Consumer Cyclical",
    "AVGO": "Technology",
    "LLY": "Healthcare",
    "JPM": "Financial Services",
    "V": "Financial Services",
    "UNH": "Healthcare",
    "XOM": "Energy",
    "NFLX": "Communication Services",
    "AMD": "Technology",
    "COST": "Consumer Defensive",
    "ORCL": "Technology",
    "WMT": "Consumer Defensive",
    "MA": "Financial Services",
    "CRM": "Technology"
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Axiom",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "logged_in": False,
    "user_name": "",
    "user_email": "",
    "weights": None,
    "returns_df": None,
    "cov_matrix": None,
    "benchmark_df": None,
    "opt_circ": None,
    "valid_tickers": None,
    "last_run_ts": None,
    "theme": "Light",
    "watchlist": ["AAPL", "NVDA", "MSFT"],
    "opt_extra_tickers": [],
    "comparison_mode": False,
    "comparison_runs": [],
    "comparison_run_counter": 0,
    "pre_auth_view": "landing",
}

MAX_COMPARISON_RUNS = 4
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Auth ──────────────────────────────────────────────────────────────────────
if "users" not in st.session_state:
    st.session_state.users = {
        "demo@axiom.ai": {"password": "demo1234", "name": "Alex Chen"},
        "guest@axiom.ai": {"password": "guest123", "name": "Guest User"},
    }

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if "signup_success" not in st.session_state:
    st.session_state.signup_success = False

def toggle_auth_mode():
    st.session_state.auth_mode = "signup" if st.session_state.auth_mode == "login" else "login"

def try_login(email: str, password: str) -> bool:
    u = st.session_state.users.get(email.strip().lower())
    if u and u["password"] == password:
        st.session_state.logged_in = True
        st.session_state.user_name = u["name"]
        st.session_state.user_email = email.strip().lower()
        return True
    return False

def do_logout():
    for k, v in _defaults.items():
        if k != "theme":
            st.session_state[k] = v
def go_to_auth():
    st.session_state.pre_auth_view = "auth"


# ── Global Theme CSS ──────────────────────────────────────────────────────────
light_vars = """
    --bg-color: #F8FAFC;
    --card-bg: white;
    --card-border: #E2E8F0;
    --text-primary: #0F172A;
    --text-secondary: #64748B;
    --input-bg: white;
    --input-border: rgba(0,0,0,0.1);
    --nav-bg: rgba(255, 255, 255, 0.85);
"""
dark_vars = """
    --bg-color: #070A14;
    --card-bg: #111827;
    --card-border: rgba(255,255,255,0.1);
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --input-bg: rgba(255,255,255,0.05);
    --input-border: rgba(255,255,255,0.1);
    --nav-bg: rgba(7, 10, 20, 0.85);
"""

if st.session_state.theme == "Light":
    theme_vars = f":root {{ {light_vars} }}"
elif st.session_state.theme == "Dark":
    theme_vars = f":root {{ {dark_vars} }}"
else:
    theme_vars = f"@media (prefers-color-scheme: light) {{ :root {{ {light_vars} }} }} @media (prefers-color-scheme: dark) {{ :root {{ {dark_vars} }} }}"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

{theme_vars}

html, body, .stApp, p, h1, h2, h3, h4, h5, h6, li, label, input, button, .stMarkdown {{ 
    font-family: 'Inter', -apple-system, sans-serif !important; 
}}

html, body, .stApp {{ background: var(--bg-color) !important; transition: background 0.3s ease; }}
#MainMenu, footer, header, .stDeployButton {{ visibility: hidden !important; display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* Form inputs */
.stTextInput > div > div > input {{
    background: var(--input-bg) !important;
    border: 1.5px solid var(--input-border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 13px 16px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: rgba(99,102,241,0.8) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}}
.stTextInput label {{ color: var(--text-secondary) !important; font-size: 13px !important; font-weight: 500 !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in and st.session_state.get("pre_auth_view", "landing") == "landing":
    st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&display=swap" rel="stylesheet">
    <style>
    .block-container {{ padding: 0 !important; max-width: 100% !important; }}
    html, body, .stApp {{
        background: #05070f !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.35), transparent),
            radial-gradient(ellipse 60% 40% at 100% 50%, rgba(56,189,248,0.12), transparent),
            radial-gradient(ellipse 50% 30% at 0% 80%, rgba(139,92,246,0.15), transparent) !important;
    }}
    .axiom-landing {{ color: #e2e8f0; font-family: 'Inter', sans-serif; }}
    .axiom-landing-nav {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 20px 6vw; max-width: 1200px; margin: 0 auto;
    }}
    .axiom-landing-logo {{ display: flex; align-items: center; gap: 10px; }}
    .axiom-landing-logo img {{ width: 40px; height: 40px; border-radius: 10px; }}
    .axiom-landing-logo span {{
        font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 800;
        color: #f8fafc; letter-spacing: -0.5px;
    }}
    .axiom-landing-hero {{
        text-align: center; padding: 48px 6vw 32px; max-width: 900px; margin: 0 auto;
    }}
    .axiom-landing-badge {{
        display: inline-block; padding: 6px 14px; border-radius: 999px;
        background: rgba(99,102,241,0.15); border: 1px solid rgba(129,140,248,0.4);
        color: #a5b4fc; font-size: 12px; font-weight: 600; letter-spacing: 0.4px;
        margin-bottom: 24px;
    }}
    .axiom-landing-hero h1 {{
        font-family: 'Space Grotesk', sans-serif; font-size: clamp(36px, 5vw, 58px);
        font-weight: 800; line-height: 1.08; letter-spacing: -2px; color: #f8fafc;
        margin: 0 0 20px;
    }}
    .axiom-landing-hero h1 em {{
        font-style: normal;
        background: linear-gradient(110deg, #818cf8 0%, #38bdf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .axiom-landing-hero p {{
        font-size: 18px; line-height: 1.65; color: #94a3b8; max-width: 640px;
        margin: 0 auto 36px;
    }}
    .axiom-landing-stats {{
        display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;
        padding: 40px 6vw; max-width: 900px; margin: 0 auto;
        border-top: 1px solid rgba(255,255,255,0.06);
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .axiom-landing-stat {{
        text-align: center; min-width: 120px;
    }}
    .axiom-landing-stat strong {{
        display: block; font-family: 'Space Grotesk', sans-serif;
        font-size: 28px; font-weight: 700; color: #f8fafc;
    }}
    .axiom-landing-stat span {{ display: block; font-size: 13px; color: #64748b; margin-top: 4px; }}
    .axiom-landing-section {{
        max-width: 1100px; margin: 0 auto; padding: 64px 6vw;
    }}
    .axiom-landing-section h2 {{
        font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 700;
        color: #f8fafc; text-align: center; margin: 0 0 12px; letter-spacing: -0.5px;
    }}
    .axiom-landing-section > p.sub {{
        text-align: center; color: #64748b; font-size: 16px; margin: 0 0 40px;
    }}
    .axiom-feature-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px;
    }}
    .axiom-feature-card {{
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 28px 24px;
        transition: border-color 0.2s, transform 0.2s;
    }}
    .axiom-feature-card:hover {{
        border-color: rgba(129,140,248,0.4); transform: translateY(-2px);
    }}
    .axiom-feature-icon {{ font-size: 28px; margin-bottom: 14px; }}
    .axiom-feature-card h3 {{
        font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 700;
        color: #f1f5f9; margin: 0 0 8px;
    }}
    .axiom-feature-card p {{ font-size: 14px; line-height: 1.55; color: #94a3b8; margin: 0; }}
    .axiom-steps {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 24px; margin-top: 8px;
    }}
    .axiom-step {{
        text-align: center; padding: 24px 16px;
    }}
    .axiom-step-num {{
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white; font-weight: 700; font-size: 14px;
        display: inline-flex; align-items: center; justify-content: center;
        margin-bottom: 14px;
    }}
    .axiom-step h3 {{ font-size: 15px; font-weight: 600; color: #f1f5f9; margin: 0 0 6px; }}
    .axiom-step p {{ font-size: 13px; color: #64748b; margin: 0; line-height: 1.5; }}
    .axiom-landing-section:last-of-type {{
        padding-bottom: 96px !important;
    }}
    div[data-testid="stButton"] {{
        text-align: center !important;
    }}
    div[data-testid="stButton"] > button {{
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 14px 32px !important;
        font-size: 16px !important; font-weight: 600 !important;
        box-shadow: 0 12px 40px rgba(99,102,241,0.45) !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
        max-width: 640px !important; width: 100% !important;
        margin: 0 auto !important; display: block !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 16px 48px rgba(99,102,241,0.55) !important;
    }}
    div[data-testid="stButton"] > button p {{ color: white !important; }}
    </style>

    <div class="axiom-landing">
        <nav class="axiom-landing-nav">
            <div class="axiom-landing-logo">
                <img src="data:image/png;base64,{LOGO_B64}" alt="Axiom">
                <span>AXIOM</span>
            </div>
        </nav>
        <section class="axiom-landing-hero">
            <div class="axiom-landing-badge">⚛ Quantum Portfolio Intelligence</div>
            <h1>Invest smarter with <em>quantum optimization</em></h1>
            <p>
                Axiom combines real market data with QAOA and VQE solvers to build
                risk-aware portfolios—then tracks, compares, and visualizes every decision.
            </p>
        </section>
    </div>
    """, unsafe_allow_html=True)

    _hc1, _hc2, _hc3 = st.columns([1, 1.2, 1])
    with _hc2:
        st.button("Get Started Today", type="primary", use_container_width=True, key="landing_cta_hero", on_click=go_to_auth)

    st.markdown("""
    <div class="axiom-landing">
        <div class="axiom-landing-stats">
            <div class="axiom-landing-stat"><strong>QAOA + VQE</strong><span>Dual quantum engines</span></div>
            <div class="axiom-landing-stat"><strong>20+</strong><span>Curated top stocks</span></div>
            <div class="axiom-landing-stat"><strong>Real-time</strong><span>Watchlist & analytics</span></div>
        </div>
        <section class="axiom-landing-section">
            <h2>Everything you need to optimize</h2>
            <p class="sub">From universe selection to side-by-side run comparison—all in one workspace.</p>
            <div class="axiom-feature-grid">
                <div class="axiom-feature-card">
                    <div class="axiom-feature-icon">🚀</div>
                    <h3>Quantum Optimizer</h3>
                    <p>Pick your assets, tune risk and weight constraints, and let QAOA or VQE find optimal allocations.</p>
                </div>
                <div class="axiom-feature-card">
                    <div class="axiom-feature-icon">📊</div>
                    <h3>Portfolio Dashboard</h3>
                    <p>Sharpe, Sortino, VaR, drawdown, and efficient-frontier charts—explained with built-in guidance.</p>
                </div>
                <div class="axiom-feature-card">
                    <div class="axiom-feature-icon">⚖️</div>
                    <h3>Multi-Run Compare</h3>
                    <p>Test different algorithms and settings, then compare performance side-by-side.</p>
                </div>
                <div class="axiom-feature-card">
                    <div class="axiom-feature-icon">⭐</div>
                    <h3>Live Watchlist</h3>
                    <p>Track up to 20 symbols with sparklines, fundamentals, and one-click optimize.</p>
                </div>
            </div>
        </section>
        <section class="axiom-landing-section" style="padding-top:0;">
            <h2>How it works</h2>
            <p class="sub">Three steps from idea to optimized portfolio.</p>
            <div class="axiom-steps">
                <div class="axiom-step">
                    <div class="axiom-step-num">1</div>
                    <h3>Build your universe</h3>
                    <p>Select from top stocks or add custom tickers with live qubit budgeting.</p>
                </div>
                <div class="axiom-step">
                    <div class="axiom-step-num">2</div>
                    <h3>Run the solver</h3>
                    <p>Configure risk tolerance, max weights, and QAOA vs VQE—optionally on IBM Quantum.</p>
                </div>
                <div class="axiom-step">
                    <div class="axiom-step-num">3</div>
                    <h3>Act on insights</h3>
                    <p>Review allocations, analytics, and correlation networks to refine your strategy.</p>
                </div>
            </div>
        </section>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
    <style>
    .block-container { padding: 6vh 1rem 1rem !important; max-width: 100% !important; }
    div[data-testid="stForm"] { border: none !important; padding: 0 !important; }

    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 13px 24px !important;
        font-size: 15px !important; font-weight: 600 !important;
        width: 100%; box-shadow: 0 8px 24px rgba(99,102,241,0.35);
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #4F46E5, #4338CA) !important;
        box-shadow: 0 12px 32px rgba(99,102,241,0.5) !important;
    }
    div[data-testid="stFormSubmitButton"] > button p { color: white !important; font-weight: 600 !important; }

    /* Match the toggle button to the form submit button shape and width */
    div[data-testid="stButton"] > button {
        width: 100% !important;
        border: 1px solid var(--input-border) !important;
        background: var(--input-bg) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        padding: 13px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #6366F1 !important;
        color: #6366F1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown(f"""
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&display=swap" rel="stylesheet">
        <div style="text-align:center; margin-bottom:40px;">
            <img src="data:image/png;base64,{LOGO_B64}" style="width:80px; height:80px; border-radius:20px; margin-bottom:16px; box-shadow: 0 8px 32px rgba(99,102,241,0.25);">
            <div style="
                font-family: 'Space Grotesk', sans-serif;
                font-size: 56px;
                font-weight: 800;
                letter-spacing: -2px;
                line-height: 1;
                background: linear-gradient(110deg, #6366F1 0%, #818CF8 45%, #38BDF8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            ">AXIOM</div>
            <div style="
                font-size: 13px;
                font-weight: 500;
                color: var(--text-secondary);
                letter-spacing: 0.5px;
            ">Quantum-Powered Portfolio Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            if st.session_state.signup_success:
                st.success("Account created successfully! Please sign in.")
                st.session_state.signup_success = False

            with st.form("login_form"):
                email = st.text_input("Email", placeholder="demo@axiom.ai")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if try_login(email, password):
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            
            st.button("Don't have an account? Sign up", on_click=toggle_auth_mode, use_container_width=True)

        else:
            with st.form("signup_form"):
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email", placeholder="john@example.com")
                new_pwd = st.text_input("Password", type="password")
                confirm_pwd = st.text_input("Confirm Password", type="password")
                signup_submitted = st.form_submit_button("Create Account", use_container_width=True)

            if signup_submitted:
                if not new_name or not new_email or not new_pwd:
                    st.error("Please fill in all fields.")
                elif new_pwd != confirm_pwd:
                    st.error("Passwords do not match.")
                elif new_email.strip().lower() in st.session_state.users:
                    st.error("Email already registered.")
                else:
                    st.session_state.users[new_email.strip().lower()] = {
                        "password": new_pwd,
                        "name": new_name.strip()
                    }
                    st.session_state.signup_success = True
                    st.session_state.auth_mode = "login"
                    st.rerun()

            st.button("Already have an account? Sign in", on_click=toggle_auth_mode, use_container_width=True)

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding: 0 2rem 2rem !important; max-width: 100% !important; }

/* ── NAV WRAPPER ── */
div[data-testid="stHorizontalBlock"]:has(#top-nav-anchor) {
    position: sticky !important; top: 0; z-index: 9999;
    background: var(--nav-bg) !important;
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--card-border) !important;
    margin: 0 -2rem 2rem -2rem !important;
    padding: 16px 2rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.02);
    width: calc(100% + 4rem) !important;
}

/* ── PAGE ELEMENTS ── */
.kpi-card {
    background: var(--card-bg) !important;
    border-radius: 16px; padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
    border: 1px solid var(--card-border) !important;
    display: flex; justify-content: space-between; align-items: center;
}
.kpi-label { color: var(--text-secondary) !important; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-value { color: var(--text-primary) !important; font-size: 26px; font-weight: 700; margin-top: 4px; }

.page-title { font-size: 24px; font-weight: 700; color: var(--text-primary) !important; letter-spacing: -0.5px; }
.page-sub { color: var(--text-secondary) !important; font-size: 14px; margin-top: 2px; }

/* Dashboard Run Button - Keep original blue */
div[data-testid="stVerticalBlock"] > div:not(:first-child) .stButton > button {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; 
    font-weight: 600 !important; 
    padding: 0 24px !important;
    height: 44px !important;
    min-width: 160px !important;
}
div[data-testid="stVerticalBlock"] > div:not(:first-child) .stButton > button p { color: white !important; font-size: 15px !important; }
.panel-title { font-size: 15px; font-weight: 600; color: var(--text-primary) !important; margin-bottom: 12px; }
.panel-title-wrap { display: flex; align-items: center; gap: 6px; margin-bottom: 12px; }
.panel-title-wrap .panel-title { margin-bottom: 0 !important; }
.kpi-label-wrap { display: flex; align-items: center; gap: 5px; }
.axiom-info {
    display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; border-radius: 50%;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35);
    color: #6366F1; font-size: 10px; font-weight: 700; font-style: italic;
    font-family: Georgia, serif; cursor: help; position: relative; flex-shrink: 0;
    text-transform: none; letter-spacing: 0;
}
.axiom-info::after {
    content: attr(data-tip);
    position: absolute; bottom: calc(100% + 8px); left: 50%;
    transform: translateX(-50%);
    background: var(--card-bg); color: var(--text-primary);
    border: 1px solid var(--card-border);
    border-radius: 8px; padding: 8px 10px;
    font-size: 12px; font-weight: 400; font-style: normal;
    font-family: 'Inter', -apple-system, sans-serif;
    line-height: 1.45; text-transform: none; letter-spacing: 0;
    white-space: normal; width: max-content; max-width: 260px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    opacity: 0; visibility: hidden; pointer-events: none;
    transition: opacity 0.15s ease, visibility 0.15s ease; z-index: 10000;
}
.axiom-info:hover::after, .axiom-info:focus::after {
    opacity: 1; visibility: visible;
}
.tab-title-wrap { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }
.tab-title-wrap span { font-size: 14px; font-weight: 600; color: var(--text-primary); }
.badge-quantum { background: rgba(99,102,241,0.1); color: #6366F1; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }

/* Force Slider Color */
div[data-testid="stSlider"] > div > div > div > div { background-color: #6366F1 !important; }
.stSlider [data-testid="stThumb"] { background-color: #6366F1 !important; border: 2px solid white !important; }

/* Force dataframe text visibility in dark mode */
[data-testid="stDataFrame"] { color: var(--text-primary) !important; }

/* Spacer */
.main-content-area { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.dialog("User Settings")
def show_settings_dialog():
    st.markdown("### Appearance")
    theme_opts = ["Light", "Dark", "System Match"]
    current = st.session_state.theme
    idx = theme_opts.index(current) if current in theme_opts else 1
    new_theme = st.radio("Theme", theme_opts, index=idx, horizontal=True)
    
    st.markdown("### Notifications")
    st.checkbox("Email Performance Reports", value=True)
    st.checkbox("SMS Trade Alerts", value=False)
    st.markdown("### Privacy")
    st.checkbox("Share Anonymous Usage Data", value=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Save Changes", type="primary", use_container_width=True):
        st.session_state.theme = new_theme
        st.rerun()

# ── BUTTON NAVIGATION ─────────────────────────────────────────────────────────
if "nav" not in st.session_state:
    st.session_state.nav = "Optimizer"
if "nav_blocked_flash" not in st.session_state:
    st.session_state.nav_blocked_flash = False
if "nav_blocked_reason" not in st.session_state:
    st.session_state.nav_blocked_reason = "optimizer"

_optimizer_ran = st.session_state.weights is not None
_compare_available = len(st.session_state.get("comparison_runs", [])) > 0
_wl_count = len(st.session_state.get("watchlist", []))
_wl_label = f"⭐ Watchlist ({_wl_count})"
_cmp_label = f"⚖️ Compare ({len(st.session_state.get('comparison_runs', []))})"

# ── Shake + grey CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes axiom-shake {
  0%,100% { transform: translateX(0); }
  15%      { transform: translateX(-6px); }
  30%      { transform: translateX(6px); }
  45%      { transform: translateX(-5px); }
  60%      { transform: translateX(5px); }
  75%      { transform: translateX(-3px); }
  90%      { transform: translateX(3px); }
}
.axiom-btn-shake { animation: axiom-shake 0.45s ease; }
.axiom-btn-locked button {
    opacity: 0.38 !important;
    cursor: not-allowed !important;
    filter: grayscale(60%) !important;
    pointer-events: auto !important;
}
</style>
""", unsafe_allow_html=True)

# logo | Optimizer | Portfolio | Analytics | Compare | Watchlist | profile
nav_cols = st.columns([1.6, 1.1, 1.1, 1.1, 1.1, 1.3, 1.7], vertical_alignment="center")

with nav_cols[0]:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px;">
        <img src="data:image/png;base64,{LOGO_B64}" style="width:36px; height:36px; border-radius:8px;">
        <span style="font-size:24px; font-weight:900; color:#1E3A8A; letter-spacing:-0.5px;">AXIOM</span>
    </div>
    """, unsafe_allow_html=True)

with nav_cols[1]:
    if st.button("🚀 Optimizer", type="primary" if st.session_state.nav == "Optimizer" else "secondary", use_container_width=True, key="nav_optimizer"):
        st.session_state.nav = "Optimizer"
        st.session_state.nav_blocked_flash = False
        st.rerun()

with nav_cols[2]:
    if not _optimizer_ran:
        st.markdown('<div class="axiom-btn-locked" id="locked-portfolio">', unsafe_allow_html=True)
    _portfolio_clicked = st.button("📊 Portfolio", type="primary" if st.session_state.nav == "Portfolio" else "secondary", use_container_width=True, key="nav_portfolio")
    if not _optimizer_ran:
        st.markdown('</div>', unsafe_allow_html=True)
    if _portfolio_clicked:
        if _optimizer_ran:
            st.session_state.nav = "Portfolio"
            st.session_state.nav_blocked_flash = False
            st.rerun()
        else:
            st.session_state.nav_blocked_flash = True
            st.session_state.nav_blocked_reason = "optimizer"
            st.rerun()

with nav_cols[3]:
    if not _optimizer_ran:
        st.markdown('<div class="axiom-btn-locked" id="locked-analytics">', unsafe_allow_html=True)
    _analytics_clicked = st.button("📈 Analytics", type="primary" if st.session_state.nav == "Analytics" else "secondary", use_container_width=True, key="nav_analytics")
    if not _optimizer_ran:
        st.markdown('</div>', unsafe_allow_html=True)
    if _analytics_clicked:
        if _optimizer_ran:
            st.session_state.nav = "Analytics"
            st.session_state.nav_blocked_flash = False
            st.rerun()
        else:
            st.session_state.nav_blocked_flash = True
            st.session_state.nav_blocked_reason = "optimizer"
            st.rerun()

with nav_cols[4]:
    if not _compare_available:
        st.markdown('<div class="axiom-btn-locked" id="locked-compare">', unsafe_allow_html=True)
    _compare_clicked = st.button(
        _cmp_label,
        type="primary" if st.session_state.nav == "Compare" else "secondary",
        use_container_width=True,
        key="nav_compare",
    )
    if not _compare_available:
        st.markdown("</div>", unsafe_allow_html=True)
    if _compare_clicked:
        if _compare_available:
            st.session_state.nav = "Compare"
            st.session_state.nav_blocked_flash = False
            st.rerun()
        else:
            st.session_state.nav_blocked_flash = True
            st.session_state.nav_blocked_reason = "compare"
            st.rerun()

with nav_cols[5]:
    if st.button(_wl_label, type="primary" if st.session_state.nav == "Watchlist" else "secondary", use_container_width=True, key="nav_watchlist"):
        st.session_state.nav = "Watchlist"
        st.session_state.nav_blocked_flash = False
        st.rerun()

with nav_cols[6]:
    # Profile Popover
    with st.popover(f"👤 {st.session_state.user_name.split()[0]}"):
        st.write(f"Logged in as **{st.session_state.user_email}**")
        if st.button("⚙️ Settings", use_container_width=True):
            show_settings_dialog()
        if st.button("🚪 Sign out", use_container_width=True):
            do_logout()
            st.rerun()

# ── Blocked nav flash ──────────────────────────────────────────────────────────
if st.session_state.nav_blocked_flash:
    _blocked_title = "Optimizer required"
    _blocked_body = "Run the Quantum Optimizer first before accessing Portfolio or Analytics."
    if st.session_state.get("nav_blocked_reason") == "compare":
        _blocked_title = "No comparison runs yet"
        _blocked_body = (
            "Enable Multi-run comparison mode on the Optimizer, run optimizations with different settings, "
            "then open Compare."
        )
    st.markdown(f"""
    <div id="axiom-blocked-banner" style="
        background: linear-gradient(135deg,rgba(239,68,68,0.12),rgba(239,68,68,0.06));
        border: 1px solid rgba(239,68,68,0.35);
        border-left: 4px solid #EF4444;
        border-radius: 10px;
        padding: 12px 18px;
        margin: 8px 0 4px;
        display: flex; align-items: center; gap: 12px;
        animation: axiom-shake 0.45s ease;
    ">
      <span style="font-size:20px;">🔒</span>
      <div>
        <div style="font-size:14px; font-weight:700; color:#EF4444;">{_blocked_title}</div>
        <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">
          {_blocked_body}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # Inject JS to shake the locked buttons too
    st.markdown("""
    <script>
    (function() {
        function shakeLockedBtns() {
            ['locked-portfolio','locked-analytics','locked-compare'].forEach(function(id) {
                var wrapper = document.getElementById(id);
                if (!wrapper) return;
                var btn = wrapper.querySelector('button');
                if (!btn) return;
                btn.classList.remove('axiom-btn-shake');
                void btn.offsetWidth; // reflow to restart animation
                btn.classList.add('axiom-btn-shake');
                btn.addEventListener('animationend', function() {
                    btn.classList.remove('axiom-btn-shake');
                }, {once: true});
            });
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', shakeLockedBtns);
        } else {
            setTimeout(shakeLockedBtns, 80);
        }
    })();
    </script>
    """, unsafe_allow_html=True)
    st.session_state.nav_blocked_flash = False

st.markdown("<hr style='margin-top:0px; margin-bottom:20px;'>", unsafe_allow_html=True)

nav = st.session_state.nav


# ── Stat tooltips (Portfolio & Analytics) ─────────────────────────────────────
STAT_TIPS = {
    "return": "Annualized expected portfolio return, based on weighted historical daily returns.",
    "volatility": "Annualized standard deviation of portfolio returns. Higher means more price swing.",
    "sharpe": "Risk-adjusted return: extra return earned per unit of total risk. Higher is better.",
    "sortino": "Like Sharpe ratio, but only penalizes downside volatility (harmful risk).",
    "var_95": "Value at Risk (95%): typical worst daily loss on 95% of trading days.",
    "max_dd": "Maximum drawdown: largest peak-to-trough drop in cumulative portfolio value.",
    "alpha": "Excess return vs the SPY benchmark after adjusting for market exposure (beta).",
    "beta": "Market sensitivity vs SPY. 1.0 tracks the market; above 1 is more volatile.",
    "allocation": "How the quantum optimizer divides capital across each asset in your portfolio.",
    "hist_perf": "Cumulative growth of $1 for your optimized portfolio vs an equal-weight benchmark.",
    "efficient_frontier": "Risk-return map of random portfolios; purple dot is your optimized solution.",
    "advanced_insights": "How your holdings move together; clusters highlight concentration and diversification risk.",
    "correlation_network": "Assets linked when correlation is strong—clusters show diversification risk.",
    "holdings": "Per-asset weight, expected return, and volatility for each holding in the portfolio.",
    "correlation_heatmap": "Pairwise correlation between assets. Low/negative values improve diversification.",
}


def _info_icon(tip: str) -> str:
    safe = html.escape(tip, quote=True)
    return f'<span class="axiom-info" tabindex="0" data-tip="{safe}" aria-label="{safe}">i</span>'


def kpi_label_html(label: str, tip_key: str) -> str:
    return (
        f'<div class="kpi-label-wrap">'
        f'<span class="kpi-label">{html.escape(label)}</span>'
        f'{_info_icon(STAT_TIPS[tip_key])}'
        f"</div>"
    )


def panel_title_html(title: str, tip_key: str) -> str:
    return (
        f'<div class="panel-title-wrap">'
        f'<span class="panel-title">{html.escape(title)}</span>'
        f'{_info_icon(STAT_TIPS[tip_key])}'
        f"</div>"
    )


def tab_heading_html(title: str, tip_key: str) -> str:
    return (
        f'<div class="tab-title-wrap">'
        f"<span>{html.escape(title)}</span>"
        f'{_info_icon(STAT_TIPS[tip_key])}'
        f"</div>"
    )


# ── SHARED LOGIC ──────────────────────────────────────────────────────────────
def get_optimal_weights(
    mean_ret,
    cov_mat,
    r_aversion,
    m_weight,
    ibm,
    t_list,
    algo="QAOA",
    transaction_cost_penalty=0.0,
    initial_allocation=None,
    sector_weights_penalty=0.0,
    sector_map=None,
    target_sectors=None
):
    qp = build_weighted_qp(
        mean_ret,
        cov_mat,
        r_aversion,
        m_weight,
        transaction_cost_penalty=transaction_cost_penalty,
        initial_allocation=initial_allocation,
        sector_weights_penalty=sector_weights_penalty,
        sector_map=sector_map,
        target_sectors=target_sectors
    )
    if algo == "VQE":
        res, circ = solve_vqe(qp, ibm, n_assets=len(list(t_list)), max_weight=m_weight)
    else:
        res, circ = solve_qaoa(qp, ibm, n_assets=len(list(t_list)), max_weight=m_weight)
    return interpret_weights(res, list(t_list)), circ


def append_comparison_run(t_list, risk, weight, algo, ibm, weights, ret_df, bench_df):
    st.session_state.comparison_run_counter += 1
    n = st.session_state.comparison_run_counter
    ret, vol, sharpe = compute_metrics(weights, ret_df)
    risk_m = compute_risk_metrics(weights, ret_df, benchmark_returns=bench_df)
    run = {
        "id": n,
        "label": f"Run {n}",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "tickers": t_list,
        "risk": risk,
        "max_weight": weight,
        "algo": algo,
        "ibm": ibm,
        "weights": weights,
        "returns_df": ret_df,
        "metrics": {
            "return": ret,
            "vol": vol,
            "sharpe": sharpe,
            "sortino": risk_m["Sortino"],
            "max_dd": risk_m["Max_DD"],
            "var_95": risk_m["VaR"],
        },
    }
    runs = list(st.session_state.comparison_runs)
    runs.append(run)
    st.session_state.comparison_runs = runs[-MAX_COMPARISON_RUNS:]


def remove_comparison_run(run_id: int):
    st.session_state.comparison_runs = [
        r for r in st.session_state.comparison_runs if r["id"] != run_id
    ]

# ─────────────────────────────────────────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────────────────────────────────────────
if nav == "Optimizer":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <div class="page-title">Quantum Optimizer</div>
        <span class="badge-quantum">⚛ Quantum Core Active</span>
    </div>
    """, unsafe_allow_html=True)

    st.session_state.comparison_mode = st.toggle(
        "Multi-run comparison mode",
        value=st.session_state.comparison_mode,
        help=f"Each successful optimization is saved for side-by-side comparison (up to {MAX_COMPARISON_RUNS} runs).",
    )
    if st.session_state.comparison_mode:
        n_saved = len(st.session_state.comparison_runs)
        if n_saved:
            st.caption(f"{n_saved} run(s) saved — open **Compare** in the nav to view results.")
        else:
            st.caption("Run optimizations with different settings; each run will be added to Compare.")

    # ── Ticker picker CSS ─────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* multiselect tags */
    span[data-baseweb="tag"] {
        background: rgba(99,102,241,0.15) !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        border-radius: 6px !important;
        color: #818CF8 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
    }
    /* multiselect dropdown */
    [data-baseweb="select"] > div {
        background: var(--input-bg) !important;
        border: 1.5px solid var(--input-border) !important;
        border-radius: 10px !important;
    }
    /* preset strip */
    .preset-strip {
        display: flex; flex-wrap: wrap; gap: 6px;
        margin-bottom: 8px;
    }
    .preset-btn {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px; font-weight: 600;
        color: #818CF8;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .preset-btn:hover { background: rgba(99,102,241,0.2); }
    /* section divider */
    .form-section-label {
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.6px; color: var(--text-secondary);
        margin: 16px 0 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.form("optimizer_form"):
        _hdr_col, _apply_col = st.columns([5, 1])
        with _hdr_col:
            st.markdown("""
            <div style='margin-bottom:12px;'>
                <span style='font-size:16px; font-weight:700; color:var(--text-primary);'>Optimization Configuration</span><br>
                <span style='font-size:13px; color:var(--text-secondary);'>Define your asset universe and fine-tune the quantum solver constraints.</span>
            </div>
            """, unsafe_allow_html=True)
        with _apply_col:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            apply_tickers = st.form_submit_button("✦ Apply", use_container_width=True)

        # Compact side-by-side asset selection and custom entry
        asset_col1, asset_col2 = st.columns([1.8, 1.2])
        with asset_col1:
            st.markdown("<div class='form-section-label' style='margin-top:0px;'>⚡ Quick Pick — Top 20 Stocks</div>", unsafe_allow_html=True)
            top20_labels = {f"{t} — {n}": t for t, n in TOP_20_STOCKS.items()}
            selected_labels = st.multiselect(
                "Select from top performers",
                options=list(top20_labels.keys()),
                default=[],
                placeholder="Choose stocks…",
                label_visibility="collapsed",
            )
            selected_from_top20 = [top20_labels[l] for l in selected_labels]
            
        with asset_col2:
            st.markdown("<div class='form-section-label' style='margin-top:0px;'>✏️ Add Custom Tickers</div>", unsafe_allow_html=True)
            custom_draft = st.text_input(
                "Extra tickers (comma-separated)",
                value="",
                placeholder="e.g. PLTR, SNOW",
                label_visibility="collapsed",
                key="opt_custom_draft",
            )

        _draft_tickers = [t.strip().upper() for t in custom_draft.split(",") if t.strip()]
        _preview = list(dict.fromkeys(
            selected_from_top20 + st.session_state.opt_extra_tickers + _draft_tickers
        ))
        if _preview:
            _badges = " ".join(
                f"<span style='background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);"
                f"border-radius:6px;padding:2px 9px;font-size:11px;font-weight:600;color:#818CF8;'>{t}</span>"
                for t in _preview
            )
            st.markdown(
                f"<div style='margin:4px 0 4px;font-size:11px;color:var(--text-secondary);font-weight:600;"
                f"text-transform:uppercase;letter-spacing:.5px;'>Selected ({len(_preview)})</div>"
                f"<div style='display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;'>{_badges}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Optimization Parameters", expanded=True):
            p_col1, p_col2, p_col3 = st.columns(3)
            
            with p_col1:
                st.markdown("<span style='font-size:11px; font-weight:700; color:#818CF8; text-transform:uppercase; letter-spacing:0.5px;'>📈 Position & Risk Control</span>", unsafe_allow_html=True)
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                risk = st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.1, help="Defines the trade-off between maximizing returns and minimizing covariance risk.")
                weight = st.slider("Max Weight per Asset", 1, 3, 1, help="Maximum allocation multiplier per stock in the QUBO solver.")
                
            with p_col2:
                st.markdown("<span style='font-size:11px; font-weight:700; color:#818CF8; text-transform:uppercase; letter-spacing:0.5px;'>⚛ Model & Solver Settings</span>", unsafe_allow_html=True)
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                algo = st.selectbox("Quantum Algorithm", ["QAOA", "VQE"], index=0, help="Quantum optimization algorithm used to minimize the portfolio QUBO.")
                return_est = st.selectbox(
                    "Expected Returns Method",
                    ["Historical Annualized", "AI Forecast (Prophet)", "AI Forecast (XGBoost)"],
                    index=0,
                    help="Defines the returns vector fed to the optimizer: historical averages or predictive AI models."
                )
                ibm = st.checkbox("Use IBM Quantum Hardware", value=False, help="Connects to physical IBM Quantum computing systems instead of local simulators.")
                
            with p_col3:
                st.markdown("<span style='font-size:11px; font-weight:700; color:#818CF8; text-transform:uppercase; letter-spacing:0.5px;'>💼 Friction & Exposure Penalties</span>", unsafe_allow_html=True)
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                tc_penalty = st.slider(
                    "Transaction Cost Penalty",
                    0.0, 1.0, 0.0, 0.05,
                    help="Penalizes portfolio rebalancing relative to your current weights to reduce trading friction/slippage."
                )
                unique_sectors = sorted(list(set(STATIC_SECTOR_MAP.values())))
                limit_sectors = st.multiselect(
                    "Sectors to De-concentrate",
                    options=unique_sectors,
                    default=[],
                    help="Select sectors where you want to actively penalize and reduce concentration."
                )
                sec_penalty = st.slider(
                    "Sector Penalty Strength",
                    0.0, 1.0, 0.0, 0.05,
                    help="Soft quadratic penalty applied to discourage high concentration in the selected sectors."
                )

        # ── Live qubit budget gauge ──────────────────────────────────────────
        _MAX_QUBITS = 18
        _n_tickers  = len(_preview)
        _qubits_used = _n_tickers * weight
        _qubit_pct   = min(_qubits_used / _MAX_QUBITS * 100, 100)
        if _qubits_used <= _MAX_QUBITS * 0.66:
            _q_color, _q_label = "#10B981", "OK"
        elif _qubits_used <= _MAX_QUBITS:
            _q_color, _q_label = "#F59E0B", "Near Limit"
        else:
            _q_color, _q_label = "#EF4444", "Over Limit — reduce tickers or weight"

        st.markdown(f"""
        <div style="margin: 12px 0 4px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:11px; font-weight:600; text-transform:uppercase;
                         letter-spacing:.6px; color:var(--text-secondary);">&#9889; Qubit Budget</span>
            <span style="font-size:12px; font-weight:700; color:{_q_color};">
              {_qubits_used} / {_MAX_QUBITS} qubits &nbsp;&middot;&nbsp; {_n_tickers} assets &times; {weight} weight &nbsp;&mdash;&nbsp; {_q_label}
            </span>
          </div>
          <div style="background:var(--card-border); border-radius:6px; height:8px; overflow:hidden;">
            <div style="width:{_qubit_pct:.1f}%; height:100%; background:{_q_color};
                        border-radius:6px; transition:width .3s ease;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        optimize_btn = st.form_submit_button("Run Optimization", use_container_width=True)

    if apply_tickers:
        _new = [t.strip().upper() for t in custom_draft.split(",") if t.strip()]
        if _new:
            st.session_state.opt_extra_tickers = list(dict.fromkeys(
                st.session_state.opt_extra_tickers + _new
            ))
            st.session_state.opt_custom_draft = ""
            st.rerun()
        else:
            st.warning("Enter at least one ticker (comma-separated).")

    if optimize_btn:
        t_list = list(dict.fromkeys(
            selected_from_top20 + st.session_state.opt_extra_tickers + _draft_tickers
        ))
        _MAX_QUBITS = 18
        _qubits_needed = len(t_list) * weight
        if not t_list:
            st.warning("Please select or enter at least one ticker.")
        elif _qubits_needed > _MAX_QUBITS:
            st.error(
                f"⚠️ **Qubit budget exceeded** — {len(t_list)} tickers × {weight} max weight "
                f"= **{_qubits_needed} qubits** (limit is {_MAX_QUBITS}).\n\n"
                f"**Fix:** remove {_qubits_needed - _MAX_QUBITS} ticker(s), "
                f"or lower *Max Weight per Asset* to {_MAX_QUBITS // len(t_list)}."
            )
        else:
            with st.spinner(f"Executing {algo} Circuit on {len(t_list)} assets ({_qubits_needed} qubits)..."):
                ret_df, err = fetch_data(t_list)
                bench_df = fetch_benchmark()
                if not err:
                    # 1. AI Forecasting Layer
                    mean_returns_input = ret_df.mean()
                    forecast_diagnostics = None
                    if "AI Forecast" in return_est:
                        model_name = "Prophet" if "Prophet" in return_est else "XGBoost"
                        with st.spinner(f"Running machine learning forecasts using {model_name} models..."):
                            from ai_forecaster import get_ai_forecasted_returns
                            forecast_series, forecast_diagnostics = get_ai_forecasted_returns(ret_df, method=model_name)
                            # Convert annualized predictions back to daily returns for the optimization scaling
                            mean_returns_input = forecast_series / 252.0
                    
                    # 2. Rebalancing initial weights
                    initial_allocation = st.session_state.get("weights", None)
                    
                    # 3. Trigger optimization
                    weights, circ = get_optimal_weights(
                        mean_returns_input,
                        ret_df.cov(),
                        risk,
                        weight,
                        ibm,
                        tuple(ret_df.columns),
                        algo=algo,
                        transaction_cost_penalty=tc_penalty,
                        initial_allocation=initial_allocation,
                        sector_weights_penalty=sec_penalty,
                        sector_map=STATIC_SECTOR_MAP,
                        target_sectors=limit_sectors
                    )
                    
                    # Save diagnostic reports in session state
                    st.session_state.forecast_diagnostics = forecast_diagnostics
                    st.session_state.forecast_method = return_est
                    
                    st.session_state.update({
                        "weights": weights,
                        "returns_df": ret_df,
                        "cov_matrix": ret_df.cov(),
                        "benchmark_df": bench_df,
                        "opt_circ": circ,
                        "last_run_ts": datetime.now().strftime("%H:%M")
                    })
                    
                    if st.session_state.comparison_mode:
                        append_comparison_run(t_list, risk, weight, algo, ibm, weights, ret_df, bench_df)
                        n_cmp = len(st.session_state.comparison_runs)
                        st.success(
                            f"Optimization completed — saved as **Run {st.session_state.comparison_run_counter}** "
                            f"({n_cmp}/{MAX_COMPARISON_RUNS} in comparison)."
                        )
                    else:
                        st.success(f"Optimization Completed — {len(t_list)} assets · {_qubits_needed} qubits used.")

elif nav == "Compare":
    st.markdown('<div class="page-title">Run Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Compare optimizer settings and portfolio outcomes side-by-side.</div>',
        unsafe_allow_html=True,
    )
    runs = st.session_state.get("comparison_runs", [])

    if not runs:
        st.info("Enable **Multi-run comparison mode** on the Optimizer, run at least one optimization, then return here.")
    else:
        hdr_l, hdr_r = st.columns([3, 1])
        with hdr_r:
            if st.button("Clear all runs", use_container_width=True):
                st.session_state.comparison_runs = []
                st.session_state.comparison_run_counter = 0
                st.rerun()

        summary_rows = []
        for r in runs:
            m = r["metrics"]
            summary_rows.append({
                "Run": r["label"],
                "Time": r["timestamp"],
                "Algorithm": r["algo"],
                "Risk": r["risk"],
                "Max Weight": r["max_weight"],
                "IBM HW": "Yes" if r["ibm"] else "No",
                "Assets": len(r["tickers"]),
                "Return": m["return"],
                "Volatility": m["vol"],
                "Sharpe": m["sharpe"],
                "Sortino": m["sortino"],
                "Max DD": m["max_dd"],
            })
        st.markdown("<div class='panel-title'>Summary</div>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Return": st.column_config.NumberColumn(format="%.2%"),
                "Volatility": st.column_config.NumberColumn(format="%.2%"),
                "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                "Sortino": st.column_config.NumberColumn(format="%.2f"),
                "Max DD": st.column_config.NumberColumn(format="%.2%"),
                "Risk": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        if len(runs) >= 2:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='panel-title'>Cumulative Performance Overlay</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_comparison_overlay(runs), use_container_width=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='panel-title'>Side-by-Side Runs</div>", unsafe_allow_html=True)

        cmp_cols = st.columns(len(runs))
        _run_colors = ["#6366F1", "#10B981", "#F59E0B", "#EC4899"]
        for col, run in zip(cmp_cols, runs):
            m = run["metrics"]
            accent = _run_colors[(run["id"] - 1) % len(_run_colors)]
            top_holdings = sorted(
                [(t, w) for t, w in run["weights"].items() if w > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            holdings_html = "".join(
                f"<div style='font-size:12px;color:var(--text-secondary);'>"
                f"{t} <span style='color:var(--text-primary);font-weight:600;'>{w*100:.1f}%</span></div>"
                for t, w in top_holdings
            ) or "<div style='font-size:12px;color:var(--text-secondary);'>No allocations</div>"

            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid var(--card-border); border-top:3px solid {accent};
                                border-radius:12px; padding:16px; background:var(--card-bg);
                                box-shadow:0 1px 3px rgba(0,0,0,0.06); margin-bottom:12px;">
                      <div style="font-size:15px;font-weight:700;color:var(--text-primary);">{run['label']}</div>
                      <div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">{run['timestamp']}</div>
                      <div style="margin-top:12px;font-size:11px;font-weight:600;text-transform:uppercase;
                                  letter-spacing:.5px;color:var(--text-secondary);">Settings</div>
                      <div style="font-size:12px;color:var(--text-primary);margin-top:4px;">
                        {run['algo']} · risk {run['risk']:.1f} · max weight {run['max_weight']}
                        {' · IBM Q' if run['ibm'] else ''}<br>
                        {len(run['tickers'])} assets
                      </div>
                      <div style="margin-top:12px;font-size:11px;font-weight:600;text-transform:uppercase;
                                  letter-spacing:.5px;color:var(--text-secondary);">Performance</div>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;">
                        <div><div class="kpi-label">Return</div><div class="kpi-value" style="font-size:18px;">{m['return']:.1%}</div></div>
                        <div><div class="kpi-label">Vol</div><div class="kpi-value" style="font-size:18px;">{m['vol']:.1%}</div></div>
                        <div><div class="kpi-label">Sharpe</div><div class="kpi-value" style="font-size:18px;">{m['sharpe']:.2f}</div></div>
                        <div><div class="kpi-label">Sortino</div><div class="kpi-value" style="font-size:18px;">{m['sortino']:.2f}</div></div>
                      </div>
                      <div style="margin-top:12px;font-size:11px;font-weight:600;text-transform:uppercase;
                                  letter-spacing:.5px;color:var(--text-secondary);">Top Holdings</div>
                      <div style="margin-top:6px;">{holdings_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(plot_allocation_donut(run["weights"]), use_container_width=True)
                if st.button("Remove", key=f"rm_cmp_{run['id']}", use_container_width=True):
                    remove_comparison_run(run["id"])
                    st.rerun()

        if len(runs) == 1:
            st.caption("Add another run in comparison mode to enable the performance overlay and full side-by-side view.")

elif nav == "Portfolio":
    st.markdown('<div class="page-title">Portfolio Dashboard</div>', unsafe_allow_html=True)
    if not st.session_state.weights:
        st.info("Run Optimizer first.")
    else:
        w, r = st.session_state.weights, st.session_state.returns_df
        b = st.session_state.get("benchmark_df", None)
        
        # Enable Live Bloomberg-style Autorefresh Tracking
        col_title, col_toggle = st.columns([2, 1])
        with col_toggle:
            live_mode = st.toggle(
                "Live Bloomberg Tracking", 
                value=st.session_state.get("portfolio_live_mode", False), 
                help="Automatically polls real-time asset prices and calculates intraday portfolio P&L."
            )
            st.session_state.portfolio_live_mode = live_mode
        
        live_prices = {}
        day_changes = {}
        live_port_return = 0.0
        
        if live_mode:
            from streamlit_autorefresh import st_autorefresh
            # Autorefresh every 10 seconds. Limit to 1000 refreshes to protect resource limits.
            refresh_count = st_autorefresh(interval=10000, limit=1000, key="portfolio_autorefresh")
            
            # Fetch current prices & intraday changes for active assets
            active_tickers = [t for t in w.keys() if w[t] > 0]
            with st.spinner("Streaming live market feeds..."):
                for t in active_tickers:
                    try:
                        ticker_obj = yf.Ticker(t)
                        # We use fast_info for extremely low overhead
                        f_info = ticker_obj.fast_info
                        live_prices[t] = f_info.last_price
                        day_changes[t] = ((f_info.last_price - f_info.previous_close) / f_info.previous_close) * 100.0
                    except Exception:
                        # Graceful fallback to historical data
                        live_prices[t] = r[t].iloc[-1]
                        day_changes[t] = r[t].iloc[-1] * 100.0
            
            # Compute live portfolio return
            sum_w = sum(w[t] for t in active_tickers)
            if sum_w > 0:
                live_port_return = sum((w[t] / sum_w) * day_changes[t] for t in active_tickers)
        
        # KPI ROW
        ret, vol, sharpe = compute_metrics(w, r)
        risk_metrics = compute_risk_metrics(w, r, benchmark_returns=b)
        
        var_95, cvar_95 = risk_metrics["VaR"], risk_metrics["CVaR"]
        sortino = risk_metrics["Sortino"]
        max_dd = risk_metrics["Max_DD"]
        alpha = risk_metrics["Alpha"]
        beta = risk_metrics["Beta"]
        
        if live_mode:
            p_val = 100000.0 * (1 + live_port_return / 100.0)
            p_pnl = 100000.0 * (live_port_return / 100.0)
            pnl_sign = "+" if p_pnl >= 0 else ""
            pnl_color = "#10B981" if p_pnl >= 0 else "#EF4444"
            
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="kpi-card" style="padding:16px; border-left: 3px solid {pnl_color};"><div>{kpi_label_html("Live Value ($100K Principal)", "live_value")}<div class="kpi-value">${p_val:,.2f}</div></div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Daily P&L", "live_pnl")}<div class="kpi-value" style="color:{pnl_color}">{pnl_sign}${p_pnl:,.2f} ({live_port_return:+.2f}%)</div></div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Volatility (Ann.)", "volatility")}<div class="kpi-value">{vol:.2%}</div></div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Sharpe Ratio", "sharpe")}<div class="kpi-value">{sharpe:.2f}</div></div></div>', unsafe_allow_html=True)
            
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            # Streaming feed banner
            feed_items = []
            for t in active_tickers:
                price = live_prices.get(t, 0.0)
                change = day_changes.get(t, 0.0)
                color = "#10B981" if change >= 0 else "#EF4444"
                sign = "+" if change >= 0 else ""
                feed_items.append(f"<span style='font-weight:700; color:var(--text-primary);'>{t}:</span> <span style='color:var(--text-secondary);'>${price:.2f}</span> <span style='color:{color}; font-weight:600;'>({sign}{change:.2f}%)</span>")
            
            feed_html = " &nbsp;&bull;&nbsp; ".join(feed_items)
            st.markdown(f"""
            <div style='background:var(--card-bg); border:1px solid var(--card-border); border-radius:8px; padding:10px 16px; font-size:12px; display:flex; align-items:center; gap:12px; overflow-x:auto; white-space:nowrap;'>
                <span style='background:#E0E7FF; color:#4F46E5; font-weight:700; font-size:10px; padding:2px 6px; border-radius:4px; text-transform:uppercase;'>Live Feed</span>
                <div style='overflow:hidden;'>{feed_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Expected Return", "return")}<div class="kpi-value">{ret:.2%}</div></div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Volatility", "volatility")}<div class="kpi-value">{vol:.2%}</div></div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Sharpe Ratio", "sharpe")}<div class="kpi-value">{sharpe:.2f}</div></div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Sortino Ratio", "sortino")}<div class="kpi-value">{sortino:.2f}</div></div></div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("VaR (95%)", "var_95")}<div class="kpi-value" style="color:#EF4444">{var_95:.2%}</div></div></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Max Drawdown", "max_dd")}<div class="kpi-value" style="color:#EF4444">{max_dd:.2%}</div></div></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Alpha", "alpha")}<div class="kpi-value" style="color:#10B981">{"-" if alpha==0 else f"{alpha:.2%}"}</div></div></div>', unsafe_allow_html=True)
        r4.markdown(f'<div class="kpi-card" style="padding:16px;"><div>{kpi_label_html("Beta", "beta")}<div class="kpi-value">{"-" if beta==1.0 else f"{beta:.2f}"}</div></div></div>', unsafe_allow_html=True)
        
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        
        c0, c1 = st.columns([1, 1.5])
        with c0:
            st.markdown(panel_title_html("Allocation", "allocation"), unsafe_allow_html=True)
            st.plotly_chart(plot_allocation_donut(w), use_container_width=True)
        with c1:
            st.markdown(panel_title_html("Historical Performance", "hist_perf"), unsafe_allow_html=True)
            st.plotly_chart(plot_cumulative_performance(r, w), use_container_width=True)
        
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        
        st.markdown(panel_title_html("Efficient Frontier", "efficient_frontier"), unsafe_allow_html=True)
        st.plotly_chart(plot_efficient_frontier(r, w), use_container_width=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        
        st.markdown(panel_title_html("Advanced Insights", "advanced_insights"), unsafe_allow_html=True)
        st.markdown(tab_heading_html("Asset Correlation Network", "correlation_network"), unsafe_allow_html=True)
        st.plotly_chart(plot_correlation_network(st.session_state.returns_df.corr(), threshold=0.3), use_container_width=True)
        
        # AI Forecast Diagnostics Card
        if st.session_state.get("forecast_diagnostics") is not None:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            st.markdown(panel_title_html("AI Forecast Diagnostics", "ai_forecast"), unsafe_allow_html=True)
            
            diag = st.session_state.forecast_diagnostics
            method = st.session_state.get("forecast_method", "AI Forecast")
            
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.2); border-radius:10px; padding:16px; margin-bottom:16px;'>
                <div style='font-weight:700; color:#818CF8; font-size:14px; margin-bottom:4px;'>🧠 {method} Model Insights</div>
                <div style='font-size:13px; color:var(--text-secondary);'>The quantum solver incorporated the following 1-month forward annualized expected returns predictions (rather than standard historical return averages):</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Render in rows of 4 columns for clean scaling
            active_tickers = [t for t in w.keys() if w[t] > 0]
            num_cols = 4
            for i in range(0, len(active_tickers), num_cols):
                chunk = active_tickers[i:i+num_cols]
                cols = st.columns(len(chunk))
                for idx, t in enumerate(chunk):
                    t_diag = diag.get(t, {})
                    status = t_diag.get("status", "Success")
                    predicted_val = t_diag.get("annualized_pct", 0.0)
                    with cols[idx]:
                        st.metric(
                            label=f"{t} Forecasted Return",
                            value=f"{predicted_val:.2f}%",
                            delta=f"Status: {status}",
                            delta_color="normal" if "Success" in status else "off"
                        )
                        
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown(panel_title_html("Detailed Holdings Breakdown", "holdings"), unsafe_allow_html=True)
        
        import numpy as np
        tickers = list(w.keys())
        allocations = [w[t] * 100 for t in tickers if w[t] > 0]
        active_tickers = [t for t in tickers if w[t] > 0]
        
        if len(active_tickers) > 0:
            ann_ret = r[active_tickers].mean() * 252 * 100
            ann_vol = r[active_tickers].std() * np.sqrt(252) * 100
            
            # Reconstruct AI forecast returns side-by-side if available
            has_forecast = (st.session_state.get("forecast_diagnostics") is not None)
            
            df_data = {
                "Ticker": active_tickers,
                "Allocation (%)": allocations,
            }
            
            if has_forecast:
                df_data["Hist Return (%)"] = ann_ret.values
                df_data["AI Forecast Return (%)"] = [
                    st.session_state.forecast_diagnostics.get(t, {}).get("annualized_pct", 0.0)
                    for t in active_tickers
                ]
            else:
                df_data["Expected Return (%)"] = ann_ret.values
                
            df_data["Volatility (%)"] = ann_vol.values
            
            holdings_df = pd.DataFrame(df_data)
            holdings_df = holdings_df.sort_values(by="Allocation (%)", ascending=False).reset_index(drop=True)
            
            col_config = {
                "Allocation (%)": st.column_config.ProgressColumn(
                    "Allocation (%)",
                    help=STAT_TIPS["allocation"],
                    format="%.2f",
                    min_value=0,
                    max_value=100,
                ),
                "Volatility (%)": st.column_config.NumberColumn(
                    format="%.2f",
                    help="Annualized standard deviation of this asset's daily returns.",
                ),
            }
            if has_forecast:
                col_config["Hist Return (%)"] = st.column_config.NumberColumn(
                    format="%.2f",
                    help="Annualized average return for this asset based on historical daily prices.",
                )
                col_config["AI Forecast Return (%)"] = st.column_config.NumberColumn(
                    format="%.2f",
                    help="1-month forward annualized return forecasted by the AI model.",
                )
            else:
                col_config["Expected Return (%)"] = st.column_config.NumberColumn(
                    format="%.2f",
                    help="Annualized average return for this asset based on historical daily prices.",
                )
                
            st.dataframe(
                holdings_df,
                use_container_width=True, 
                hide_index=True,
                column_config=col_config
            )
        else:
            st.warning("No assets allocated.")

elif nav == "Analytics":
    st.markdown('<div class="page-title">Advanced Analytics</div>', unsafe_allow_html=True)
    if not st.session_state.weights:
        st.info("Run the Optimizer first to see analytics.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(panel_title_html("Asset Correlation Heatmap", "correlation_heatmap"), unsafe_allow_html=True)
            st.plotly_chart(plot_correlation_heatmap(st.session_state.cov_matrix), use_container_width=True)
        with col2:
            st.markdown(panel_title_html("Efficient Frontier", "efficient_frontier"), unsafe_allow_html=True)
            st.plotly_chart(plot_efficient_frontier(st.session_state.returns_df, st.session_state.weights), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  WATCHLIST PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif nav == "Watchlist":
    import math

    st.markdown("""
    <style>
    .wl-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 16px 18px 10px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
    }
    .wl-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        border-radius: 14px 0 0 14px;
    }
    .wl-card.up::before  { background: #10B981; }
    .wl-card.down::before { background: #EF4444; }
    .wl-ticker  { font-size:18px; font-weight:800; color:var(--text-primary); }
    .wl-price   { font-size:22px; font-weight:700; color:var(--text-primary); margin-top:4px; }
    .wl-change-up   { font-size:13px; font-weight:600; color:#10B981; }
    .wl-change-down { font-size:13px; font-weight:600; color:#EF4444; }
    .wl-stat-label  { font-size:10px; font-weight:600; text-transform:uppercase;
                      letter-spacing:.5px; color:var(--text-secondary); }
    .wl-stat-value  { font-size:13px; font-weight:600; color:var(--text-primary); margin-top:2px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Page header ──────────────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([3, 2])
    with hdr_l:
        st.markdown('<div class="page-title">⭐ Watchlist</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Track tickers in real-time · 5-min cached · up to 20 symbols</div>', unsafe_allow_html=True)
    with hdr_r:
        with st.form("wl_add_form", clear_on_submit=True):
            add_col, btn_col = st.columns([3, 1])
            with add_col:
                new_tick = st.text_input("Add ticker", placeholder="e.g. TSLA", label_visibility="collapsed")
            with btn_col:
                add_btn = st.form_submit_button("＋ Add", use_container_width=True)
        if add_btn and new_tick:
            ok, msg = add_ticker(new_tick)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    wl = st.session_state.get("watchlist", [])
    if not wl:
        st.info("Your watchlist is empty. Add a ticker above to get started.")
    else:
        # ── Fetch all data ────────────────────────────────────────────────────
        all_data = {}
        with st.spinner("Refreshing watchlist data…"):
            for sym in wl:
                d = fetch_watchlist_ticker(sym)
                if d:
                    all_data[sym] = d

        # ── KPI strip ─────────────────────────────────────────────────────────
        gainers  = [d for d in all_data.values() if d["change_1d"] >= 0]
        losers   = [d for d in all_data.values() if d["change_1d"] < 0]
        avg_chg  = sum(d["change_1d"] for d in all_data.values()) / max(len(all_data), 1)
        best     = max(all_data.values(), key=lambda d: d["change_1d"], default=None)
        worst    = min(all_data.values(), key=lambda d: d["change_1d"], default=None)
        avg_str, avg_color = fmt_change(avg_chg)
        avg_html_color = "#10B981" if avg_chg >= 0 else "#EF4444"

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(f'<div class="kpi-card" style="padding:14px;"><div><div class="kpi-label">Watching</div><div class="kpi-value">{len(wl)}</div></div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card" style="padding:14px;"><div><div class="kpi-label">Gainers</div><div class="kpi-value" style="color:#10B981">{len(gainers)}</div></div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card" style="padding:14px;"><div><div class="kpi-label">Losers</div><div class="kpi-value" style="color:#EF4444">{len(losers)}</div></div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card" style="padding:14px;"><div><div class="kpi-label">Avg Change</div><div class="kpi-value" style="color:{avg_html_color}">{avg_str}</div></div></div>', unsafe_allow_html=True)
        best_txt  = f"{best['ticker']} {fmt_change(best['change_1d'])[0]}"  if best  else "—"
        worst_txt = f"{worst['ticker']} {fmt_change(worst['change_1d'])[0]}" if worst else "—"
        k5.markdown(f'<div class="kpi-card" style="padding:14px;"><div><div class="kpi-label">Best · Worst</div><div class="kpi-value" style="font-size:14px;"><span style="color:#10B981">{best_txt}</span><br><span style="color:#EF4444">{worst_txt}</span></div></div></div>', unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        # ── Card grid (3 columns) ─────────────────────────────────────────────
        cols_per_row = 3
        syms = list(all_data.keys())
        rows = math.ceil(len(syms) / cols_per_row)

        for row_i in range(rows):
            grid = st.columns(cols_per_row)
            for col_i in range(cols_per_row):
                idx = row_i * cols_per_row + col_i
                if idx >= len(syms):
                    break
                sym  = syms[idx]
                d    = all_data[sym]
                is_up = d["change_1d"] >= 0
                chg_str, _ = fmt_change(d["change_1d"])
                period_str, _ = fmt_change(d["change_period"])
                chg_cls  = "wl-change-up" if is_up else "wl-change-down"
                card_cls = "up" if is_up else "down"

                with grid[col_i]:
                    st.markdown(
                        f"""
                        <div class="wl-card {card_cls}">
                          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div>
                              <div class="wl-ticker">{sym}</div>
                              <div class="wl-price">{fmt_price(d['price'])}</div>
                            </div>
                            <div style="text-align:right;">
                              <div class="{chg_cls}" style="font-size:16px;">{chg_str}</div>
                              <div class="wl-stat-label" style="margin-top:2px;">3-mo: <span class="{chg_cls}">{period_str}</span></div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Sparkline
                    st.plotly_chart(
                        plot_sparkline(d["history"], positive=is_up),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                    # Expandable detail panel
                    with st.expander(f"▸ {sym} details"):
                        s1, s2, s3, s4 = st.columns(4)
                        s1.markdown(f'<div class="wl-stat-label">Volume</div><div class="wl-stat-value">{fmt_volume(d["volume"])}</div>', unsafe_allow_html=True)
                        s2.markdown(f'<div class="wl-stat-label">Mkt Cap</div><div class="wl-stat-value">{fmt_mktcap(d["mkt_cap"])}</div>', unsafe_allow_html=True)
                        s3.markdown(f'<div class="wl-stat-label">52W High</div><div class="wl-stat-value">{fmt_price(d["high_52w"])}</div>', unsafe_allow_html=True)
                        s4.markdown(f'<div class="wl-stat-label">52W Low</div><div class="wl-stat-value">{fmt_price(d["low_52w"])}</div>', unsafe_allow_html=True)
                        st.plotly_chart(
                            plot_watchlist_detail(d["history"], sym, positive=is_up),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                        action_l, action_r = st.columns(2)
                        with action_l:
                            if st.button(f"🗑 Remove {sym}", key=f"rm_{sym}", use_container_width=True):
                                remove_ticker(sym)
                                st.rerun()
                        with action_r:
                            if st.button(f"🚀 Optimize with {sym}", key=f"opt_{sym}", use_container_width=True):
                                st.session_state.nav = "Optimizer"
                                st.rerun()
