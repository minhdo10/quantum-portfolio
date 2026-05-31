import pandas as pd
import numpy as np
import streamlit as st
import warnings

# Suppress warnings from Prophet/Stan
warnings.filterwarnings("ignore")

@st.cache_data(show_spinner=False, ttl=600)
def get_ai_forecasted_returns(returns_df: pd.DataFrame, method: str = "Prophet", forecast_days: int = 30) -> tuple[pd.Series, dict]:
    """
    Computes forecasted 1-month (annualized equivalent) expected returns for all assets.
    
    Parameters:
    - returns_df: DataFrame of daily returns
    - method: "Prophet" or "XGBoost"
    - forecast_days: calendar days (approx 30 for 1-month)
    
    Returns:
    - pd.Series: Forecasted annualized expected returns vector.
    - dict: Diagnostics/metadata about the forecast.
    """
    forecasted_returns = {}
    diagnostics = {}
    tickers = list(returns_df.columns)
    
    # Calculate historical annualized mean return as a default/fallback
    historical_annualized = returns_df.mean() * 252
    
    for ticker in tickers:
        try:
            # We reconstruct the price series (starting at $100)
            daily_series = returns_df[ticker]
            price_series = (1 + daily_series).cumprod() * 100.0
            
            if method.lower() == "prophet":
                # Lazy import to avoid loading unless requested
                from prophet import Prophet
                
                # Format data for Prophet: 'ds' (datetimes) and 'y' (values)
                df_prophet = pd.DataFrame({
                    "ds": price_series.index.tz_localize(None) if hasattr(price_series.index, "tz") else price_series.index,
                    "y": price_series.values
                })
                
                # Limit training data to last 1-2 years for speed and relevance
                if len(df_prophet) > 252:
                    df_prophet = df_prophet.tail(252)
                
                # Instantiate Prophet with quiet logging
                m = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=True,
                    growth="linear"
                )
                m.fit(df_prophet)
                
                # Create future dates and predict
                future = m.make_future_dataframe(periods=forecast_days, freq="D")
                forecast = m.predict(future)
                
                # Retrieve the last historical price and the forecasted price
                last_hist_price = df_prophet["y"].iloc[-1]
                forecast_val = forecast["yhat"].iloc[-1]
                
                # Calculate expected 1-month return and annualize it
                one_month_return = (forecast_val - last_hist_price) / last_hist_price
                # Annualize: (1 + R_1m)^12 - 1
                annualized_return = (1 + one_month_return) ** 12 - 1
                
                # Sanity check bounds to avoid extreme predictions
                annualized_return = max(min(annualized_return, 1.5), -0.80)
                
                forecasted_returns[ticker] = annualized_return
                diagnostics[ticker] = {
                    "model": "Prophet",
                    "status": "Success",
                    "predicted_1m_pct": one_month_return * 100,
                    "annualized_pct": annualized_return * 100
                }
                
            elif method.lower() == "xgboost":
                from xgboost import XGBRegressor
                
                # Setup supervised learning: predict the 20-day forward return using lagged returns
                lag_periods = [1, 2, 3, 5, 10, 20]
                n_forward = 20 # ~1 business month
                
                df_features = pd.DataFrame(index=daily_series.index)
                df_features["target"] = (1 + daily_series).rolling(window=n_forward).multiplicative_integration_helper() if hasattr(daily_series, "rolling") else None
                # Custom rolling compound return for target:
                # Let's compute actual forward return: R_forward_20 = (Price_{t+20} - Price_t) / Price_t
                prices = price_series.values
                forward_returns = []
                for i in range(len(prices)):
                    if i + n_forward < len(prices):
                        forward_returns.append((prices[i + n_forward] - prices[i]) / prices[i])
                    else:
                        forward_returns.append(np.nan)
                
                df_features["target"] = forward_returns
                
                # Add lags
                for lag in lag_periods:
                    df_features[f"lag_{lag}"] = daily_series.shift(lag)
                
                df_features = df_features.dropna()
                
                if len(df_features) < 40:
                    # Fallback if there is not enough historical data for lagging
                    raise ValueError("Not enough data to train XGBoost.")
                    
                X = df_features[[f"lag_{lag}" for lag in lag_periods]].values
                y = df_features["target"].values
                
                model = XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
                model.fit(X, y)
                
                # Predict for the very latest step (using the most recent lags)
                latest_features = np.array([[daily_series.shift(lag).iloc[-1] for lag in lag_periods]])
                predicted_1m = float(model.predict(latest_features)[0])
                
                # Annualize: (1 + R_1m)^12 - 1
                annualized_return = (1 + predicted_1m) ** 12 - 1
                annualized_return = max(min(annualized_return, 1.5), -0.80)
                
                forecasted_returns[ticker] = annualized_return
                diagnostics[ticker] = {
                    "model": "XGBoost",
                    "status": "Success",
                    "predicted_1m_pct": predicted_1m * 100,
                    "annualized_pct": annualized_return * 100
                }
                
            else:
                raise ValueError(f"Unknown forecasting method: {method}")
                
        except Exception as e:
            # Fallback to historical return
            forecasted_returns[ticker] = historical_annualized[ticker]
            diagnostics[ticker] = {
                "model": method,
                "status": f"Fallback (Error: {str(e)})",
                "predicted_1m_pct": (historical_annualized[ticker] / 12) * 100,
                "annualized_pct": historical_annualized[ticker] * 100
            }
            
    return pd.Series(forecasted_returns), diagnostics
