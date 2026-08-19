"""
Deterministic market data fetcher and technical indicator calculator.

IMPORTANT: All numerical values passed to the LLM are computed here,
in pure Python with yfinance + pandas/numpy.  The LLM is never asked
to "calculate" RSI or MACD — it receives the already-computed numbers
as part of its context.  This is the single most important hallucination-
prevention measure in the technical analysis pipeline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

from app.config.settings import get_settings
from app.domain.models import FundamentalMetrics, TechnicalIndicators

log = structlog.get_logger(__name__)


def _safe(value: float | None) -> float | None:
    """Return None if value is NaN or infinite."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _compute_rsi(prices: pd.Series, period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().clip(lower=1e-10)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return _safe(val)


def _compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float | None, float | None, float | None]:
    if len(prices) < slow + signal:
        return None, None, None
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return _safe(macd_line.iloc[-1]), _safe(signal_line.iloc[-1]), _safe(histogram.iloc[-1])


async def fetch_technical_indicators(ticker: str) -> TechnicalIndicators:
    """
    Download 1 year of daily OHLCV data and compute indicators.

    Runs in a thread pool so it doesn't block the event loop.
    """
    settings = get_settings()
    errors: list[str] = []

    def _fetch() -> TechnicalIndicators:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1y", timeout=settings.yfinance_timeout_seconds)

            if hist.empty:
                errors.append(f"No price history available for {ticker}")
                return TechnicalIndicators(
                    ticker=ticker,
                    as_of_date=datetime.utcnow(),
                    computation_errors=errors,
                )

            close = hist["Close"]
            volume = hist["Volume"]

            sma_20 = _safe(close.rolling(20).mean().iloc[-1])
            sma_50 = _safe(close.rolling(50).mean().iloc[-1])
            ema_12 = _safe(close.ewm(span=12, adjust=False).mean().iloc[-1])
            ema_26 = _safe(close.ewm(span=26, adjust=False).mean().iloc[-1])
            rsi_14 = _compute_rsi(close)
            macd, macd_sig, macd_hist = _compute_macd(close)

            # Annualised volatility: std of log returns × √252
            log_returns = np.log(close / close.shift(1)).dropna()
            ann_vol = _safe(log_returns.std() * np.sqrt(252))

            # 3-month annualised return
            three_m = close.iloc[-1] / close.iloc[max(0, len(close) - 63)] - 1
            ann_return = _safe(three_m * (252 / 63))

            vol_avg_20d = _safe(volume.rolling(20).mean().iloc[-1])
            high_52w = _safe(close.rolling(252).max().iloc[-1])
            low_52w = _safe(close.rolling(252).min().iloc[-1])
            current_price = _safe(close.iloc[-1])

            return TechnicalIndicators(
                ticker=ticker,
                as_of_date=datetime.utcnow(),
                current_price=current_price,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_12=ema_12,
                ema_26=ema_26,
                rsi_14=rsi_14,
                macd=macd,
                macd_signal=macd_sig,
                macd_histogram=macd_hist,
                annualized_volatility=ann_vol,
                annualized_return_3m=ann_return,
                volume_avg_20d=vol_avg_20d,
                price_52w_high=high_52w,
                price_52w_low=low_52w,
                computation_errors=errors,
            )
        except Exception as exc:
            log.error("technical_indicator_error", ticker=ticker, error=str(exc))
            errors.append(str(exc))
            return TechnicalIndicators(
                ticker=ticker,
                as_of_date=datetime.utcnow(),
                computation_errors=errors,
            )

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)


async def fetch_fundamental_metrics(ticker: str) -> FundamentalMetrics:
    """Fetch key financial metrics from yfinance info dict."""

    def _fetch() -> FundamentalMetrics:
        errors: list[str] = []
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            def _get(key: str) -> float | None:
                return _safe(info.get(key))

            return FundamentalMetrics(
                ticker=ticker,
                as_of_date=datetime.utcnow(),
                market_cap=_get("marketCap"),
                pe_ratio=_get("trailingPE"),
                pb_ratio=_get("priceToBook"),
                revenue_ttm=_get("totalRevenue"),
                revenue_growth_yoy=_get("revenueGrowth"),
                gross_margin=_get("grossMargins"),
                operating_margin=_get("operatingMargins"),
                net_margin=_get("profitMargins"),
                debt_to_equity=_get("debtToEquity"),
                current_ratio=_get("currentRatio"),
                free_cash_flow=_get("freeCashflow"),
                eps_ttm=_get("trailingEps"),
                dividend_yield=_get("dividendYield"),
                fetch_errors=errors,
            )
        except Exception as exc:
            log.error("fundamental_metrics_error", ticker=ticker, error=str(exc))
            return FundamentalMetrics(
                ticker=ticker,
                as_of_date=datetime.utcnow(),
                fetch_errors=[str(exc)],
            )

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)
