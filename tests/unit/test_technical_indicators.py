"""Tests for deterministic technical indicator computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.market_data import _compute_macd, _compute_rsi, _safe


class TestSafe:
    def test_returns_none_for_nan(self) -> None:
        assert _safe(float("nan")) is None

    def test_returns_none_for_inf(self) -> None:
        assert _safe(float("inf")) is None

    def test_rounds_value(self) -> None:
        assert _safe(1.23456789) == 1.2346

    def test_returns_none_for_none(self) -> None:
        assert _safe(None) is None


class TestRSI:
    def _make_prices(self, n: int) -> pd.Series:
        """Create a simple ascending price series."""
        return pd.Series(range(1, n + 1), dtype=float)

    def test_returns_none_when_insufficient_data(self) -> None:
        prices = self._make_prices(10)
        result = _compute_rsi(prices, period=14)
        assert result is None

    def test_rsi_in_valid_range(self) -> None:
        prices = self._make_prices(30)
        result = _compute_rsi(prices, period=14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_rising_prices_yield_high_rsi(self) -> None:
        # Continuously rising prices → RSI should be high (>70 overbought)
        prices = pd.Series([100.0 + i * 2.0 for i in range(50)], dtype=float)
        result = _compute_rsi(prices)
        assert result is not None
        assert result > 70

    def test_falling_prices_yield_low_rsi(self) -> None:
        # Continuously falling prices → RSI should be low (<30 oversold)
        prices = pd.Series([100.0 - i * 2.0 for i in range(50)], dtype=float)
        result = _compute_rsi(prices)
        assert result is not None
        assert result < 30


class TestMACD:
    def _make_prices(self, n: int) -> pd.Series:
        return pd.Series(range(1, n + 1), dtype=float)

    def test_returns_none_tuple_when_insufficient_data(self) -> None:
        prices = self._make_prices(10)
        macd, signal, hist = _compute_macd(prices)
        assert macd is None
        assert signal is None
        assert hist is None

    def test_returns_values_with_sufficient_data(self) -> None:
        prices = self._make_prices(50)
        macd, signal, hist = _compute_macd(prices)
        assert macd is not None
        assert signal is not None
        assert hist is not None

    def test_histogram_is_macd_minus_signal(self) -> None:
        prices = self._make_prices(50)
        macd, signal, hist = _compute_macd(prices)
        if macd is not None and signal is not None and hist is not None:
            assert abs(hist - (macd - signal)) < 0.01
