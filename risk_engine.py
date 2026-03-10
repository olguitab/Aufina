import os
from typing import Dict, Tuple
import math
import numpy as np

from universe import get_sector


class RiskEngine:
    """Portfolio-level risk checks for buy decisions."""

    def __init__(self):
        is_hosted = any(
            os.environ.get(flag)
            for flag in ("RENDER", "RENDER_SERVICE_ID", "RAILWAY_ENVIRONMENT", "K_SERVICE")
        )
        # AGRESIVO: aumentar límites de exposición
        self.max_position_pct = float(os.environ.get("RISK_MAX_POSITION_PCT", 0.60 if is_hosted else 0.50))
        self.max_sector_pct = float(os.environ.get("RISK_MAX_SECTOR_PCT", 0.90 if is_hosted else 0.85))
        self.max_total_invested_pct = float(os.environ.get("RISK_MAX_TOTAL_INVESTED_PCT", 1.0 if is_hosted else 1.0))
        self.max_open_positions = int(os.environ.get("RISK_MAX_OPEN_POSITIONS", 40 if is_hosted else 32))
        self.min_order_clp = float(os.environ.get("RISK_MIN_ORDER_CLP", 10000 if is_hosted else 10000))

    @staticmethod
    def _portfolio_values(
        cash_balance: float,
        positions: Dict[str, float],
        ticker_prices: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        position_values: Dict[str, float] = {}
        for ticker, qty in positions.items():
            if qty <= 0:
                continue
            px = ticker_prices.get(ticker, 0.0)
            if px <= 0:
                continue
            position_values[ticker] = qty * px

        invested_value = sum(position_values.values())
        equity = max(float(cash_balance), 0.0) + invested_value
        return equity, position_values

    def validate_buy(
        self,
        ticker: str,
        proposed_qty: float,
        proposed_price: float,
        cash_balance: float,
        positions: Dict[str, float],
        ticker_prices: Dict[str, float],
    ) -> Tuple[bool, str]:
        if proposed_qty <= 0 or proposed_price <= 0:
            return False, "orden inválida"

        order_value = proposed_qty * proposed_price
        if order_value < self.min_order_clp:
            return False, f"orden menor al mínimo ({self.min_order_clp:,.0f} CLP)"

        if order_value > max(cash_balance, 0.0):
            return False, "capital insuficiente"

        equity, position_values = self._portfolio_values(cash_balance, positions, ticker_prices)
        if equity <= 0:
            return False, "equity inválido"

        open_positions = len([t for t, v in positions.items() if v > 0])
        is_new_position = positions.get(ticker, 0) <= 0
        if is_new_position and open_positions >= self.max_open_positions:
            return False, f"límite de posiciones abiertas ({self.max_open_positions})"

        current_ticker_value = position_values.get(ticker, 0.0)
        projected_ticker_value = current_ticker_value + order_value
        if (projected_ticker_value / equity) > self.max_position_pct:
            return False, (
                f"concentración por activo excedida ({projected_ticker_value / equity:.1%} > "
                f"{self.max_position_pct:.1%})"
            )

        invested_value = sum(position_values.values())
        projected_invested = invested_value + order_value
        if (projected_invested / equity) > self.max_total_invested_pct:
            return False, (
                f"exposición total excedida ({projected_invested / equity:.1%} > "
                f"{self.max_total_invested_pct:.1%})"
            )

        sector = get_sector(ticker)
        current_sector_value = 0.0
        for tk, val in position_values.items():
            if get_sector(tk) == sector:
                current_sector_value += val
        projected_sector = current_sector_value + order_value
        if (projected_sector / equity) > self.max_sector_pct:
            return False, (
                f"concentración sectorial excedida ({sector}: {projected_sector / equity:.1%} > "
                f"{self.max_sector_pct:.1%})"
            )

        return True, "ok"

    def portfolio_risk_snapshot(
        self,
        cash_balance: float,
        positions: Dict[str, float],
        ticker_prices: Dict[str, float],
    ) -> Dict[str, float]:
        equity, position_values = self._portfolio_values(cash_balance, positions, ticker_prices)
        invested_value = sum(position_values.values())
        invested_pct = (invested_value / equity) if equity > 0 else 0.0
        max_single_pct = max(((v / equity) for v in position_values.values()), default=0.0) if equity > 0 else 0.0

        sector_values: Dict[str, float] = {}
        for ticker, value in position_values.items():
            sector = get_sector(ticker)
            sector_values[sector] = sector_values.get(sector, 0.0) + value
        max_sector_pct = max(((v / equity) for v in sector_values.values()), default=0.0) if equity > 0 else 0.0

        return {
            "equity": equity,
            "invested_value": invested_value,
            "invested_pct": invested_pct,
            "open_positions": float(len(position_values)),
            "max_single_position_pct": max_single_pct,
            "max_sector_pct": max_sector_pct,
        }

    def estimate_parametric_var(
        self,
        cash_balance: float,
        positions: Dict[str, float],
        ticker_prices: Dict[str, float],
        returns_vol: Dict[str, float],
        correlation_matrix: Dict[str, Dict[str, float]] = None,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """Simple 1-day parametric VaR using weighted volatility and normal z-score approximation."""
        equity, position_values = self._portfolio_values(cash_balance, positions, ticker_prices)
        if equity <= 0:
            return {"var_1d_clp": 0.0, "var_1d_pct": 0.0, "confidence": confidence, "method": "n/a"}

        invested_value = sum(position_values.values())
        if invested_value <= 0:
            return {"var_1d_clp": 0.0, "var_1d_pct": 0.0, "confidence": confidence, "method": "n/a"}

        tickers = list(position_values.keys())
        weights = np.array([position_values[t] / invested_value for t in tickers], dtype=float)
        vols = np.array([abs(float(returns_vol.get(t, 0.02))) for t in tickers], dtype=float)

        method = "diagonal"
        if correlation_matrix:
            corr = np.eye(len(tickers), dtype=float)
            for i, ti in enumerate(tickers):
                for j, tj in enumerate(tickers):
                    if i == j:
                        corr[i, j] = 1.0
                    else:
                        cij = correlation_matrix.get(ti, {}).get(tj)
                        if cij is None:
                            cij = correlation_matrix.get(tj, {}).get(ti)
                        if cij is None:
                            cij = 0.0
                        corr[i, j] = float(np.clip(cij, -0.99, 0.99))

            cov = np.outer(vols, vols) * corr
            portfolio_var = float(weights.T @ cov @ weights)
            portfolio_vol = math.sqrt(max(portfolio_var, 0.0))
            method = "covariance"
        else:
            weighted_var = float(np.sum((weights ** 2) * (vols ** 2)))
            portfolio_vol = math.sqrt(max(weighted_var, 0.0))

        # z-score approximation for common confidence levels
        z = 1.65 if confidence >= 0.95 else 1.28
        var_clp = invested_value * z * portfolio_vol
        var_pct = (var_clp / equity) if equity > 0 else 0.0

        return {
            "var_1d_clp": float(var_clp),
            "var_1d_pct": float(var_pct),
            "confidence": float(confidence),
            "method": method,
        }

    def build_correlation_matrix(
        self,
        returns_history: Dict[str, list],
        min_periods: int = 20,
    ) -> Dict[str, Dict[str, float]]:
        """Builds a rolling correlation matrix from per-ticker return series."""
        if not returns_history:
            return {}

        filtered = {}
        for ticker, series in returns_history.items():
            arr = np.array(series, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.shape[0] >= min_periods:
                filtered[ticker] = arr

        if len(filtered) < 2:
            return {}

        min_len = min(len(v) for v in filtered.values())
        aligned = {k: v[-min_len:] for k, v in filtered.items()}
        matrix = np.array([aligned[k] for k in aligned.keys()], dtype=float)
        corr = np.corrcoef(matrix)
        tickers = list(aligned.keys())

        output: Dict[str, Dict[str, float]] = {}
        for i, ti in enumerate(tickers):
            output[ti] = {}
            for j, tj in enumerate(tickers):
                value = float(corr[i, j]) if not np.isnan(corr[i, j]) else 0.0
                output[ti][tj] = float(np.clip(value, -0.99, 0.99))
        return output

    def run_stress_tests(
        self,
        cash_balance: float,
        positions: Dict[str, float],
        ticker_prices: Dict[str, float],
    ) -> Dict[str, float]:
        """Deterministic stress scenarios over invested portfolio value."""
        equity, position_values = self._portfolio_values(cash_balance, positions, ticker_prices)
        invested_value = sum(position_values.values())
        if equity <= 0 or invested_value <= 0:
            return {
                "shock_mild_pct": 0.0,
                "shock_moderate_pct": 0.0,
                "shock_severe_pct": 0.0,
                "shock_mild_clp": 0.0,
                "shock_moderate_clp": 0.0,
                "shock_severe_clp": 0.0,
            }

        mild = invested_value * 0.03
        moderate = invested_value * 0.07
        severe = invested_value * 0.12

        return {
            "shock_mild_pct": mild / equity,
            "shock_moderate_pct": moderate / equity,
            "shock_severe_pct": severe / equity,
            "shock_mild_clp": mild,
            "shock_moderate_clp": moderate,
            "shock_severe_clp": severe,
        }
