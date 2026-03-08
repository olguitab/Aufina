from typing import Dict, Any


PRESET_SCENARIOS = {
    "risk_off": {
        "market_shock_pct": -0.07,
        "copper_shock_pct": -0.05,
        "usdclp_shock_pct": 0.03,
    },
    "bull_rebound": {
        "market_shock_pct": 0.05,
        "copper_shock_pct": 0.04,
        "usdclp_shock_pct": -0.02,
    },
    "commodity_crash": {
        "market_shock_pct": -0.04,
        "copper_shock_pct": -0.12,
        "usdclp_shock_pct": 0.05,
    },
}


class ScenarioSimulator:
    def run(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        cash_balance: float,
        scenario_name: str = "risk_off",
        custom_shock_pct: float = None,
    ) -> Dict[str, Any]:
        scenario = PRESET_SCENARIOS.get(scenario_name, PRESET_SCENARIOS["risk_off"])
        market_shock_pct = float(custom_shock_pct) if custom_shock_pct is not None else float(scenario["market_shock_pct"])

        initial_invested = 0.0
        shocked_invested = 0.0
        per_ticker = {}
        for ticker, qty in positions.items():
            if qty <= 0:
                continue
            px = float(prices.get(ticker, 0.0) or 0.0)
            if px <= 0:
                continue

            base_val = qty * px
            shocked_val = base_val * (1.0 + market_shock_pct)
            initial_invested += base_val
            shocked_invested += shocked_val
            per_ticker[ticker] = {
                "base_value": base_val,
                "shocked_value": shocked_val,
                "pnl_clp": shocked_val - base_val,
            }

        initial_equity = float(cash_balance) + initial_invested
        shocked_equity = float(cash_balance) + shocked_invested
        pnl_clp = shocked_equity - initial_equity
        pnl_pct = (pnl_clp / (initial_equity + 1e-9)) if initial_equity > 0 else 0.0

        return {
            "scenario": scenario_name,
            "market_shock_pct": market_shock_pct,
            "inputs": scenario,
            "initial_equity": initial_equity,
            "shocked_equity": shocked_equity,
            "pnl_clp": pnl_clp,
            "pnl_pct": pnl_pct,
            "per_ticker": per_ticker,
        }
