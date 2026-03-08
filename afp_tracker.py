from typing import Dict

from universe import get_sector


AFP_BIASED_SECTORS = {
    "Financials": 0.25,
    "Utilities": 0.20,
    "Consumer": 0.15,
    "Consumer Staples": 0.15,
    "Materials": 0.10,
}


class AFPTracker:
    def estimate_pressure(self, ticker: str, technical_data: Dict) -> Dict:
        sector = get_sector(ticker)
        sector_bias = AFP_BIASED_SECTORS.get(sector, 0.0)

        daily_return = float(technical_data.get("DailyReturn_Pct", 0.0)) / 100.0
        volume_ratio = float(technical_data.get("Volume_Ratio", 1.0))
        ma_bias = 0.10 if float(technical_data.get("Dist_MA50_Pct", 0.0)) > 0 else -0.10

        flow_component = max(-0.6, min(0.6, (volume_ratio - 1.0) * 0.35))
        momentum_component = max(-0.5, min(0.5, daily_return * 4.0))
        pressure_score = max(
            -1.0,
            min(1.0, sector_bias + flow_component + momentum_component + ma_bias),
        )

        if pressure_score >= 0.25:
            pressure_type = "buying"
        elif pressure_score <= -0.25:
            pressure_type = "selling"
        else:
            pressure_type = "neutral"

        return {
            "ticker": ticker,
            "sector": sector,
            "pressure_score": float(pressure_score),
            "pressure_type": pressure_type,
        }
