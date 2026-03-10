from typing import Dict

import yfinance as yf


class RegimeDetector:
    def detect(self, benchmark: str = "CHILE.SN") -> Dict:
        try:
            hist = yf.Ticker(benchmark).history(period="4mo")
            if hist is None or hist.empty or len(hist) < 25:
                return {
                    "regime": "sideways",
                    "confidence": 0.4,
                    "ret_20d": 0.0,
                    "vol_20d": 0.02,
                }

            close = hist["Close"]
            ret_20d = float((close.iloc[-1] / close.iloc[-21]) - 1.0) if len(close) >= 21 else 0.0
            vol_20d = float(close.pct_change().rolling(20).std().iloc[-1] or 0.02)

            # Forzar a nunca ser 'bear', solo 'bull' o 'sideways'
            if ret_20d >= 0.05:
                regime = "bull"
            else:
                regime = "sideways"

            confidence = min(0.95, max(0.35, abs(ret_20d) * 6.0 + 0.35))
            return {
                "regime": regime,
                "confidence": float(confidence),
                "ret_20d": ret_20d,
                "vol_20d": vol_20d,
            }
        except Exception:
            return {
                "regime": "sideways",
                "confidence": 0.4,
                "ret_20d": 0.0,
                "vol_20d": 0.02,
            }
