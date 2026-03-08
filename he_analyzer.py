import re
from typing import Dict


MATERIAL_PATTERNS = {
    "dividendo": 0.35,
    "fusión": 0.50,
    "adquisición": 0.45,
    "hecho esencial": 0.50,
    "aumento de capital": 0.40,
    "downgrade": -0.35,
    "rebaja": -0.30,
    "demanda": -0.25,
    "sanción": -0.40,
    "huelga": -0.20,
    "quiebra": -0.70,
    "insolvencia": -0.70,
    "cambio de clasificación": 0.20,
    "colocación": 0.15,
}


class HEAnalyzer:
    def analyze(self, ticker: str, news_text: str) -> Dict:
        text = (news_text or "").lower()
        if not text or "no news found" in text:
            return {
                "ticker": ticker,
                "event_detected": False,
                "event_type": "none",
                "urgency": "low",
                "impact_score": 0.0,
                "summary": "Sin evento material detectado",
            }

        impact_score = 0.0
        hits = []
        for pattern, weight in MATERIAL_PATTERNS.items():
            if re.search(rf"\b{re.escape(pattern)}\b", text):
                impact_score += weight
                hits.append(pattern)

        impact_score = max(-1.0, min(1.0, impact_score))
        abs_impact = abs(impact_score)
        if abs_impact >= 0.55:
            urgency = "high"
        elif abs_impact >= 0.30:
            urgency = "medium"
        else:
            urgency = "low"

        if impact_score > 0.10:
            event_type = "material_positive"
        elif impact_score < -0.10:
            event_type = "material_negative"
        else:
            event_type = "non_material"

        summary_hits = ", ".join(hits[:4]) if hits else "sin keywords"
        return {
            "ticker": ticker,
            "event_detected": abs_impact >= 0.20,
            "event_type": event_type,
            "urgency": urgency,
            "impact_score": float(impact_score),
            "summary": f"HE scan: {summary_hits}",
        }
