import os
from typing import List


DEFAULT_EXCLUDED_TICKERS = (
    "SECURITY.SN,CENCOSHOPP.SN,ORO_BLANCO.SN,PILMAIQUEN.SN,"
    "NUEVAPOLAR.SN,MULTIFOODS.SN,INVERMAR.SN,BICECORP.SN,AZULAZUL.SN"
)


# Core IPSA names (high liquidity)
IPSA_WATCHLIST: List[str] = [
    'CHILE.SN', 'SQM-B.SN', 'CENCOSUD.SN', 'ENELAM.SN', 'FALABELLA.SN', 'LTM.SN',
    'BCI.SN', 'BSANTANDER.SN', 'COPEC.SN', 'CMPC.SN', 'AGUAS-A.SN', 'PARAUCO.SN',
    'ANDINA-B.SN', 'VAPORES.SN', 'CCU.SN', 'IAM.SN', 'SECURITY.SN', 'MALLPLAZA.SN',
    'SONDA.SN', 'ENTEL.SN', 'SMU.SN', 'RIPLEY.SN', 'CAP.SN', 'ILC.SN',
    'CENCOSHOPP.SN', 'CONCHATORO.SN', 'ENELCHILE.SN', 'COLBUN.SN', 'ORO_BLANCO.SN', 'VSPT.SN'
]

# Additional Chilean names (IGPA / mid-small caps with relevant local participation)
CHILE_EXPANDED_WATCHLIST: List[str] = [
    'SALFACORP.SN', 'HITES.SN', 'BESALCO.SN', 'FORUS.SN', 'PAZ.SN', 'EMBONOR-B.SN',
    'NUEVAPOLAR.SN', 'SK.SN', 'PILMAIQUEN.SN', 'VOLCAN.SN', 'ELECMETAL.SN', 'MOLYMET.SN',
    'MASISA.SN', 'CAMANCHACA.SN', 'WATTS.SN', 'INVERMAR.SN', 'MULTIFOODS.SN',
    'CGE.SN', 'HABITAT.SN', 'BICECORP.SN', 'INVERCAP.SN', 'MELON.SN', 'AZULAZUL.SN',
    'SCHWAGER.SN', 'MINERA.SN'
]

# Optional global references for diversification / calibration
GLOBAL_WATCHLIST: List[str] = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'NFLX', 'AVGO'
]

# Lightweight sector map for concentration controls
SECTOR_MAP = {
    'CHILE.SN': 'Financials',
    'BSANTANDER.SN': 'Financials',
    'BCI.SN': 'Financials',
    'SECURITY.SN': 'Financials',
    'BICECORP.SN': 'Financials',
    'HABITAT.SN': 'Financials',
    'SQM-B.SN': 'Materials',
    'CMPC.SN': 'Materials',
    'CAP.SN': 'Materials',
    'MOLYMET.SN': 'Materials',
    'COPEC.SN': 'Energy',
    'ENELAM.SN': 'Utilities',
    'ENELCHILE.SN': 'Utilities',
    'COLBUN.SN': 'Utilities',
    'AGUAS-A.SN': 'Utilities',
    'CENCOSUD.SN': 'Consumer',
    'CENCOSHOPP.SN': 'Consumer',
    'FALABELLA.SN': 'Consumer',
    'PARAUCO.SN': 'Consumer',
    'MALLPLAZA.SN': 'Consumer',
    'SMU.SN': 'Consumer',
    'RIPLEY.SN': 'Consumer',
    'HITES.SN': 'Consumer',
    'FORUS.SN': 'Consumer',
    'NUEVAPOLAR.SN': 'Consumer',
    'ANDINA-B.SN': 'Consumer Staples',
    'CCU.SN': 'Consumer Staples',
    'CONCHATORO.SN': 'Consumer Staples',
    'VSPT.SN': 'Consumer Staples',
    'WATTS.SN': 'Consumer Staples',
    'MULTIFOODS.SN': 'Consumer Staples',
    'ENTEL.SN': 'Telecom',
    'SONDA.SN': 'Technology',
    'SK.SN': 'Industrial',
    'SALFACORP.SN': 'Industrial',
    'BESALCO.SN': 'Industrial',
    'PAZ.SN': 'Real Estate',
    'IAM.SN': 'Holding',
    'ORO_BLANCO.SN': 'Holding',
    'LTM.SN': 'Transport',
    'VAPORES.SN': 'Transport',
}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _normalize_ticker(ticker: str) -> str:
    cleaned = (ticker or "").strip().upper()
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    return cleaned


def _excluded_tickers() -> set:
    raw = os.environ.get("EXCLUDED_TICKERS", DEFAULT_EXCLUDED_TICKERS)
    if raw is None or not str(raw).strip():
        raw = DEFAULT_EXCLUDED_TICKERS
    return {_normalize_ticker(t) for t in raw.split(",") if t.strip()}


def _clean_watchlist(items: List[str]) -> List[str]:
    excluded = _excluded_tickers()
    normalized = (_normalize_ticker(t) for t in items)
    filtered = [t for t in normalized if t and t not in excluded]
    return _dedupe_keep_order(filtered)


def get_trading_watchlist(include_global: bool = False) -> List[str]:
    """Returns the watchlist for live scanning/execution."""
    base = _clean_watchlist(IPSA_WATCHLIST + CHILE_EXPANDED_WATCHLIST)
    if include_global:
        return _clean_watchlist(base + GLOBAL_WATCHLIST)
    return base


def get_training_watchlist(include_global: bool = True) -> List[str]:
    """Returns the watchlist used for feature generation and model training."""
    base = _clean_watchlist(IPSA_WATCHLIST + CHILE_EXPANDED_WATCHLIST)
    if include_global:
        return _clean_watchlist(base + GLOBAL_WATCHLIST)
    return base


def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(_normalize_ticker(ticker), 'Other')
