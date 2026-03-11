from abc import ABC, abstractmethod
from typing import Dict, Any

from execution import PortfolioManager
from paper_trading import PaperPortfolio


class BrokerInterface(ABC):
    @abstractmethod
    def get_balance(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def place_order(self, ticker: str, signal: str, price: float, reasoning: str, confidence: float = 0.5) -> Dict[str, Any]:
        raise NotImplementedError


class PaperBrokerAdapter(BrokerInterface):
    def __init__(self, portfolio: PaperPortfolio):
        self.portfolio = portfolio

    def get_balance(self) -> float:
        return float(self.portfolio.balance)

    def get_positions(self) -> Dict[str, float]:
        return dict(self.portfolio.positions)

    def place_order(self, ticker: str, signal: str, price: float, reasoning: str, confidence: float = 0.5, aggressive: bool = False) -> Dict[str, Any]:
        before_balance = float(self.portfolio.balance)
        self.portfolio.execute_order(ticker=ticker, signal=signal, price=price, reasoning=reasoning, confidence=confidence, aggressive=aggressive)
        return {
            "status": "ok",
            "mode": "paper",
            "balance_before": before_balance,
            "balance_after": float(self.portfolio.balance),
            "signal": signal,
            "ticker": ticker,
        }


class ManualRealBrokerAdapter(BrokerInterface):
    """Bridge adapter for real portfolio while execution remains manual-confirmed."""

    def __init__(self, portfolio: PortfolioManager):
        self.portfolio = portfolio

    def get_balance(self) -> float:
        return float(self.portfolio.balance)

    def get_positions(self) -> Dict[str, float]:
        return dict(self.portfolio.positions)

    def place_order(self, ticker: str, signal: str, price: float, reasoning: str, confidence: float = 0.5) -> Dict[str, Any]:
        return {
            "status": "manual_required",
            "mode": "real",
            "signal": signal,
            "ticker": ticker,
            "price": float(price),
            "reasoning": reasoning,
            "confidence": float(confidence),
            "note": "Real broker API not configured yet; execute via manual confirmation channel.",
        }
