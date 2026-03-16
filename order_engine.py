"""
Order Engine for Aureus — Order Book Trading System (Fase 8).

Manages the full lifecycle of trading offers:
  offer → PENDING → CONFIRMED / EXPIRED / CANCELLED

Key rules:
  - Offers must be within ±10% of the last known transaction price
  - Commission of 0.14% per transaction
  - Pending offers timeout after a configurable window (default 10 min)
  - Cash is reserved on BUY offers, shares are locked on SELL offers
  - Nothing is finalized until the user confirms the offer was accepted
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

from database import TradingDB

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
COMMISSION_PCT = float(os.environ.get("ORDER_COMMISSION_PCT", 0.0014))
PRICE_BOUND_PCT = float(os.environ.get("ORDER_PRICE_BOUND_PCT", 0.10))
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("ORDER_TIMEOUT_SECONDS", 600))


class OrderEngine:
    """Manages order-book style offer placement, validation, and lifecycle."""

    def __init__(self, portfolio_manager=None):
        self.portfolio = portfolio_manager
        self.commission_pct = COMMISSION_PCT
        self.price_bound_pct = PRICE_BOUND_PCT
        self.timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    # ── Price Bounds ───────────────────────────────────────────────────────

    @staticmethod
    def validate_price_bounds(
        offer_price: float,
        last_trade_price: float,
        bound_pct: float = PRICE_BOUND_PCT,
    ) -> Tuple[bool, str]:
        """Returns (ok, reason). Rejects offers outside ±bound_pct of last_trade_price."""
        if last_trade_price <= 0:
            return False, "precio de referencia inválido"
        if offer_price <= 0:
            return False, "precio de oferta inválido"

        lower = last_trade_price * (1.0 - bound_pct)
        upper = last_trade_price * (1.0 + bound_pct)

        if offer_price < lower:
            return False, (
                f"precio ${offer_price:,.0f} está por debajo del límite inferior "
                f"(${lower:,.0f}, -{bound_pct:.0%} del último precio ${last_trade_price:,.0f})"
            )
        if offer_price > upper:
            return False, (
                f"precio ${offer_price:,.0f} excede el límite superior "
                f"(${upper:,.0f}, +{bound_pct:.0%} del último precio ${last_trade_price:,.0f})"
            )
        return True, "ok"

    # ── Commission ─────────────────────────────────────────────────────────

    @staticmethod
    def apply_commission(
        price: float,
        quantity: float,
        side: str,
        commission_pct: float = COMMISSION_PCT,
    ) -> Tuple[float, float]:
        """
        Returns (commission_clp, total_cost).
        BUY: total_cost = price*qty + commission
        SELL: total_cost = price*qty - commission  (what you receive)
        """
        gross = price * quantity
        commission_clp = gross * commission_pct
        if side.upper() == "BUY":
            total_cost = gross + commission_clp
        else:
            total_cost = gross - commission_clp
        return round(commission_clp, 2), round(total_cost, 2)

    # ── Offer Price Suggestion ─────────────────────────────────────────────

    def calculate_offer_price(
        self,
        side: str,
        current_price: float,
        last_trade_price: float,
        confidence: float = 0.5,
    ) -> float:
        """
        Suggest a competitive offer price within the ±10% band.

        BUY:  Slightly below current price (better chance of acceptance).
              Higher confidence → closer to current price (more aggressive).
        SELL: Slightly above current price.
              Higher confidence → closer to current price (faster fill).
        """
        if last_trade_price <= 0:
            last_trade_price = current_price
        if current_price <= 0:
            return 0.0

        lower_bound = last_trade_price * (1.0 - self.price_bound_pct)
        upper_bound = last_trade_price * (1.0 + self.price_bound_pct)

        # Confidence maps aggressiveness: 0.0 = conservative, 1.0 = aggressive
        aggressiveness = min(max(confidence, 0.0), 1.0)

        if side.upper() == "BUY":
            # BUY: start from current price, allow small discount
            # High confidence → bid close to current price
            # Low confidence → bid lower (more room for negotiation)
            max_discount = 0.03  # up to 3% below current
            discount = max_discount * (1.0 - aggressiveness)
            suggested = current_price * (1.0 - discount)
        else:
            # SELL: start from current price, allow small premium
            max_premium = 0.03
            premium = max_premium * (1.0 - aggressiveness)
            suggested = current_price * (1.0 + premium)

        # Clamp to bounds
        suggested = max(lower_bound, min(upper_bound, suggested))
        return round(suggested, 2)

    # ── Offer Quantity Suggestion ──────────────────────────────────────────

    def calculate_offer_quantity(
        self,
        side: str,
        price: float,
        confidence: float,
        cash_balance: float,
        positions: Dict[str, float],
        ticker: str = "",
        aggressive: bool = False,
    ) -> int:
        """Calculate suggested quantity considering commission and available capital."""
        if price <= 0:
            return 0

        if side.upper() == "BUY":
            # Reserve cash for price + commission
            effective_price_per_share = price * (1.0 + self.commission_pct)

            # Use portfolio sizing logic: allocate based on confidence
            active_positions = sum(1 for v in positions.values() if v > 0)
            if active_positions < 2:
                if confidence > 0.60:
                    risk_pct = 0.10 if aggressive else 0.05
                elif confidence > 0.45:
                    risk_pct = 0.07 if aggressive else 0.035
                else:
                    risk_pct = 0.04 if aggressive else 0.02
            else:
                if confidence > 0.60:
                    risk_pct = 0.07 if aggressive else 0.035
                elif confidence > 0.45:
                    risk_pct = 0.05 if aggressive else 0.025
                else:
                    risk_pct = 0.03 if aggressive else 0.015

            # Must also exclude cash already reserved in pending orders
            reserved_cash = self._get_reserved_cash()
            available_cash = max(cash_balance - reserved_cash, 0.0)

            amount_to_invest = available_cash * risk_pct
            qty = int(amount_to_invest / effective_price_per_share)

            # Minimum 1 share if affordable
            if qty == 0 and available_cash >= effective_price_per_share:
                qty = 1
            return qty

        else:  # SELL
            current_qty = positions.get(ticker, 0)
            # Exclude shares already locked in pending sell offers
            locked_shares = self._get_locked_shares(ticker)
            available_qty = max(int(current_qty - locked_shares), 0)
            return available_qty

    # ── Place Offer ────────────────────────────────────────────────────────

    def place_offer(
        self,
        ticker: str,
        side: str,
        price: float,
        quantity: int,
        reasoning: str = "",
        confidence: float = 0.5,
        last_trade_price: float = 0.0,
    ) -> Tuple[bool, str, int]:
        """
        Place a new offer. Validates price bounds, reserves cash/shares.
        Returns (success, message, order_id).
        """
        if quantity <= 0:
            return False, "cantidad debe ser mayor a 0", 0
        if price <= 0:
            return False, "precio debe ser mayor a 0", 0

        # Validate price bounds
        ref_price = last_trade_price if last_trade_price > 0 else price
        ok, reason = self.validate_price_bounds(price, ref_price, self.price_bound_pct)
        if not ok:
            return False, reason, 0

        commission_clp, total_cost = self.apply_commission(price, quantity, side)

        side_upper = side.upper()

        if side_upper == "BUY":
            # Check available cash (minus already reserved)
            if self.portfolio:
                reserved_cash = self._get_reserved_cash()
                available = max(self.portfolio.balance - reserved_cash, 0.0)
                if total_cost > available:
                    return False, (
                        f"capital insuficiente: disponible ${available:,.0f} "
                        f"(reservado en ofertas: ${reserved_cash:,.0f}), "
                        f"costo total ${total_cost:,.0f}"
                    ), 0
        elif side_upper == "SELL":
            if self.portfolio:
                current_qty = self.portfolio.positions.get(ticker, 0)
                locked = self._get_locked_shares(ticker)
                available_qty = current_qty - locked
                if quantity > available_qty:
                    return False, (
                        f"acciones insuficientes: tienes {current_qty}, "
                        f"bloqueadas en ofertas {locked}, disponibles {available_qty}"
                    ), 0

        now_iso = datetime.now(timezone.utc).isoformat()
        order_id = TradingDB.create_order({
            "created_at": now_iso,
            "ticker": ticker,
            "side": side_upper,
            "offer_price": price,
            "quantity": quantity,
            "commission_pct": self.commission_pct,
            "commission_clp": commission_clp,
            "total_cost": total_cost,
            "reasoning": reasoning,
            "confidence": confidence,
            "metadata": {
                "last_trade_price": last_trade_price,
                "price_bound_pct": self.price_bound_pct,
            },
        })

        logger.info(
            f"[ORDER] {side_upper} offer placed: {ticker} x{quantity} @ ${price:,.0f} "
            f"(commission ${commission_clp:,.0f}, total ${total_cost:,.0f}) → ID #{order_id}"
        )
        return True, f"Oferta #{order_id} creada exitosamente", order_id

    # ── Confirm Offer ──────────────────────────────────────────────────────

    def confirm_offer(self, order_id: int) -> Tuple[bool, str]:
        """
        User confirms the offer was accepted by a counterparty.
        Finalizes the trade: updates portfolio balance and positions.
        """
        pending = TradingDB.load_pending_orders()
        order = next((o for o in pending if o["id"] == order_id), None)
        if not order:
            return False, f"Oferta #{order_id} no encontrada o ya no está pendiente."

        ticker = order["ticker"]
        side = order["side"]
        price = order["offer_price"]
        qty = order["quantity"]
        total_cost = order["total_cost"]
        commission_clp = order["commission_clp"]

        if self.portfolio:
            if side == "BUY":
                cost = total_cost  # price*qty + commission
                if self.portfolio.balance < cost:
                    return False, f"Capital insuficiente para confirmar (${self.portfolio.balance:,.0f} < ${cost:,.0f})."
                self.portfolio.balance -= cost
                prev_qty = self.portfolio.positions.get(ticker, 0)
                self.portfolio.positions[ticker] = prev_qty + qty

                from database import TradingDB as DB
                DB.save_state(self.portfolio.balance)
                DB.save_position(ticker, self.portfolio.positions[ticker], price)
                DB.log_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                    "signal": "BUY",
                    "price": price,
                    "quantity": qty,
                    "reasoning": f"Oferta #{order_id} confirmada. Comisión: ${commission_clp:,.0f}",
                    "confidence": order.get("confidence", 0.5),
                })
            elif side == "SELL":
                revenue = total_cost  # price*qty - commission
                self.portfolio.balance += revenue
                current_qty = self.portfolio.positions.get(ticker, 0)
                remaining = max(current_qty - qty, 0)
                self.portfolio.positions[ticker] = remaining

                from database import TradingDB as DB
                DB.save_state(self.portfolio.balance)
                DB.save_position(ticker, remaining, price)
                DB.log_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ticker": ticker,
                    "signal": "SELL",
                    "price": price,
                    "quantity": qty,
                    "reasoning": f"Oferta #{order_id} confirmada. Comisión: ${commission_clp:,.0f}",
                    "confidence": order.get("confidence", 0.5),
                })

        TradingDB.confirm_order(order_id)
        logger.info(f"[ORDER] Offer #{order_id} CONFIRMED: {side} {ticker} x{qty} @ ${price:,.0f}")
        return True, f"✅ Oferta #{order_id} confirmada. {side} {qty}x {ticker} @ ${price:,.0f}."

    # ── Cancel Offer ───────────────────────────────────────────────────────

    def cancel_offer(self, order_id: int) -> Tuple[bool, str]:
        """User manually cancels a pending offer. Frees reserved cash/shares."""
        pending = TradingDB.load_pending_orders()
        order = next((o for o in pending if o["id"] == order_id), None)
        if not order:
            return False, f"Oferta #{order_id} no encontrada o ya no está pendiente."

        TradingDB.cancel_order(order_id)
        logger.info(
            f"[ORDER] Offer #{order_id} CANCELLED: {order['side']} {order['ticker']} "
            f"x{order['quantity']} @ ${order['offer_price']:,.0f}"
        )
        return True, f"❌ Oferta #{order_id} cancelada. Capital/acciones liberados."

    # ── Expire Stale Offers ────────────────────────────────────────────────

    def expire_stale_offers(self) -> List[Dict[str, Any]]:
        """
        Auto-cancel offers that have been pending too long.
        Returns the list of expired orders so the caller can re-evaluate.
        """
        expired = TradingDB.expire_stale_orders(self.timeout_seconds)
        for order in expired:
            logger.info(
                f"[ORDER] Offer #{order['id']} EXPIRED (timeout {self.timeout_seconds}s): "
                f"{order['side']} {order['ticker']} x{order['quantity']} @ ${order['offer_price']:,.0f}"
            )
        if expired:
            logger.info(f"[ORDER] {len(expired)} offers expired. Capital/acciones liberados.")
        return expired

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _get_reserved_cash(self) -> float:
        """Sum of total_cost for all pending BUY orders."""
        pending = TradingDB.load_pending_orders()
        return sum(
            o.get("total_cost", 0.0) for o in pending if o.get("side") == "BUY"
        )

    def _get_locked_shares(self, ticker: str) -> float:
        """Sum of quantity for all pending SELL orders on this ticker."""
        pending = TradingDB.load_pending_orders(ticker=ticker)
        return sum(
            o.get("quantity", 0.0) for o in pending if o.get("side") == "SELL"
        )

    def get_pending_summary(self) -> Dict[str, Any]:
        """Returns a summary of all pending orders for display."""
        pending = TradingDB.load_pending_orders()
        reserved_cash = sum(o["total_cost"] for o in pending if o["side"] == "BUY")
        locked_tickers = {}
        for o in pending:
            if o["side"] == "SELL":
                locked_tickers[o["ticker"]] = locked_tickers.get(o["ticker"], 0) + o["quantity"]
        return {
            "pending_count": len(pending),
            "reserved_cash_clp": reserved_cash,
            "locked_sell_shares": locked_tickers,
            "orders": pending,
        }
