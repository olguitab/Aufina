"""
Unit tests for OrderEngine — Order Book Trading System (Fase 8).

Tests cover:
  - Price-bound validation (±10%)
  - Commission calculation (0.14%)
  - Offer price suggestion (within bounds)
  - Place + confirm flow
  - Place + cancel flow
  - Stale-offer expiration (timeout)
"""

import os
import sys
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a temporary DB for testing
_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
TEST_DB_PATH = _test_db.name
_test_db.close()

# Patch DB_PATH before importing modules
os.environ.setdefault("GROQ_API_KEY", "test")
import paths
paths.TRADING_DB_FILE = TEST_DB_PATH

import database
database.DB_PATH = TEST_DB_PATH

from database import TradingDB
from order_engine import OrderEngine


class TestPriceBounds(unittest.TestCase):
    """Tests for validate_price_bounds."""

    def test_exact_boundary_lower(self):
        """Price exactly at -10% should be accepted."""
        last_price = 10000
        offer = last_price * 0.90  # exactly -10%
        ok, reason = OrderEngine.validate_price_bounds(offer, last_price)
        self.assertTrue(ok, f"Should accept exact lower bound, got: {reason}")

    def test_exact_boundary_upper(self):
        """Price exactly at +10% should be accepted."""
        last_price = 10000
        offer = last_price * 1.10  # exactly +10%
        ok, reason = OrderEngine.validate_price_bounds(offer, last_price)
        self.assertTrue(ok, f"Should accept exact upper bound, got: {reason}")

    def test_within_bounds(self):
        """Price at +5% should be accepted."""
        ok, reason = OrderEngine.validate_price_bounds(10500, 10000)
        self.assertTrue(ok)

    def test_below_lower_bound(self):
        """Price at -11% must be rejected."""
        last_price = 10000
        offer = last_price * 0.89  # -11%
        ok, reason = OrderEngine.validate_price_bounds(offer, last_price)
        self.assertFalse(ok)
        self.assertIn("debajo", reason)

    def test_above_upper_bound(self):
        """Price at +11% must be rejected."""
        last_price = 10000
        offer = last_price * 1.11  # +11%
        ok, reason = OrderEngine.validate_price_bounds(offer, last_price)
        self.assertFalse(ok)
        self.assertIn("excede", reason)

    def test_invalid_reference_price(self):
        ok, reason = OrderEngine.validate_price_bounds(100, 0)
        self.assertFalse(ok)

    def test_invalid_offer_price(self):
        ok, reason = OrderEngine.validate_price_bounds(0, 100)
        self.assertFalse(ok)


class TestCommission(unittest.TestCase):
    """Tests for apply_commission."""

    def test_buy_commission(self):
        """BUY: total_cost = price*qty + commission"""
        commission, total = OrderEngine.apply_commission(10000, 10, "BUY")
        gross = 10000 * 10
        expected_commission = gross * 0.0014  # 140
        self.assertAlmostEqual(commission, expected_commission, places=2)
        self.assertAlmostEqual(total, gross + expected_commission, places=2)

    def test_sell_commission(self):
        """SELL: total_cost = price*qty - commission (what you receive)"""
        commission, total = OrderEngine.apply_commission(10000, 10, "SELL")
        gross = 10000 * 10
        expected_commission = gross * 0.0014
        self.assertAlmostEqual(commission, expected_commission, places=2)
        self.assertAlmostEqual(total, gross - expected_commission, places=2)

    def test_custom_commission(self):
        """Custom commission rate should work."""
        commission, total = OrderEngine.apply_commission(1000, 5, "BUY", commission_pct=0.01)
        self.assertAlmostEqual(commission, 50.0, places=2)
        self.assertAlmostEqual(total, 5050.0, places=2)


class TestOfferPrice(unittest.TestCase):
    """Tests for calculate_offer_price."""

    def setUp(self):
        self.engine = OrderEngine()

    def test_buy_price_within_bounds(self):
        """Suggested buy price must be within ±10%."""
        price = self.engine.calculate_offer_price("BUY", 10000, 10000, confidence=0.5)
        self.assertGreaterEqual(price, 10000 * 0.90)
        self.assertLessEqual(price, 10000 * 1.10)

    def test_sell_price_within_bounds(self):
        """Suggested sell price must be within ±10%."""
        price = self.engine.calculate_offer_price("SELL", 10000, 10000, confidence=0.5)
        self.assertGreaterEqual(price, 10000 * 0.90)
        self.assertLessEqual(price, 10000 * 1.10)

    def test_high_confidence_buy_near_current(self):
        """High confidence buy should be close to current price."""
        price = self.engine.calculate_offer_price("BUY", 10000, 10000, confidence=0.95)
        self.assertGreater(price, 9800)  # within ~2% of current

    def test_low_confidence_buy_lower(self):
        """Low confidence buy should be further from current price."""
        high_conf_price = self.engine.calculate_offer_price("BUY", 10000, 10000, confidence=0.95)
        low_conf_price = self.engine.calculate_offer_price("BUY", 10000, 10000, confidence=0.1)
        self.assertLess(low_conf_price, high_conf_price)


class TestOrderLifecycle(unittest.TestCase):
    """Tests for place, confirm, cancel, and expire flows."""

    def setUp(self):
        """Set up a fresh DB for each test."""
        # Reset DB
        conn = sqlite3.connect(TEST_DB_PATH)
        conn.execute("DROP TABLE IF EXISTS orders")
        conn.execute("DROP TABLE IF EXISTS global_state")
        conn.execute("DROP TABLE IF EXISTS positions")
        conn.execute("DROP TABLE IF EXISTS trades")
        conn.commit()
        conn.close()
        TradingDB.init_db()

        # Create a mock portfolio
        self.mock_portfolio = MagicMock()
        self.mock_portfolio.balance = 1_000_000.0
        self.mock_portfolio.positions = {}

        self.engine = OrderEngine(portfolio_manager=self.mock_portfolio)

    def test_place_buy_offer(self):
        """Place a BUY offer and verify it's PENDING."""
        ok, msg, order_id = self.engine.place_offer(
            ticker="BSAC.SN", side="BUY", price=45000, quantity=5,
            reasoning="test buy", confidence=0.7, last_trade_price=45000,
        )
        self.assertTrue(ok, f"Should succeed, got: {msg}")
        self.assertGreater(order_id, 0)

        pending = TradingDB.load_pending_orders()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["ticker"], "BSAC.SN")
        self.assertEqual(pending[0]["side"], "BUY")
        self.assertEqual(pending[0]["status"], "PENDING")

    def test_place_offer_out_of_bounds(self):
        """Offer outside ±10% should be rejected."""
        ok, msg, order_id = self.engine.place_offer(
            ticker="BSAC.SN", side="BUY", price=50000, quantity=5,
            reasoning="too high", confidence=0.7, last_trade_price=40000,
        )
        self.assertFalse(ok)
        self.assertEqual(order_id, 0)

    def test_place_offer_insufficient_balance(self):
        """Should reject if total cost exceeds available cash."""
        self.mock_portfolio.balance = 100.0  # very low balance
        ok, msg, order_id = self.engine.place_offer(
            ticker="BSAC.SN", side="BUY", price=45000, quantity=5,
            reasoning="no cash", confidence=0.7, last_trade_price=45000,
        )
        self.assertFalse(ok)
        self.assertIn("capital insuficiente", msg)

    def test_confirm_offer(self):
        """Confirm an offer → trade finalized and position updated."""
        ok, msg, order_id = self.engine.place_offer(
            ticker="BSAC.SN", side="BUY", price=45000, quantity=5,
            reasoning="test confirm", confidence=0.7, last_trade_price=45000,
        )
        self.assertTrue(ok)

        # Mock save methods
        with patch.object(TradingDB, 'save_state'), \
             patch.object(TradingDB, 'save_position'), \
             patch.object(TradingDB, 'log_trade'):
            ok_confirm, msg_confirm = self.engine.confirm_offer(order_id)
        self.assertTrue(ok_confirm, f"Confirm should succeed, got: {msg_confirm}")

        # Verify order is now CONFIRMED
        pending = TradingDB.load_pending_orders()
        self.assertEqual(len(pending), 0)

        confirmed = TradingDB.load_orders(status_filter="CONFIRMED")
        self.assertEqual(len(confirmed), 1)

    def test_cancel_offer(self):
        """Cancel an offer → CANCELLED status."""
        ok, msg, order_id = self.engine.place_offer(
            ticker="BSAC.SN", side="BUY", price=45000, quantity=5,
            reasoning="test cancel", confidence=0.7, last_trade_price=45000,
        )
        self.assertTrue(ok)

        ok_cancel, msg_cancel = self.engine.cancel_offer(order_id)
        self.assertTrue(ok_cancel)

        pending = TradingDB.load_pending_orders()
        self.assertEqual(len(pending), 0)

        cancelled = TradingDB.load_orders(status_filter="CANCELLED")
        self.assertEqual(len(cancelled), 1)

    def test_expire_stale_offers(self):
        """Orders older than timeout should be expired automatically."""
        # Place an offer with a past created_at timestamp
        past_time = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()
        TradingDB.create_order({
            "created_at": past_time,
            "ticker": "BSAC.SN",
            "side": "BUY",
            "offer_price": 45000,
            "quantity": 5,
            "commission_pct": 0.0014,
            "commission_clp": 31.5,
            "total_cost": 225031.5,
            "reasoning": "test expire",
            "confidence": 0.5,
            "metadata": {},
        })

        # Verify it's pending
        pending = TradingDB.load_pending_orders()
        self.assertEqual(len(pending), 1)

        # Expire with 600s timeout (order is 700s old)
        self.engine.timeout_seconds = 600
        expired = self.engine.expire_stale_offers()
        self.assertEqual(len(expired), 1)

        # Verify pending is now empty
        pending = TradingDB.load_pending_orders()
        self.assertEqual(len(pending), 0)

        # Verify it's marked EXPIRED
        expired_orders = TradingDB.load_orders(status_filter="EXPIRED")
        self.assertEqual(len(expired_orders), 1)

    def test_confirm_nonexistent_order(self):
        """Confirming a non-existent order should fail."""
        ok, msg = self.engine.confirm_offer(99999)
        self.assertFalse(ok)

    def test_cancel_nonexistent_order(self):
        """Cancelling a non-existent order should fail."""
        ok, msg = self.engine.cancel_offer(99999)
        self.assertFalse(ok)


class TestPendingSummary(unittest.TestCase):
    """Tests for get_pending_summary."""

    def setUp(self):
        conn = sqlite3.connect(TEST_DB_PATH)
        conn.execute("DROP TABLE IF EXISTS orders")
        conn.commit()
        conn.close()
        TradingDB.init_db()

        self.mock_portfolio = MagicMock()
        self.mock_portfolio.balance = 1_000_000.0
        self.mock_portfolio.positions = {}
        self.engine = OrderEngine(portfolio_manager=self.mock_portfolio)

    def test_empty_summary(self):
        summary = self.engine.get_pending_summary()
        self.assertEqual(summary["pending_count"], 0)
        self.assertEqual(summary["reserved_cash_clp"], 0)

    def test_summary_with_orders(self):
        self.engine.place_offer(
            ticker="BSAC.SN", side="BUY", price=45000, quantity=5,
            reasoning="test", confidence=0.7, last_trade_price=45000,
        )
        summary = self.engine.get_pending_summary()
        self.assertEqual(summary["pending_count"], 1)
        self.assertGreater(summary["reserved_cash_clp"], 0)


def tearDownModule():
    """Clean up temp DB."""
    try:
        os.unlink(TEST_DB_PATH)
    except Exception:
        pass


if __name__ == "__main__":
    unittest.main()
