import os
from datetime import datetime, time, timedelta
import time as time_module
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

class MarketData:
    MACRO_CACHE_TTL_SECONDS = int(os.environ.get("MACRO_CACHE_TTL_SECONDS", 90))
    INVALID_TICKER_COOLDOWN_SECONDS = int(os.environ.get("INVALID_TICKER_COOLDOWN_SECONDS", 6 * 3600))
    _macro_cache_data: Dict[str, float] | None = None
    _macro_cache_expires_at: float = 0.0
    _copper_returns_cache: pd.Series | None = None
    _copper_returns_expires_at: float = 0.0
    _invalid_ticker_until: Dict[str, float] = {}

    MARKET_TZ = ZoneInfo("America/Santiago")
    MARKET_OPEN = time(
        hour=int(os.environ.get("SSE_OPEN_HOUR", 9)),
        minute=int(os.environ.get("SSE_OPEN_MINUTE", 30)),
    )
    MARKET_CLOSE = time(
        hour=int(os.environ.get("SSE_CLOSE_HOUR", 16)),
        minute=int(os.environ.get("SSE_CLOSE_MINUTE", 0)),
    )

    @classmethod
    def is_santiago_market_open(cls, now: datetime = None) -> bool:
        """Returns True if Bolsa de Santiago is open (Mon-Fri, configured local hours)."""
        current = now.astimezone(cls.MARKET_TZ) if now else datetime.now(cls.MARKET_TZ)
        if current.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        current_local_time = current.time()
        return cls.MARKET_OPEN <= current_local_time < cls.MARKET_CLOSE

    @classmethod
    def seconds_until_next_santiago_open(cls, now: datetime = None) -> int:
        """Returns seconds remaining until next market open in Santiago timezone."""
        current = now.astimezone(cls.MARKET_TZ) if now else datetime.now(cls.MARKET_TZ)

        if cls.is_santiago_market_open(current):
            return 0

        target_date = current.date()

        if current.weekday() >= 5:
            days_to_monday = 7 - current.weekday()
            target_date = target_date + timedelta(days=days_to_monday)
        else:
            if current.time() >= cls.MARKET_CLOSE:
                target_date = target_date + timedelta(days=1)

            while target_date.weekday() >= 5:
                target_date = target_date + timedelta(days=1)

        next_open = datetime.combine(target_date, cls.MARKET_OPEN, tzinfo=cls.MARKET_TZ)
        return max(int((next_open - current).total_seconds()), 0)

    @staticmethod
    def get_comprehensive_data(ticker: str, date: datetime.date = None) -> Dict[str, Any]:
        """Fetches all necessary data (Technicals, News, Status) in a single workflow.
        Reduces YFinance overhead significantly.
        If date is provided, returns data as of that date (historical close)."""
        ticker = MarketData.normalize_ticker(ticker)
        now_ts = time_module.time()
        blocked_until = MarketData._invalid_ticker_until.get(ticker, 0.0)
        if blocked_until > now_ts:
            return {
                "is_active": False,
                "current_price": 0,
                "error": "Ticker temporarily skipped (no recent data)",
            }

        try:
            stock = yf.Ticker(ticker)
            # 1. Fetch historical data (3 months handles everything)
            hist = MarketData._safe_history(stock, period="3mo")
            if hist.empty or len(hist) < 2:
                # Delisted or invalid symbol: Return a neutral state to avoid hanging
                MarketData._invalid_ticker_until[ticker] = now_ts + max(300, MarketData.INVALID_TICKER_COOLDOWN_SECONDS)
                return {"is_active": False, "current_price": 0, "error": "No data/Delisted"}

            MarketData._invalid_ticker_until.pop(ticker, None)

            close = hist['Close']
            # Si se pasa una fecha, buscar el precio de cierre de ese día
            if date is not None:
                # Buscar el índice más cercano <= date
                close_hist = close.copy()
                close_hist.index = pd.to_datetime(close_hist.index).date
                if date in close_hist.index:
                    current = close_hist.loc[date]
                    idx = list(close_hist.index).index(date)
                    prev_close = close_hist.iloc[idx - 1] if idx > 0 else current
                else:
                    # Buscar la fecha más cercana anterior
                    prev_dates = [d for d in close_hist.index if d < date]
                    if prev_dates:
                        last_date = max(prev_dates)
                        current = close_hist.loc[last_date]
                        idx = list(close_hist.index).index(last_date)
                        prev_close = close_hist.iloc[idx - 1] if idx > 0 else current
                    else:
                        current = close.iloc[-1]
                        prev_close = close.iloc[-2]
            else:
                # During market hours, try to get a more real-time price than the last history row
                current = close.iloc[-1]
                prev_close = close.iloc[-2]
                
                # If market is open, try fast_info/info for a lower-latency quote
                if MarketData.is_santiago_market_open():
                    try:
                        fast_info = getattr(stock, "fast_info", None)
                        if fast_info:
                            ltp = float(fast_info.get("lastPrice") or 0)
                            if ltp > 0:
                                current = ltp
                        else:
                            # info is slower but more thorough
                            info = getattr(stock, "info", None)
                            if info:
                                ltp = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
                                if ltp > 0:
                                    current = ltp
                    except Exception:
                        pass
            
            # --- Technical Indicators ---
            ma20 = close.rolling(window=20).mean().iloc[-1]
            ma50 = close.rolling(window=50).mean().iloc[-1]
            ma200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else close.rolling(window=50).mean().iloc[-1]
            
            daily_return = ((current - prev_close) / prev_close) * 100
            five_day_return = ((current - close.iloc[-6]) / close.iloc[-6]) * 100 if len(close) >= 6 else 0.0
            ten_day_return = ((current - close.iloc[-11]) / close.iloc[-11]) * 100 if len(close) >= 11 else 0.0
            dist_ma20 = ((current - ma20) / ma20) * 100
            dist_ma50 = ((current - ma50) / ma50) * 100 if ma50 else 0.0
            dist_ma200 = ((current - ma200) / ma200) * 100 if ma200 else 0.0
            volatility = close.pct_change().rolling(window=20).std().iloc[-1]
            volatility_5d = close.pct_change().rolling(window=5).std().iloc[-1]
            recent_returns_30d = close.pct_change().dropna().tail(30).tolist()
            stock_returns = close.pct_change().dropna()
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal

            bb_mid = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = bb_mid + (2 * bb_std)
            bb_lower = bb_mid - (2 * bb_std)
            bb_width = ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / (bb_mid.iloc[-1] + 1e-9)) if len(bb_mid) else 0.0
            bb_position = (current - bb_lower.iloc[-1]) / ((bb_upper.iloc[-1] - bb_lower.iloc[-1]) + 1e-9) if len(bb_upper) else 0.5

            obv = (np.sign(close.diff()).fillna(0.0) * hist['Volume'].fillna(0.0)).cumsum()
            obv_ma20 = obv.rolling(window=20).mean().iloc[-1] if len(obv) >= 20 else obv.iloc[-1]
            volume_ratio = hist['Volume'].iloc[-1] / (hist['Volume'].rolling(window=20).mean().iloc[-1] + 1e-9)
            
            month_ago = close.iloc[-21] if len(close) >= 21 else close.iloc[0]
            monthly_return = ((current - month_ago) / month_ago) * 100
            
            # --- Gatekeeper Logic (is_active) ---
            daily_range = (hist['High'].iloc[-1] - hist['Low'].iloc[-1]) / (hist['Low'].iloc[-1] + 1e-9)
            is_active = (abs(daily_return) > 0.5) or (daily_range > 0.01)
            
            # --- News ---
            news_items = getattr(stock, 'news', [])
            if news_items is None:
                news_items = []
            else:
                news_items = news_items[:3]
                
            formatted_news = []
            if news_items:
                for item in news_items:
                    formatted_news.append(f"Headline: {item.get('title', '')} (Source: {item.get('publisher', 'Unknown')})")
            news_text = "\n\n".join(formatted_news) if formatted_news else "No news found."

            # --- ATR (Volatility for Stops) ---
            high_low = hist['High'] - hist['Low']
            high_close = (hist['High'] - hist['Close'].shift()).abs()
            low_close = (hist['Low'] - hist['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            atr_ratio = atr / (current + 1e-9)

            # --- ADV (Average Daily Volume - 20d) ---
            adv_20d = hist['Volume'].rolling(window=20).mean().iloc[-1]

            copper_corr_20d = 0.0
            try:
                copper_returns = MarketData._get_copper_returns_series()
                if copper_returns is not None and not copper_returns.empty:
                    aligned = pd.concat(
                        [stock_returns.tail(25), copper_returns.tail(25)],
                        axis=1,
                        join="inner",
                    ).dropna()
                    if len(aligned) >= 10:
                        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                        if corr == corr:
                            copper_corr_20d = float(corr)
            except Exception:
                copper_corr_20d = 0.0

            volatility_safe = max(float(volatility) if volatility == volatility else 0.02, 0.005)
            atr_ratio_safe = max(float(atr_ratio) if atr_ratio == atr_ratio else 0.01, 0.002)
            adv_safe = max(float(adv_20d) if adv_20d == adv_20d else 0.0, 1.0)
            liquidity_score = float(min(1.0, max(0.0, (adv_safe / (adv_safe + 1_000_000.0)) * (1.0 / (1.0 + 8.0 * atr_ratio_safe)))))
            est_impact_1pct_adv = float((atr_ratio_safe * 100.0) * np.sqrt(max(0.01, 0.01 * adv_safe / adv_safe)))
            
            # --- Commodity Correlation: Copper (HG=F) ---
            macro_returns = MarketData._get_macro_returns()

            return {
                "is_active": is_active,
                "current_price": current,
                "close_price": current, # for consistent access in UI
                "technical_data": {
                    "CurrentPrice": current,
                    "MA20": ma20,
                    "MA50": ma50,
                    "MA200": ma200,
                    "RSI_14": current_rsi,
                    "ATR": atr,
                    "ATR_Ratio": atr_ratio,
                    "ADV_20d": adv_20d,
                    "Realtime_Copper_Corr_20d": copper_corr_20d,
                    "Liquidity_Score": liquidity_score,
                    "Est_Impact_1pct_ADV_Pct": est_impact_1pct_adv,
                    "Copper_Return": macro_returns["Macro_Copper_Ret"],
                    "MonthlyReturn_Pct": monthly_return,
                    "DailyReturn_Pct": daily_return,
                    "FiveDayReturn_Pct": five_day_return,
                    "TenDayReturn_Pct": ten_day_return,
                    "Dist_MA20_Pct": dist_ma20,
                    "Dist_MA50_Pct": dist_ma50,
                    "Dist_MA200_Pct": dist_ma200,
                    "Volatility_20d": volatility,
                    "Volatility_5d": volatility_5d,
                    "Recent_Returns_30d": [float(x) for x in recent_returns_30d],
                    "MACD": float(macd.iloc[-1]) if len(macd) else 0.0,
                    "MACD_Signal": float(macd_signal.iloc[-1]) if len(macd_signal) else 0.0,
                    "MACD_Hist": float(macd_hist.iloc[-1]) if len(macd_hist) else 0.0,
                    "BB_Width": float(bb_width),
                    "BB_Position": float(bb_position),
                    "OBV": float(obv.iloc[-1]) if len(obv) else 0.0,
                    "OBV_MA20": float(obv_ma20),
                    "Volume_Ratio": float(volume_ratio),
                    **macro_returns,
                    "Trend": "Bullish" if ma20 > ma50 else "Bearish"
                },
                "news_text": news_text
            }
        except Exception as e:
            fallback_price = 0.0
            try:
                stock = yf.Ticker(ticker)
                fast_info = getattr(stock, "fast_info", None) or {}
                if hasattr(fast_info, "get"):
                    fallback_price = float(fast_info.get("lastPrice") or fast_info.get("regularMarketPrice") or 0.0)
                if fallback_price <= 0:
                    info = getattr(stock, "info", {}) or {}
                    fallback_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose") or 0.0)
            except Exception:
                fallback_price = 0.0

            if fallback_price > 0:
                macro_returns = MarketData._get_macro_returns()
                return {
                    "is_active": True,
                    "current_price": fallback_price,
                    "close_price": fallback_price,
                    "technical_data": {
                        "CurrentPrice": fallback_price,
                        "MA20": fallback_price,
                        "MA50": fallback_price,
                        "MA200": fallback_price,
                        "RSI_14": 50.0,
                        "ATR": 0.0,
                        "ATR_Ratio": 0.01,
                        "ADV_20d": 0.0,
                        "Realtime_Copper_Corr_20d": 0.0,
                        "Liquidity_Score": 0.5,
                        "Est_Impact_1pct_ADV_Pct": 0.0,
                        "Copper_Return": macro_returns["Macro_Copper_Ret"],
                        "MonthlyReturn_Pct": 0.0,
                        "DailyReturn_Pct": 0.0,
                        "FiveDayReturn_Pct": 0.0,
                        "TenDayReturn_Pct": 0.0,
                        "Dist_MA20_Pct": 0.0,
                        "Dist_MA50_Pct": 0.0,
                        "Dist_MA200_Pct": 0.0,
                        "Volatility_20d": 0.02,
                        "Volatility_5d": 0.02,
                        "Recent_Returns_30d": [],
                        "MACD": 0.0,
                        "MACD_Signal": 0.0,
                        "MACD_Hist": 0.0,
                        "BB_Width": 0.0,
                        "BB_Position": 0.5,
                        "OBV": 0.0,
                        "OBV_MA20": 0.0,
                        "Volume_Ratio": 1.0,
                        **macro_returns,
                        "Trend": "Sideways",
                    },
                    "news_text": "No news found.",
                    "warning": "Using quote-price fallback (history unavailable)",
                }

            MarketData._invalid_ticker_until[ticker] = time_module.time() + max(300, MarketData.INVALID_TICKER_COOLDOWN_SECONDS)
            print(f"Error in comprehensive data for {ticker}: {e}")
            return {"is_active": False, "current_price": 0, "error": str(e)}

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        cleaned = (ticker or "").strip().upper()
        if cleaned.startswith("$"):
            cleaned = cleaned[1:]
        return cleaned

    @staticmethod
    def _safe_history(stock: yf.Ticker, period: str, max_attempts: int = 3) -> pd.DataFrame:
        """Reads historical data with lightweight retries for transient network/provider issues."""
        last_error = None
        for attempt in range(max_attempts):
            try:
                return stock.history(period=period)
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    time_module.sleep(0.7 * (attempt + 1))
        if last_error:
            raise last_error
        return pd.DataFrame()

    @classmethod
    def _get_macro_returns(cls) -> Dict[str, float]:
        now = time_module.time()
        if cls._macro_cache_data is not None and now < cls._macro_cache_expires_at:
            return dict(cls._macro_cache_data)

        macro_returns = {
            "Macro_Copper_Ret": 0.0,
            "Macro_SP500_Ret": 0.0,
            "Macro_USDCLP_Ret": 0.0,
            "Macro_Lithium_Ret": 0.0,
            "Macro_MSCI_EM_Ret": 0.0,
            "Macro_VIX_Ret": 0.0,
        }

        for sym, key in {
            "HG=F": "Macro_Copper_Ret",
            "^GSPC": "Macro_SP500_Ret",
            "CLP=X": "Macro_USDCLP_Ret",
            "ALB": "Macro_Lithium_Ret",
            "EEM": "Macro_MSCI_EM_Ret",
            "^VIX": "Macro_VIX_Ret",
        }.items():
            try:
                m_hist = cls._safe_history(yf.Ticker(sym), period="2d")
                if len(m_hist) >= 2:
                    macro_returns[key] = (m_hist['Close'].iloc[-1] - m_hist['Close'].iloc[-2]) / m_hist['Close'].iloc[-2]
            except Exception:
                continue

        cls._macro_cache_data = dict(macro_returns)
        cls._macro_cache_expires_at = now + max(10, cls.MACRO_CACHE_TTL_SECONDS)
        return macro_returns

    @classmethod
    def _get_copper_returns_series(cls) -> pd.Series:
        now = time_module.time()
        if cls._copper_returns_cache is not None and now < cls._copper_returns_expires_at:
            return cls._copper_returns_cache

        try:
            c_hist = cls._safe_history(yf.Ticker("HG=F"), period="3mo")
            if c_hist is None or c_hist.empty:
                cls._copper_returns_cache = pd.Series(dtype=float)
            else:
                sr = c_hist["Close"].pct_change().dropna()
                sr.index = pd.to_datetime(sr.index).tz_localize(None)
                cls._copper_returns_cache = sr
        except Exception:
            cls._copper_returns_cache = pd.Series(dtype=float)

        cls._copper_returns_expires_at = now + max(10, cls.MACRO_CACHE_TTL_SECONDS)
        return cls._copper_returns_cache

    @staticmethod
    def is_volatile_or_trending(ticker: str) -> bool:
        """DEPRECATED: Use get_comprehensive_data().is_active instead."""
        return True
