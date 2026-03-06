import os
from datetime import datetime, time, timedelta
import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

class MarketData:
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
    def get_comprehensive_data(ticker: str) -> Dict[str, Any]:
        """Fetches all necessary data (Technicals, News, Status) in a single workflow.
        Reduces YFinance overhead significantly.
        """
        try:
            stock = yf.Ticker(ticker)
            # 1. Fetch historical data (3 months handles everything)
            hist = stock.history(period="3mo")
            if hist.empty or len(hist) < 2:
                # Delisted or invalid symbol: Return a neutral state to avoid hanging
                return {"is_active": False, "current_price": 0, "error": "No data/Delisted"}

            close = hist['Close']
            current = close.iloc[-1]
            prev_close = close.iloc[-2]
            
            # --- Technical Indicators ---
            ma20 = close.rolling(window=20).mean().iloc[-1]
            ma50 = close.rolling(window=50).mean().iloc[-1]
            
            daily_return = ((current - prev_close) / prev_close) * 100
            five_day_return = ((current - close.iloc[-6]) / close.iloc[-6]) * 100 if len(close) >= 6 else 0.0
            dist_ma20 = ((current - ma20) / ma20) * 100
            volatility = close.pct_change().rolling(window=20).std().iloc[-1]
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
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

            # --- ADV (Average Daily Volume - 20d) ---
            adv_20d = hist['Volume'].rolling(window=20).mean().iloc[-1]
            
            # --- Commodity Correlation: Copper (HG=F) ---
            copper_ret = 0.0
            try:
                copper = yf.Ticker("HG=F")
                c_hist = copper.history(period="2d")
                if len(c_hist) >= 2:
                    copper_ret = (c_hist['Close'].iloc[-1] - c_hist['Close'].iloc[-2]) / c_hist['Close'].iloc[-2]
            except:
                pass

            return {
                "is_active": is_active,
                "current_price": current,
                "technical_data": {
                    "CurrentPrice": current,
                    "MA20": ma20,
                    "MA50": ma50,
                    "RSI_14": current_rsi,
                    "ATR": atr,
                    "ADV_20d": adv_20d,
                    "Copper_Return": copper_ret,
                    "MonthlyReturn_Pct": monthly_return,
                    "DailyReturn_Pct": daily_return,
                    "FiveDayReturn_Pct": five_day_return,
                    "Dist_MA20_Pct": dist_ma20,
                    "Volatility_20d": volatility,
                    "Trend": "Bullish" if ma20 > ma50 else "Bearish"
                },
                "news_text": news_text
            }
        except Exception as e:
            print(f"Error in comprehensive data for {ticker}: {e}")
            return {"is_active": True, "error": str(e)}

    @staticmethod
    def is_volatile_or_trending(ticker: str) -> bool:
        """DEPRECATED: Use get_comprehensive_data().is_active instead."""
        return True
