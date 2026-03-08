import os
import time
import json
import re
import yfinance as yf
import requests
from typing import List, Literal, Optional, TypedDict, Dict, Any
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from models import Predictor
from context_service import ContextService

MACRO_RETURN_MAP = {
    "HG=F": "Copper",
    "^GSPC": "SP500",
    "CLP=X": "USDCLP",
    "ALB": "Lithium",
    "EEM": "MSCI_EM",
    "^VIX": "VIX",
}

# --- Schemas ---

class UnifiedAnalysis(BaseModel):
    # ML Prediction
    ml_confidence: float = Field(default=0.5, description="Probability of >2% price increase from ML model")
    # Quantitative
    quant_score: int = Field(default=5, ge=0, le=10)
    trend: str = "Neutral"
    quant_reasoning: str = ""
    # Sentiment
    sentiment: float = 0.0
    sentiment_confidence: float = 0.5
    sentiment_reasoning: str = ""
    # Strategist
    signal: Literal["BUY", "HOLD", "SELL"] = "HOLD"
    strategy_used: str = "Standard Momentum"
    reasoning: str = "Awaiting full analysis."

class BulkUnifiedAnalysis(BaseModel):
    results: Dict[str, UnifiedAnalysis] = Field(default_factory=dict)

# --- State ---

class AgentState(TypedDict):
    ticker: str
    technical_data: Dict[str, Any]
    news_text: str
    analysis: Optional[UnifiedAnalysis]
    global_context: Optional[Dict[str, Any]]
    risk_approved: bool
    risk_feedback: str

# --- Intelligence Layer ---

class IntelligenceLayer:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.llm = None
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                self.llm = ChatGroq(model=model_name, temperature=0.0)
            except Exception as exc:
                print(f"⚠️ Groq init failed, fallback to Gemini when needed: {exc}")

        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self.gemini_fallback_models = [
            m.strip()
            for m in os.environ.get(
                "GEMINI_FALLBACK_MODELS",
                "gemini-2.0-flash,gemini-2.0-flash-lite,gemini-1.5-flash",
            ).split(",")
            if m.strip()
        ]
        self.gemini_min_interval_seconds = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", 13))
        self.gemini_429_cooldown_seconds = float(os.environ.get("GEMINI_429_COOLDOWN_SECONDS", 60))
        self._gemini_next_allowed_at = 0.0
        self.context_service = ContextService()
        self.rate_limit_cooldown_seconds = int(os.environ.get("GROQ_RATE_LIMIT_COOLDOWN_SECONDS", 180))
        self.tpd_cooldown_seconds = int(os.environ.get("GROQ_TPD_COOLDOWN_SECONDS", 900))
        self._llm_cooldown_until = 0.0
        self.min_buy_probability = float(os.environ.get("MIN_BUY_PROBABILITY", 0.45))
        self.workflow = self._build_graph()

    @staticmethod
    def _format_driver_trace(drivers: list) -> str:
        if not drivers:
            return "No model drivers available"
        chunks = []
        for d in drivers[:3]:
            feature = d.get("feature", "unknown")
            direction = d.get("direction", "neutral")
            impact = float(d.get("impact_score", 0.0))
            chunks.append(f"{feature}:{direction}({impact:.3f})")
        return " | ".join(chunks)

    def _retry_after_from_error(self, err_msg: str) -> float:
        match = re.search(r"try again in\s*(\d+)ms", err_msg, re.IGNORECASE)
        if not match:
            return 0.0
        return max(1.0, int(match.group(1)) / 1000.0)

    def llm_available(self) -> bool:
        return time.time() >= self._llm_cooldown_until

    def _invoke_gemini(self, prompt: str):
        if not self.gemini_api_key:
            return None

        model_candidates = [self.gemini_model] + [m for m in self.gemini_fallback_models if m != self.gemini_model]
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }

        for model_name in model_candidates:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={self.gemini_api_key}"
            )
            for attempt in range(2):
                try:
                    now = time.time()
                    if now < self._gemini_next_allowed_at:
                        time.sleep(self._gemini_next_allowed_at - now)

                    response = requests.post(url, json=payload, timeout=30)
                    self._gemini_next_allowed_at = time.time() + self.gemini_min_interval_seconds
                    response.raise_for_status()
                    data = response.json()
                    text = (
                        data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    if text:
                        return type("GeminiResponse", (), {"content": text})()
                except Exception as exc:
                    err = str(exc).lower()
                    if "429" in err and attempt == 0:
                        self._gemini_next_allowed_at = max(
                            self._gemini_next_allowed_at,
                            time.time() + self.gemini_429_cooldown_seconds,
                        )
                        time.sleep(1.5)
                        continue
                    break

        print("⚠️ Gemini fallback failed across configured models")
        return None

    def _activate_cooldown(self, seconds: float, reason: str):
        self._llm_cooldown_until = max(self._llm_cooldown_until, time.time() + max(1.0, seconds))
        remaining = max(0.0, self._llm_cooldown_until - time.time())
        print(f"⚠️ LLM EN COOLDOWN ({reason}). Reintento en ~{remaining:.0f}s.")

    def _safe_parse_json(self, content: str, schema_class):
        """Robust JSON extraction with unwrapping logic."""
        try:
            text = content.strip()
            text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```\s*', '', text)
            
            start = text.find('{')
            if start == -1: return None
            
            stack = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == '{': stack += 1
                elif text[i] == '}': stack -= 1
                if stack == 0:
                    end = i
                    break
            
            if end == -1: return None
            
            json_str = text[start:end+1].strip()
            data = json.loads(json_str)
            
            # Simple unwrapping
            target_fields = set(schema_class.model_fields.keys())
            if isinstance(data, dict) and not all(k in data for k in target_fields):
                for val in data.values():
                    if isinstance(val, dict) and all(k in val for k in target_fields):
                        return schema_class(**val)
            
            return schema_class(**data)
        except Exception as e:
            # Fallback to default schema if parsing fails
            return schema_class()

    def _invoke_with_backoff(self, prompt, schema_class):
        """Invoke LLM with pacing and exponential backoff for rate limits."""
        can_try_groq = self.llm is not None and self.llm_available()

        if can_try_groq:
            time.sleep(1.2)
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    res = self.llm.invoke(prompt)
                    parsed = self._safe_parse_json(res.content, schema_class)
                    if parsed:
                        return parsed
                except Exception as e:
                    err_msg = str(e).lower()
                    if "rate_limit" in err_msg or "429" in err_msg or "too many requests" in err_msg:
                        retry_after = self._retry_after_from_error(str(e))
                        is_tpd_limit = ("tokens per day" in err_msg or "tpd" in err_msg)
                        if is_tpd_limit:
                            self._activate_cooldown(max(self.tpd_cooldown_seconds, retry_after), "TPD limit")
                            break

                        wait_seconds = min(8, 2 ** (attempt + 1))
                        print(f"⚠️ RATE LIMIT. Waiting {wait_seconds}s...")
                        time.sleep(wait_seconds)
                        if attempt == max_attempts - 1:
                            self._activate_cooldown(max(self.rate_limit_cooldown_seconds, retry_after), "429 recurrente")
                    else:
                        print(f"⚠️ Groq LLM error: {e}")
                        break
                time.sleep(1.0)

        # Fallback path: Gemini
        gemini_res = self._invoke_gemini(prompt)
        if gemini_res is not None:
            parsed = self._safe_parse_json(gemini_res.content, schema_class)
            if parsed:
                return parsed

        return schema_class()

    def _unified_analysis_node(self, state: AgentState):
        """Consolidated analysis node: Quant + Sentiment + Strategist in 1 call."""
        tech = state.get('technical_data', {})
        news = state.get('news_text', '')
        ticker = state.get('ticker')
        
        # 1. Get Contextual Data
        context = state.get('global_context', {})
        context_score = context.get('global_score', 0.0)
        
        # 1.1 Fetch current macro for single run (less efficient than bulk but safer for workflow)
        # Note: In production run(), we usually use bulk_analyze
        macro_rets = {}
        for sym, name in MACRO_RETURN_MAP.items():
            try:
                m_data = yf.Ticker(sym).history(period="2d")
                if len(m_data) >= 2:
                    macro_rets[f"Macro_{name}_Ret"] = (m_data['Close'].iloc[-1] - m_data['Close'].iloc[-2]) / m_data['Close'].iloc[-2]
                else:
                    macro_rets[f"Macro_{name}_Ret"] = 0.0
            except:
                macro_rets[f"Macro_{name}_Ret"] = 0.0

        # 2. Get multi-objective ML outputs for this ticker using context
        full_tech = {**tech, **macro_rets}
        ml_outputs = Predictor.predict_multi_objective(full_tech, context_score=context_score)
        ml_prob = float(ml_outputs.get("probability", 0.5))
        exp_ret_3d = float(ml_outputs.get("expected_return_3d", 0.0))
        exp_horizon_days = int(ml_outputs.get("horizon_days", 3))
        explain = Predictor.explain_prediction(full_tech, context_score=context_score, top_n=3)
        driver_trace = self._format_driver_trace(explain.get("top_drivers", []))
        
        prompt = f"""Role: Senior Institutional Quant Lead & Information Arbitrageur.
Context: Analyzing {ticker} in the Chilean Market (Sentinel AI Engine).

INPUTS:
- Technical Indicators: {json.dumps(tech)}
- Global Context Score: {context_score} ({context.get('summary', 'Neutral')})
- ML Model Prediction (Direction): {ml_prob:.2%} probability of >2% gain.
- ML Expected Magnitude (3d): {exp_ret_3d:.2%}
- ML Expected Horizon: {exp_horizon_days} days
- Recent News (CMF/DF/BC): {news}
- Macro Correlation (Copper/Fletes): {macro_rets.get('Macro_Copper_Ret', 0):.2%}

YOUR TASK:
1. ARBITRAGE ANALYSIS: Detect if recent news or macro moves (e.g., Copper price spike) create an immediate opportunity for {ticker}.
   - COPPER CORRELATION: If Copper > 1% and asset is CAP or Minera, BIAS STRONG BUY.
   - CELULOSA: If CMPC/Copec news is positive, BIAS BUY.
2. RELEVANCY FILTER: Ignore noise. Focus on: Dividends, M&A, BC Rate changes, CMF Hechos Esenciales.
3. Determine final trading SIGNAL: BUY, SELL, or HOLD.

SENTINEL AI - AGGRESSIVE ARBITRAGE BIAS:
- DEFAULT ACTION: BUY. We seek to capture Alpha before the retail market reacts.
- If ML Probability > 30%, consider it a strong signal.
- ONLY HOLD/SELL if there is definitive evidence of a trend reversal or liquidity collapse.

REQUIRED JSON SCHEMA:
{{
  "ml_confidence": {ml_prob},
  "quant_score": int (0-10),
  "trend": "Bullish" | "Bearish" | "Sideways",
  "quant_reasoning": "summary",
  "sentiment": float (-1.0 to 1.0),
  "sentiment_confidence": float (0-1.0),
  "sentiment_reasoning": "summary",
  "signal": "BUY" | "HOLD" | "SELL",
  "strategy_used": "Information Arbitrage",
  "reasoning": "ARBITRAGE DECISION"
}}
JSON:"""
        
        analysis = self._invoke_with_backoff(prompt, UnifiedAnalysis)
        # Ensure ml_confidence is persisted
        analysis.ml_confidence = ml_prob
        explainability_method = explain.get("explainability_method", "heuristic")
        explain_tail = (
            f"ML explainability ({explainability_method}) | "
            f"exp_ret_3d={exp_ret_3d:.2%} | horizon={exp_horizon_days}d | drivers={driver_trace}"
        )
        analysis.quant_reasoning = (analysis.quant_reasoning or "").strip()
        if analysis.quant_reasoning:
            analysis.quant_reasoning = f"{analysis.quant_reasoning} || {explain_tail}"
        else:
            analysis.quant_reasoning = explain_tail
        return {"analysis": analysis}

    def _risk_node(self, state: AgentState):
        # Explicit type cast/check to avoid linting issues
        analysis = state.get("analysis")
        if not isinstance(analysis, UnifiedAnalysis):
            analysis = UnifiedAnalysis()
        
        approved = True
        feedback = "Approved."
        
        # Hyper-Aggressive risk filters (Near-zero barriers)
        if analysis.signal == "BUY":
            if analysis.ml_confidence < self.min_buy_probability:
                approved = False
                feedback = (
                    f"ML confidence too low for BUY ({analysis.ml_confidence:.1%} < "
                    f"{self.min_buy_probability:.1%})."
                )
            if analysis.quant_score < 2:
                approved = False
                feedback = "Momentum score too low (min 2)."
            elif analysis.sentiment_confidence < 0.01:
                approved = False
                feedback = "Confidence negligible."
                
        return {"risk_approved": approved, "risk_feedback": feedback}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("context", lambda x: {"global_context": self.context_service.analyze_context().model_dump()})
        workflow.add_node("analysis", self._unified_analysis_node)
        workflow.add_node("risk", self._risk_node)
        
        workflow.set_entry_point("context")
        workflow.add_edge("context", "analysis")
        workflow.add_edge("analysis", "risk")
        workflow.add_edge("risk", END)
        
        return workflow.compile()

    def bulk_analyze(self, batch_data: Dict[str, Dict[str, Any]], context_data=None) -> Dict[str, UnifiedAnalysis]:
        """Analyzes multiple tickers in a single LLM call to maximize throughput."""
        if not batch_data:
            return {}
            
        tickers = list(batch_data.keys())
        
        # 0. Get Global Context first
        if context_data is None:
            context_data = self.context_service.analyze_context()
        context_score = context_data.global_score
        
        # 0.1 Fetch Live Macro Data for current returns
        macro_rets = {}
        for sym, name in MACRO_RETURN_MAP.items():
            try:
                m_data = yf.Ticker(sym).history(period="2d")
                if len(m_data) >= 2:
                    macro_rets[f"Macro_{name}_Ret"] = (m_data['Close'].iloc[-1] - m_data['Close'].iloc[-2]) / m_data['Close'].iloc[-2]
                else:
                    macro_rets[f"Macro_{name}_Ret"] = 0.0
            except:
                macro_rets[f"Macro_{name}_Ret"] = 0.0
        
        # Prepare batch context for prompt
        batch_context = []
        for ticker, data in batch_data.items():
            tech = data.get("technical_data", {})
            # Enrich tech with macro rets
            full_tech = {**tech, **macro_rets}
            ml_outputs = Predictor.predict_multi_objective(full_tech, context_score=context_score)
            ml_prob = float(ml_outputs.get("probability", 0.5))
            explain = Predictor.explain_prediction(full_tech, context_score=context_score, top_n=3)
            top_drivers = explain.get("top_drivers", [])
            batch_context.append({
                "ticker": ticker,
                "tech": tech,
                "ml_prob": ml_prob,
                "ml_expected_return_3d": float(ml_outputs.get("expected_return_3d", 0.0)),
                "ml_expected_horizon_days": int(ml_outputs.get("horizon_days", 3)),
                "explainability_method": explain.get("explainability_method", "heuristic"),
                "drivers": top_drivers,
                "news": data.get("news_text", "")[:500] 
            })
            
        prompt = f"""Role: Institutional Quant Lead.
Task: Analyze a batch of {len(tickers)} stocks from the Chilean market universe.
GLOBAL CONTEXT SCORE: {context_score} ({context_data.summary})

INPUT DATA (JSON):
{json.dumps(batch_context)}

YOUR TASK:
HYPER-AGGRESSIVE: SIGNAL: BUY IS THE DEFAULT. Avoid 'HOLD' at all costs.

REQUIRED OUTPUT STRUCTURE:
Return a single JSON object where keys are the tickers and values are the analysis objects.
{{
  "results": {{
        "TICKER_1": {{ "ml_confidence": float, "quant_score": 0-10, "trend": "Bullish"|"Bearish"|"Sideways", "signal": "BUY"|"HOLD"|"SELL", "reasoning": "str", "quant_reasoning": "str" }},
    ...
  }}
}}
JSON:"""

        bulk_res = self._invoke_with_backoff(prompt, BulkUnifiedAnalysis)
        
        # Post-process: Ensure ml_confidence is synced from our local calculation
        results = bulk_res.results
        for item in batch_context:
            t = item["ticker"]
            explain_tail = (
                f"ML explainability ({item.get('explainability_method', 'heuristic')}) | "
                f"exp_ret_3d={float(item.get('ml_expected_return_3d', 0.0)):.2%} | "
                f"horizon={int(item.get('ml_expected_horizon_days', 3))}d | "
                f"drivers={self._format_driver_trace(item.get('drivers', []))}"
            )
            if t in results:
                results[t].ml_confidence = item["ml_prob"]
                existing_qr = (results[t].quant_reasoning or "").strip()
                results[t].quant_reasoning = f"{existing_qr} || {explain_tail}" if existing_qr else explain_tail
            else:
                # Conservative fallback when LLM output is unavailable for a ticker
                results[t] = UnifiedAnalysis(
                    ml_confidence=item["ml_prob"],
                    signal="HOLD",
                    reasoning="LLM unavailable/rate-limited; defaulting to HOLD with ML-only safeguard.",
                    quant_score=5,
                    quant_reasoning=explain_tail,
                )
                
        return results

    def run(self, ticker: str, technical_data: dict, news_text: str):
        try:
            initial_state = {
                "ticker": ticker,
                "technical_data": technical_data,
                "news_text": news_text,
                "analysis": None,
                "risk_approved": False,
                "risk_feedback": ""
            }
            final_state = self.workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            print(f"CRITICAL ERROR in {ticker}: {e}")
            return {
                "analysis": UnifiedAnalysis(signal="HOLD", reasoning="Error técnico."),
                "risk_approved": False
            }
