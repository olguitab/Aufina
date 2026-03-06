import os
import json
import time
import random
import re
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import requests

class ContextAnalysis(BaseModel):
    global_score: float = Field(description="Score from -1.0 (Catastrophic) to 1.0 (Excellent) for the market context")
    event_type: str = Field(description="Category of the event: Law, Catastrophe, Economic, Political, etc.")
    summary: str = Field(description="Brief summary of the global context")
    impact_level: str = Field(description="Low, Medium, High, Extreme")

class ContextService:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.llm = ChatGroq(model=model_name, temperature=0.0)
        self.news_api_key = os.environ.get("NEWS_API_KEY") # Optional, defaults to generic search if not present
        self.context_cache_ttl = int(os.environ.get("CONTEXT_CACHE_TTL_SECONDS", 300))
        self.rate_limit_cooldown_seconds = int(os.environ.get("GROQ_RATE_LIMIT_COOLDOWN_SECONDS", 180))
        self.tpd_cooldown_seconds = int(os.environ.get("GROQ_TPD_COOLDOWN_SECONDS", 900))
        self._cached_context: ContextAnalysis | None = None
        self._cache_expires_at = 0.0
        self._rate_limited_until = 0.0

    def _retry_after_from_error(self, err_msg: str) -> float:
        match = re.search(r"try again in\s*(\d+)ms", err_msg, re.IGNORECASE)
        if not match:
            return 0.0
        return max(1.0, int(match.group(1)) / 1000.0)

    def _activate_cooldown(self, seconds: float, reason: str):
        self._rate_limited_until = max(self._rate_limited_until, time.time() + max(1.0, seconds))
        remaining = max(0.0, self._rate_limited_until - time.time())
        print(f"⚠️ CONTEXTO EN COOLDOWN ({reason}). Reintento en ~{remaining:.0f}s.")

    def _invoke_with_backoff(self, prompt: str, max_attempts: int = 2):
        now = time.time()
        if now < self._rate_limited_until:
            wait_left = self._rate_limited_until - now
            raise RuntimeError(f"Context LLM cooldown active for {wait_left:.1f}s")

        last_error = None
        for attempt in range(max_attempts):
            try:
                # Light pacing to avoid burst traffic
                time.sleep(1.2)
                return self.llm.invoke(prompt)
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                is_rate_limit = ("rate_limit" in err_msg or "429" in err_msg or "too many requests" in err_msg)

                if not is_rate_limit and attempt == max_attempts - 1:
                    break

                if is_rate_limit:
                    retry_after = self._retry_after_from_error(str(e))
                    is_tpd_limit = ("tokens per day" in err_msg or "tpd" in err_msg)

                    if is_tpd_limit:
                        cooldown = max(self.tpd_cooldown_seconds, retry_after)
                        self._activate_cooldown(cooldown, "TPD limit")
                        break

                    wait_seconds = min(8, 2 ** (attempt + 1)) + random.uniform(0.0, 0.8)
                    print(f"⚠️ RATE LIMIT CONTEXTO. Reintentando en {wait_seconds:.1f}s...")
                    time.sleep(wait_seconds)

                    if attempt == max_attempts - 1:
                        self._activate_cooldown(max(self.rate_limit_cooldown_seconds, retry_after), "429 recurrente")
                else:
                    time.sleep(1.5)

        raise last_error if last_error else RuntimeError("Unknown LLM invoke error")

    def fetch_global_context(self) -> str:
        """Fetches high-level news about Chile and Global Economy."""
        # For MVP, we use a mix of known financial news endpoints or simulated high-impact headlines
        # In a real scenario, this would call a News API (e.g., NewsAPI.org or Google News)
        headlines = [
            "Chilean Congress discusses new tax reform affecting mining companies.",
            "Central Bank of Chile maintains interest rates amid inflation concerns.",
            "Global copper prices see a slight decline due to overseas demand softening.",
            "New environmental regulations proposed for energy sector in Chile."
        ]
        return "\n".join(headlines)

    def analyze_context(self) -> ContextAnalysis:
        """Analyzes the fetched news to provide a quantified context score."""
        now = time.time()
        if self._cached_context is not None and now < self._cache_expires_at:
            return self._cached_context

        context_text = self.fetch_global_context()
        
        prompt = f"""Role: Senior Macroeconomic Analyst.
Analyze the following news context for the Chilean Market (IPSA) and determine the overall market sentiment.
Provide a numerical score between -1.0 (extremely bearish/catastrophic) and 1.0 (extremely bullish).

News Headlines:
{context_text}

Analyze the impact of:
1. Legal/Regulatory changes (Laws).
2. Economical shifts (Inflation, Rates).
3. Catastrophic events or major disruptions.

Return ONLY a JSON object following this schema:
{{
    "global_score": float (-1.0 to 1.0),
    "event_type": "Law" | "Economic" | "Catastrophe" | "Political",
    "summary": "Short summary",
    "impact_level": "Low" | "Medium" | "High" | "Extreme"
}}
JSON:"""

        try:
            response = self._invoke_with_backoff(prompt)
            # Basic JSON extraction
            content = response.content.strip()
            
            # Use a robust extraction for the first JSON object
            start = content.find('{')
            if start == -1:
                raise ValueError("No JSON object found")
            
            # Simple brace counting
            stack = 0
            end = -1
            for i in range(start, len(content)):
                if content[i] == '{':
                    stack += 1
                elif content[i] == '}':
                    stack -= 1
                    if stack == 0:
                        end = i + 1
                        break
            
            if end == -1:
                raise ValueError("Incomplete JSON object")
                
            json_str = content[start:end]
            data = json.loads(json_str)
            # If data is wrapped in a list or another key, unwrap it
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            elif isinstance(data, dict) and "results" in data:
                data = data["results"]
            
            parsed = ContextAnalysis(**data)
            self._cached_context = parsed
            self._cache_expires_at = time.time() + self.context_cache_ttl
            return parsed
        except Exception as e:
            print(f"Error analyzing global context: {e}")
            fallback = ContextAnalysis(
                global_score=0.0,
                event_type="Neutral",
                summary="Unable to analyze context. Defaulting to neutral.",
                impact_level="Low"
            )
            self._cached_context = fallback
            self._cache_expires_at = time.time() + min(self.context_cache_ttl, 60)
            return fallback

if __name__ == "__main__":
    # Test current context
    from dotenv import load_dotenv
    load_dotenv()
    service = ContextService()
    analysis = service.analyze_context()
    print(analysis.model_dump_json(indent=2))
