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
from ingestion import NewsEngine


def _is_hosted_runtime() -> bool:
    return any(
        os.environ.get(flag)
        for flag in ("RENDER", "RENDER_SERVICE_ID", "RAILWAY_ENVIRONMENT", "K_SERVICE")
    )


def _env_int(name: str, local_default: int, hosted_default: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip() != "":
        return int(raw)
    if hosted_default is not None and _is_hosted_runtime():
        return int(hosted_default)
    return int(local_default)


def _env_float(name: str, local_default: float, hosted_default: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    if hosted_default is not None and _is_hosted_runtime():
        return float(hosted_default)
    return float(local_default)

class ContextAnalysis(BaseModel):
    global_score: float = Field(description="Score from -1.0 (Catastrophic) to 1.0 (Excellent) for the market context")
    event_type: str = Field(description="Category of the event: Law, Catastrophe, Economic, Political, etc.")
    summary: str = Field(description="Brief summary of the global context")
    impact_level: str = Field(description="Low, Medium, High, Extreme")

class ContextService:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.llm = None
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            try:
                self.llm = ChatGroq(model=model_name, temperature=0.0)
            except Exception as exc:
                print(f"⚠️ ContextService LLM unavailable at init: {exc}")
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
        self.gemini_min_interval_seconds = _env_float("GEMINI_MIN_INTERVAL_SECONDS", local_default=13, hosted_default=18)
        self.gemini_429_cooldown_seconds = _env_float("GEMINI_429_COOLDOWN_SECONDS", local_default=60, hosted_default=180)
        self.gemini_max_models_per_call = _env_int("GEMINI_MAX_MODELS_PER_CALL", local_default=2, hosted_default=1)
        self.gemini_max_attempts = _env_int("GEMINI_MAX_ATTEMPTS", local_default=2, hosted_default=1)
        self.allow_gemini_fallback_when_groq_limited = os.environ.get(
            "ALLOW_GEMINI_WHEN_GROQ_LIMITED",
            "0" if _is_hosted_runtime() else "1",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._gemini_next_allowed_at = 0.0
        self.news_api_key = os.environ.get("NEWS_API_KEY")
        self.news_api_url = os.environ.get("NEWS_API_URL", "https://newsapi.org/v2/everything")
        self.news_api_query = os.environ.get(
            "NEWS_API_QUERY",
            'Chile OR IPSA OR "Banco Central de Chile" OR cobre OR minería',
        )
        self.news_api_page_size = int(os.environ.get("NEWS_API_PAGE_SIZE", 8))
        self.news_engine = NewsEngine()
        self.context_cache_ttl = _env_int("CONTEXT_CACHE_TTL_SECONDS", local_default=300, hosted_default=900)
        self.rate_limit_cooldown_seconds = _env_int("GROQ_RATE_LIMIT_COOLDOWN_SECONDS", local_default=120, hosted_default=180)
        self.tpd_cooldown_seconds = _env_int("GROQ_TPD_COOLDOWN_SECONDS", local_default=900, hosted_default=900)
        self.enable_headline_sentiment_llm = os.environ.get(
            "CONTEXT_ENABLE_HEADLINE_SENTIMENT_LLM",
            "0" if _is_hosted_runtime() else "1",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._cached_context: ContextAnalysis | None = None
        self._cache_expires_at = 0.0
        self._rate_limited_until = 0.0

    def _score_headlines_sentiment(self, context_text: str) -> float:
        """Scores aggregated headlines sentiment in Spanish from -1.0 to 1.0."""
        prompt = f"""Role: Analista de Sentimiento Financiero para mercado chileno.
Evalúa los siguientes titulares y devuelve SOLO un JSON.

Titulares:
{context_text}

Formato requerido:
{{
  "sentiment_score": float (-1.0 a 1.0),
  "rationale": "resumen breve"
}}
JSON:"""

        try:
            response = self._invoke_with_backoff(prompt)
            content = (response.content or "").strip()
            start = content.find('{')
            if start == -1:
                return 0.0

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
                return 0.0

            payload = json.loads(content[start:end])
            score = float(payload.get("sentiment_score", 0.0))
            return max(-1.0, min(1.0, score))
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_headline(line: str) -> str:
        cleaned = re.sub(r"\s+", " ", line or "").strip()
        cleaned = cleaned.lstrip("-|•*")
        return cleaned

    @staticmethod
    def _extract_lines(raw_text: str) -> List[str]:
        if not raw_text:
            return []
        lines = [ContextService._normalize_headline(x) for x in raw_text.splitlines()]
        return [x for x in lines if len(x) >= 24 and "error" not in x.lower()]

    def _fetch_news_api_headlines(self) -> List[str]:
        if not self.news_api_key:
            return []

        params = {
            "q": self.news_api_query,
            "language": "es",
            "sortBy": "publishedAt",
            "pageSize": self.news_api_page_size,
            "apiKey": self.news_api_key,
        }

        try:
            resp = requests.get(self.news_api_url, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("articles", []) if isinstance(payload, dict) else []

            headlines: List[str] = []
            for article in articles:
                title = self._normalize_headline(article.get("title", ""))
                source = (article.get("source") or {}).get("name", "NewsAPI")
                if title and len(title) >= 20:
                    headlines.append(f"NEWSAPI: {title} ({source})")
            return headlines
        except Exception as exc:
            print(f"⚠️ Error fetching NewsAPI headlines: {exc}")
            return []

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
        if self.llm is None:
            if not self.allow_gemini_fallback_when_groq_limited:
                raise RuntimeError("Context LLM unavailable and Gemini fallback disabled during cooldown")
            gemini_res = self._invoke_gemini(prompt)
            if gemini_res is not None:
                return gemini_res
            raise RuntimeError("Context LLM unavailable (missing key or init failure)")

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
                is_auth_failure = ("401" in err_msg or "unauthorized" in err_msg or "invalid api key" in err_msg)
                is_rate_limit = ("rate_limit" in err_msg or "429" in err_msg or "too many requests" in err_msg)

                if is_auth_failure:
                    self._activate_cooldown(self.tpd_cooldown_seconds, "auth failure (401)")
                    self.llm = None
                    gemini_res = self._invoke_gemini(prompt)
                    if gemini_res is not None:
                        return gemini_res
                    break

                if not is_rate_limit and attempt == max_attempts - 1:
                    break

                if is_rate_limit:
                    retry_after = self._retry_after_from_error(str(e))
                    is_tpd_limit = ("tokens per day" in err_msg or "tpd" in err_msg)

                    if is_tpd_limit:
                        cooldown = max(self.tpd_cooldown_seconds, retry_after)
                        self._activate_cooldown(cooldown, "TPD limit")
                        if not self.allow_gemini_fallback_when_groq_limited:
                            break
                        break

                    wait_seconds = min(8, 2 ** (attempt + 1)) + random.uniform(0.0, 0.8)
                    print(f"⚠️ RATE LIMIT CONTEXTO. Reintentando en {wait_seconds:.1f}s...")
                    time.sleep(wait_seconds)

                    if attempt == max_attempts - 1:
                        self._activate_cooldown(max(self.rate_limit_cooldown_seconds, retry_after), "429 recurrente")
                        if not self.allow_gemini_fallback_when_groq_limited:
                            break
                else:
                    time.sleep(1.5)

        raise last_error if last_error else RuntimeError("Unknown LLM invoke error")

    def _invoke_gemini(self, prompt: str):
        if not self.gemini_api_key:
            return None

        now = time.time()
        if now < self._gemini_next_allowed_at:
            return None

        model_candidates = [self.gemini_model] + [m for m in self.gemini_fallback_models if m != self.gemini_model]
        model_candidates = model_candidates[: max(1, self.gemini_max_models_per_call)]
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }

        for model_name in model_candidates:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={self.gemini_api_key}"
            )
            for attempt in range(max(1, self.gemini_max_attempts)):
                try:
                    now = time.time()
                    if now < self._gemini_next_allowed_at:
                        return None

                    response = requests.post(url, json=payload, timeout=30)
                    if response.status_code == 429:
                        retry_after_header = response.headers.get("Retry-After")
                        retry_after = 0.0
                        try:
                            retry_after = float(retry_after_header) if retry_after_header else 0.0
                        except Exception:
                            retry_after = 0.0

                        cooldown = max(self.gemini_429_cooldown_seconds, retry_after, self.gemini_min_interval_seconds)
                        self._gemini_next_allowed_at = max(self._gemini_next_allowed_at, time.time() + cooldown)
                        print(f"⚠️ Context Gemini 429 on {model_name}. Cooldown ~{cooldown:.0f}s")
                        return None

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
                        return None
                    break

        print("⚠️ Context Gemini fallback failed across configured models")
        return None

    def fetch_global_context(self) -> str:
        """Fetches high-level news about Chile and global macro context using real sources."""
        headlines: List[str] = []

        # 1) Local curated sources (CMF / BCCh / DF)
        try:
            local_news_blob = self.news_engine.fetch_latest_news()
            headlines.extend(self._extract_lines(local_news_blob))
        except Exception as exc:
            print(f"⚠️ Error fetching local context sources: {exc}")

        # 2) Optional NewsAPI enrichment
        headlines.extend(self._fetch_news_api_headlines())

        # De-duplicate while preserving order
        deduped = list(dict.fromkeys(headlines))

        if not deduped:
            deduped = [
                "Mercado chileno sin titulares procesables en este momento.",
                "Contexto macro neutral por ausencia de señales de alta convicción.",
            ]

        # Keep prompt compact to reduce token costs
        return "\n".join(deduped[:20])

    def analyze_context(self) -> ContextAnalysis:
        """Analyzes the fetched news to provide a quantified context score."""
        now = time.time()
        if self._cached_context is not None and now < self._cache_expires_at:
            return self._cached_context

        if self.llm is None:
            fallback = ContextAnalysis(
                global_score=0.0,
                event_type="Neutral",
                summary="LLM no configurado. Contexto neutral basado en fuentes de noticias.",
                impact_level="Low"
            )
            self._cached_context = fallback
            self._cache_expires_at = time.time() + min(self.context_cache_ttl, 120)
            return fallback

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

            # Blend macro summary score with optional direct headline sentiment score
            headline_sentiment = (
                self._score_headlines_sentiment(context_text)
                if self.enable_headline_sentiment_llm
                else float(parsed.global_score)
            )
            blended = (0.75 * float(parsed.global_score)) + (0.25 * float(headline_sentiment))
            parsed.global_score = max(-1.0, min(1.0, blended))

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
