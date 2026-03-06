import os
import requests
from bs4 import BeautifulSoup
import feedparser
from typing import List, Dict

class NewsEngine:
    """
    Sentinel AI Information Arbitrage Engine.
    Monitors CMF, Banco Central, and Diario Financiero for Alpha generation.
    """
    def __init__(self):
        self.sources = {
            "df": "https://www.df.cl/noticias/site/tax/port/all/rss_todas.xml",
            "cmf": "https://www.cmfchile.cl/portal/principal/613/w3-propertyvalue-18451.html", # Hechos Esenciales
            "bc": "https://www.bcentral.cl/web/banco-central/noticias"
        }
        
    def fetch_latest_news(self) -> str:
        """Consolidates news from all sources for LLM processing."""
        all_news = []
        all_news.append(self._fetch_df_rss())
        all_news.append(self._fetch_cmf_headlines())
        all_news.append(self._fetch_bc_headlines())
        
        # Add a mock "Hecho Esencial" for testing if requested
        if os.environ.get("SENTINEL_SIMULATE_ARBITRAGE") == "true":
            all_news.append("HECHO ESENCIAL: Sociedad Química y Minera de Chile (SQM-B) "
                           "anuncia acuerdo estratégico con el Estado para extensión de concesión hasta 2060.")
            
        return "\n\n".join([n for n in all_news if n])

    def _fetch_df_rss(self) -> str:
        """Fetches Diario Financiero RSS feed."""
        try:
            feed = feedparser.parse(self.sources["df"])
            entries = [f"DF: {e.title} - {e.summary[:200]}" for e in feed.entries[:5]]
            return "\n".join(entries)
        except Exception as e:
            print(f"Error fetching DF RSS: {e}")
            return "DF: No se pudo obtener noticias."

    def _fetch_cmf_headlines(self) -> str:
        """
        Scrapes CMF Hechos Esenciales portal.
        Note: In a production environment, this would use a more robust parser or API.
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(self.sources["cmf"], headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            # Look for typical CMF headline patterns
            headlines = [h.get_text().strip() for h in soup.find_all(['h3', 'a'], limit=10) if h.get_text()]
            relevant = [h for h in headlines if len(h) > 20][:5]
            return "CMF HECHOS ESENCIALES: " + " | ".join(relevant)
        except:
            return "CMF: Error al conectar con el portal de Hechos Esenciales."

    def _fetch_bc_headlines(self) -> str:
        """Fetches latest from Banco Central."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(self.sources["bc"], headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            headlines = [h.get_text().strip() for h in soup.select('h3, .title')[:3]]
            return "BANCO CENTRAL: " + " | ".join(headlines)
        except:
            return "BC: Error al conectar con Banco Central."

    def scrape_specific_article(self, url: str) -> str:
        """Deep scrape for detailed analysis."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            return ' '.join([p.get_text() for p in paragraphs])[:3000]
        except:
            return ""
