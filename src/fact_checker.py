"""
src/fact_checker.py — Live Web Fact-Checking & News Verification Layer.

Dynamically cross-checks modern news headlines and articles against real-time,
credible news reporting (Reuters, AP, BBC, NYT, Bloomberg, etc.) and fact-checking
databases (PolitiFact, Snopes, FactCheck.org, etc.) via live Google News RSS feeds.

Provides:
1. Extraction of search queries from news headlines/articles.
2. Real-time news corroboration across established journalistic agencies.
3. Fact-check debunk detection (identifying whether claims are marked hoax/myth).
4. Multi-source evidence scoring to prevent both false positives and false negatives on recent news.
"""
import os
import sys
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Curated list of established, high-credibility news sources and wire services
CREDIBLE_OUTLETS = {
    "reuters", "associated press", "ap news", "bbc", "bbc news", "bloomberg",
    "the new york times", "new york times", "the washington post", "washington post",
    "the wall street journal", "wall street journal", "wsj", "financial times",
    "the guardian", "npr", "abc news", "cbs news", "nbc news", "pbs", "afp",
    "politico", "time", "the hill", "nature", "science", "nasa", "who", "cdc",
    "forbes", "fortune", "cnbc", "axios", "propublica", "the economist"
}

# Recognized fact-checking organizations
FACT_CHECK_ORGANIZATIONS = {
    "snopes", "politifact", "factcheck.org", "reuters fact check",
    "ap news fact check", "lead stories", "check your fact", "full fact"
}

# Strong debunking keywords in headlines
DEBUNK_KEYWORDS = {
    "fact check", "debunk", "debunked", "hoax", "false claim",
    "myth", "myths", "not true", "fake news", "no evidence",
    "misleading", "disproven", "untrue", "unfounded"
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "of", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "should", "could", "can", "may", "might", "must", "that", "this", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "their", "his", "her", "its",
    "our", "my", "your", "said", "says", "breaking", "news", "report", "reports"
}


class LiveNewsFactChecker:
    _instance = None

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    @classmethod
    def get_instance(cls, timeout: int = 5):
        if cls._instance is None:
            cls._instance = cls(timeout=timeout)
        return cls._instance

    def extract_search_query(self, text: str) -> str:
        """Extracts the most salient search query from headline or article."""
        if not text or not text.strip():
            return ""

        # Take first sentence or up to first 120 characters
        first_line = text.strip().split("\n")[0]
        sentence = re.split(r"[.!?]", first_line)[0].strip()
        if not sentence:
            sentence = text.strip()[:120]

        # Clean punctuation but preserve alphanumeric words
        clean_words = [w.strip(".,!?\"'()[]:;–—") for w in sentence.split()]
        filtered = [w for w in clean_words if len(w) > 2 and w.lower() not in STOPWORDS]

        # Use up to top 8 meaningful words
        query = " ".join(filtered[:8])
        return query if query else sentence[:60]

    def fetch_news_rss(self, query: str) -> List[Dict[str, str]]:
        """Queries Google News RSS and returns parsed article dictionaries."""
        if not query:
            return []

        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                xml_content = response.read()
                root = ET.fromstring(xml_content)
                items = root.findall(".//item")

                results = []
                for item in items[:10]:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate")
                    source_elem = item.find("source")

                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                    pub_date = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""
                    source = source_elem.text.strip() if source_elem is not None and source_elem.text else ""

                    if title:
                        results.append({
                            "title": title,
                            "link": link,
                            "pub_date": pub_date,
                            "source": source
                        })
                return results
        except Exception:
            return []

    def verify_claim(self, text: str) -> Dict[str, Any]:
        """
        Cross-checks the input against real-time news sources.
        Returns verification status, matching credible outlets, debunk flags, and score.
        """
        query = self.extract_search_query(text)
        if not query:
            return {
                "status": "UNAVAILABLE",
                "query_used": "",
                "total_articles_found": 0,
                "credible_sources_count": 0,
                "credible_sources": [],
                "debunk_articles_count": 0,
                "debunk_articles": [],
                "corroboration_score": 0.0,
                "summary": "Could not extract search query from input."
            }

        # Query 1: Direct news corroboration
        news_articles = self.fetch_news_rss(query)

        # Query 2: Dedicated fact-checking search
        fc_query = f"{query} fact check"
        fc_articles = self.fetch_news_rss(fc_query)

        credible_sources = []
        debunk_articles = []

        # Separate debunks from pure news
        for art in news_articles + fc_articles:
            src_lower = art.get("source", "").lower()
            title_lower = art.get("title", "").lower()

            is_fact_checker = any(fc in src_lower for fc in FACT_CHECK_ORGANIZATIONS)
            has_debunk = any(re.search(rf"\b{re.escape(k)}\b", title_lower) for k in DEBUNK_KEYWORDS)
            is_credible = any(cred in src_lower for cred in CREDIBLE_OUTLETS)

            if is_fact_checker or has_debunk:
                if art not in debunk_articles:
                    debunk_articles.append(art)
            elif is_credible:
                if art not in credible_sources:
                    credible_sources.append(art)

        # Calculate corroboration metrics
        total_news_found = len(news_articles) + len(fc_articles)
        num_credible = len(credible_sources)
        num_debunks = len(debunk_articles)

        if num_debunks >= 1:
            status = "REFUTED"
            corroboration_score = 0.05
            top_debunk = debunk_articles[0]["title"]
            top_src = debunk_articles[0].get("source", "Fact-checker")
            summary = (
                f"Refuted by independent fact-checking: '{top_debunk}' ({top_src})."
            )
        elif num_credible >= 2:
            status = "CORROBORATED"
            corroboration_score = 0.95
            sources_names = ", ".join([s["source"] for s in credible_sources[:3]])
            summary = (
                f"Corroborated by {num_credible} independent mainstream news outlets ({sources_names})."
            )
        elif num_credible == 1:
            status = "PARTIALLY_CORROBORATED"
            corroboration_score = 0.65
            summary = (
                f"Single source mention in {credible_sources[0]['source']}, but lacks corroboration from a second independent agency."
            )
        elif total_news_found == 0:
            status = "NO_EVIDENCE"
            corroboration_score = 0.20
            summary = (
                "No live news reporting was found across major wire services or mainstream news."
            )
        else:
            status = "NO_EVIDENCE"
            corroboration_score = 0.40
            summary = (
                f"Found {total_news_found} general web mentions, but none from established wire services or fact-checkers."
            )

        return {
            "status": status,
            "evidence": status,
            "query_used": query,
            "total_articles_found": total_news_found,
            "credible_sources_count": num_credible,
            "credible_sources": credible_sources[:4],
            "debunk_articles_count": num_debunks,
            "debunk_articles": debunk_articles[:3],
            "corroboration_score": corroboration_score,
            "summary": summary
        }


def verify_news_live(text: str) -> Dict[str, Any]:
    """Helper functional interface for live fact-checking."""
    checker = LiveNewsFactChecker.get_instance()
    return checker.verify_claim(text)
