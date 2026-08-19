"""
News fetcher — pulls recent articles about a ticker from multiple sources.

Returns raw article text.  The Sentiment agent analyses this text;
it never invents news that wasn't fetched here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

import httpx
import structlog
import yfinance as yf
from bs4 import BeautifulSoup

from app.config.settings import get_settings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class NewsArticle:
    title: str
    body: str
    url: str
    published_at: datetime | None
    source: str


async def _fetch_url(client: httpx.AsyncClient, url: str) -> str:
    """Download page HTML and extract article body via BeautifulSoup."""
    try:
        resp = await client.get(url, timeout=10, follow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove navigation/ads/scripts
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        # Trim to reasonable size
        return text[:4000]
    except Exception as exc:
        log.warning("news_fetch_url_error", url=url, error=str(exc))
        return ""


async def fetch_news(ticker: str) -> list[NewsArticle]:
    """
    Fetch recent news from yfinance (Yahoo Finance RSS) and return
    structured articles with body text.
    """
    settings = get_settings()
    max_articles = settings.news_max_articles
    articles: list[NewsArticle] = []

    def _get_yf_news() -> list[dict]:
        try:
            tk = yf.Ticker(ticker)
            return tk.news or []
        except Exception as exc:
            log.warning("yfinance_news_error", ticker=ticker, error=str(exc))
            return []

    raw_news = await asyncio.get_event_loop().run_in_executor(None, _get_yf_news)
    raw_news = raw_news[:max_articles]

    if not raw_news:
        log.info("no_news_found", ticker=ticker)
        return []

    async with httpx.AsyncClient(
        headers={"User-Agent": "AlphaAgents/1.0 (research tool)"},
        timeout=15,
    ) as client:
        tasks = []
        metas = []
        for item in raw_news:
            url = item.get("link", "")
            if url:
                tasks.append(_fetch_url(client, url))
                metas.append(item)

        bodies = await asyncio.gather(*tasks, return_exceptions=True)

    for meta, body in zip(metas, bodies, strict=False):
        if isinstance(body, Exception) or not body:
            body = ""
        title = meta.get("title", "")
        pub_ts = meta.get("providerPublishTime")
        pub_dt = datetime.utcfromtimestamp(pub_ts) if pub_ts else None
        articles.append(
            NewsArticle(
                title=title,
                body=str(body),
                url=meta.get("link", ""),
                published_at=pub_dt,
                source=meta.get("publisher", "Yahoo Finance"),
            )
        )

    log.info("news_fetched", ticker=ticker, count=len(articles))
    return articles
