import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SentimentAgent/2.0)"}
_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside"]


def is_url(text: str) -> bool:
    return text.strip().startswith(("http://", "https://"))


def fetch_content(url: str) -> str:
    logger.info("Fetching content from %s", url)

    with httpx.Client(timeout=15, follow_redirects=True) as client:
        response = client.get(url.strip(), headers=_HEADERS)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    logger.info("Extracted %d characters from %s", len(text), url)
    return text
