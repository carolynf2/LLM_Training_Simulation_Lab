import requests
import time
from bs4 import BeautifulSoup
from typing import Iterator, Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse
import logging
from tqdm import tqdm
import re


class WebScraper:
    """Web scraper with rate limiting and content extraction."""

    def __init__(self, delay: float = 1.0, timeout: int = 30, user_agent: Optional[str] = None):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

        if user_agent:
            self.session.headers.update({'User-Agent': user_agent})
        else:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

    def fetch_page(self, url: str) -> Optional[requests.Response]:
        """Fetch a single web page with error handling."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None

    def extract_text(self, html_content: str, remove_scripts: bool = True) -> str:
        """Extract clean text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')

        if remove_scripts:
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

        # Get text and clean up whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract all links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)

        return links

    def scrape_url(self, url: str, extract_links: bool = False) -> Dict[str, Any]:
        """Scrape a single URL and return structured data."""
        self.logger.info(f"Scraping: {url}")

        response = self.fetch_page(url)
        if not response:
            return {'url': url, 'error': 'Failed to fetch page'}

        # Extract metadata
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ''

        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description.get('content', '') if meta_description else ''

        # Extract main content
        text_content = self.extract_text(response.text)

        result = {
            'url': url,
            'title': title_text,
            'description': description,
            'text': text_content,
            'content_length': len(text_content),
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'scraped_at': time.time()
        }

        if extract_links:
            result['links'] = self.extract_links(response.text, url)

        return result

    def scrape_urls(self, urls: List[str], extract_links: bool = False) -> Iterator[Dict[str, Any]]:
        """Scrape multiple URLs with rate limiting."""
        for url in tqdm(urls, desc="Scraping URLs"):
            result = self.scrape_url(url, extract_links)
            yield result

            # Rate limiting
            if self.delay > 0:
                time.sleep(self.delay)

    def crawl_sitemap(self, sitemap_url: str) -> List[str]:
        """Extract URLs from XML sitemap."""
        response = self.fetch_page(sitemap_url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'xml')
        urls = []

        for loc in soup.find_all('loc'):
            urls.append(loc.get_text().strip())

        return urls

    def filter_content(self, text: str, min_length: int = 100,
                      exclude_patterns: Optional[List[str]] = None) -> bool:
        """Filter content based on quality criteria."""
        if len(text) < min_length:
            return False

        if exclude_patterns:
            for pattern in exclude_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False

        return True

    def scrape_with_filters(self, urls: List[str], min_length: int = 100,
                           exclude_patterns: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Scrape URLs and filter content."""
        for result in self.scrape_urls(urls):
            if 'text' in result and self.filter_content(
                result['text'], min_length, exclude_patterns
            ):
                yield result

    def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Get metadata about a web page without full content."""
        response = self.fetch_page(url)
        if not response:
            return {'url': url, 'error': 'Failed to fetch page'}

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract various metadata
        metadata = {
            'url': url,
            'status_code': response.status_code,
            'content_type': response.headers.get('content-type', ''),
            'content_length': len(response.content),
            'title': '',
            'description': '',
            'keywords': '',
            'language': '',
            'canonical_url': url
        }

        # Title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text().strip()

        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', '').lower()
            property_name = meta.get('property', '').lower()
            content = meta.get('content', '')

            if name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = content
            elif name == 'language' or property_name == 'og:locale':
                metadata['language'] = content

        # Canonical URL
        canonical = soup.find('link', {'rel': 'canonical'})
        if canonical and canonical.get('href'):
            metadata['canonical_url'] = urljoin(url, canonical['href'])

        return metadata