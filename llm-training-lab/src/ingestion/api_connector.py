import requests
import json
import time
from typing import Iterator, Dict, Any, Optional, List
import logging
from urllib.parse import urlencode
from tqdm import tqdm


class APIConnector:
    """Connector for various APIs like Wikipedia, Common Crawl, arXiv."""

    def __init__(self, rate_limit: float = 1.0, timeout: int = 30):
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make API request with error handling."""
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None

    def wikipedia_search(self, query: str, limit: int = 10) -> Iterator[Dict[str, Any]]:
        """Search Wikipedia articles."""
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        list_url = "https://en.wikipedia.org/w/api.php"

        # First, search for articles
        search_params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit
        }

        response = self._make_request(list_url, search_params)
        if not response:
            return

        data = response.json()
        if 'query' not in data or 'search' not in data['query']:
            return

        # Get full content for each article
        for article in tqdm(data['query']['search'], desc="Fetching Wikipedia articles"):
            title = article['title']
            page_url = f"{search_url}{title.replace(' ', '_')}"

            page_response = self._make_request(page_url)
            if page_response:
                page_data = page_response.json()
                yield {
                    'title': page_data.get('title', ''),
                    'extract': page_data.get('extract', ''),
                    'url': page_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'source': 'wikipedia',
                    'timestamp': page_data.get('timestamp', ''),
                    'content_length': len(page_data.get('extract', ''))
                }

            time.sleep(self.rate_limit)

    def arxiv_search(self, query: str, max_results: int = 100,
                    category: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Search arXiv papers."""
        base_url = "http://export.arxiv.org/api/query"

        search_query = f"all:{query}"
        if category:
            search_query += f" AND cat:{category}"

        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        response = self._make_request(base_url, params)
        if not response:
            return

        # Parse XML response
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)

        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        for entry in root.findall('atom:entry', namespaces):
            title = entry.find('atom:title', namespaces)
            summary = entry.find('atom:summary', namespaces)
            published = entry.find('atom:published', namespaces)
            authors = entry.findall('atom:author', namespaces)

            author_names = []
            for author in authors:
                name = author.find('atom:name', namespaces)
                if name is not None:
                    author_names.append(name.text)

            yield {
                'title': title.text.strip() if title is not None else '',
                'abstract': summary.text.strip() if summary is not None else '',
                'authors': author_names,
                'published': published.text if published is not None else '',
                'source': 'arxiv',
                'url': entry.find('atom:id', namespaces).text if entry.find('atom:id', namespaces) is not None else ''
            }

    def common_crawl_search(self, domain: str, limit: int = 1000) -> Iterator[Dict[str, Any]]:
        """Search Common Crawl index for domain."""
        # This is a simplified implementation
        # Full implementation would require accessing CC's index
        index_url = "https://index.commoncrawl.org/CC-MAIN-2023-50-index"

        params = {
            'url': f"{domain}/*",
            'limit': limit,
            'output': 'json'
        }

        response = self._make_request(index_url, params)
        if not response:
            return

        # Process CDXJ format
        for line in response.text.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    yield {
                        'url': data.get('url', ''),
                        'timestamp': data.get('timestamp', ''),
                        'mime_type': data.get('mime', ''),
                        'status_code': data.get('status', ''),
                        'source': 'common_crawl',
                        'filename': data.get('filename', ''),
                        'offset': data.get('offset', ''),
                        'length': data.get('length', '')
                    }
                except json.JSONDecodeError:
                    continue

    def hackernews_search(self, query: str, tags: str = "story",
                         num_hits: int = 100) -> Iterator[Dict[str, Any]]:
        """Search Hacker News using Algolia API."""
        base_url = "https://hn.algolia.com/api/v1/search"

        params = {
            'query': query,
            'tags': tags,
            'hitsPerPage': min(num_hits, 1000)
        }

        response = self._make_request(base_url, params)
        if not response:
            return

        data = response.json()
        for hit in data.get('hits', []):
            yield {
                'title': hit.get('title', ''),
                'text': hit.get('story_text', '') or hit.get('comment_text', ''),
                'url': hit.get('url', ''),
                'author': hit.get('author', ''),
                'points': hit.get('points', 0),
                'num_comments': hit.get('num_comments', 0),
                'created_at': hit.get('created_at', ''),
                'source': 'hackernews',
                'object_id': hit.get('objectID', '')
            }

    def reddit_search(self, subreddit: str, query: Optional[str] = None,
                     limit: int = 100) -> Iterator[Dict[str, Any]]:
        """Search Reddit posts (requires Reddit API key for full functionality)."""
        # This is a simplified implementation using public endpoints
        if query:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {'q': query, 'limit': limit, 'sort': 'relevance'}
        else:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            params = {'limit': limit}

        headers = {'User-Agent': 'LLM-Training-Lab/1.0'}

        try:
            response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                yield {
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                    'author': post_data.get('author', ''),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_utc': post_data.get('created_utc', 0),
                    'subreddit': post_data.get('subreddit', ''),
                    'source': 'reddit'
                }
        except requests.RequestException as e:
            self.logger.error(f"Reddit API error: {e}")
            return

    def fetch_rss_feed(self, feed_url: str, limit: int = 50) -> Iterator[Dict[str, Any]]:
        """Fetch and parse RSS/Atom feed."""
        response = self._make_request(feed_url)
        if not response:
            return

        # Parse XML feed
        from xml.etree import ElementTree as ET
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            return

        # Handle both RSS and Atom feeds
        items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')

        for i, item in enumerate(items[:limit]):
            # RSS format
            title_elem = item.find('title')
            description_elem = item.find('description') or item.find('.//{http://www.w3.org/2005/Atom}content')
            link_elem = item.find('link') or item.find('.//{http://www.w3.org/2005/Atom}link')
            pubdate_elem = item.find('pubDate') or item.find('.//{http://www.w3.org/2005/Atom}published')

            yield {
                'title': title_elem.text if title_elem is not None else '',
                'description': description_elem.text if description_elem is not None else '',
                'link': link_elem.text if link_elem is not None else (link_elem.get('href') if link_elem is not None else ''),
                'pub_date': pubdate_elem.text if pubdate_elem is not None else '',
                'source': 'rss_feed',
                'feed_url': feed_url
            }