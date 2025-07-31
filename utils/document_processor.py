import logging
import re
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles web scraping, text extraction, and chunking of documents.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}.")

    def scrape_and_extract_content(self, url: str, user_agent: str = "Mozilla/5.0", depth: int = 0, max_depth: int = 1) -> str:
        if depth > max_depth:
            return ""

        full_text_content = ""
        visited_urls = set()
        
        def _get_page_text(target_url: str) -> Optional[str]:
            if target_url in visited_urls:
                return None
            visited_urls.add(target_url)

            try:
                headers = {'User-Agent': user_agent}
                logger.info(f"Scraping URL: {target_url} (Depth: {depth})")
                response = requests.get(target_url, headers=headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                for selector in ['script', 'style', 'header', 'footer', 'nav', 'aside', '.sidebar', '.ad-block', '.menu', '#comments', '[role="navigation"]']:
                    for element in soup.find_all(selector):
                        element.decompose()

                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'span', 'div'])
                text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in text_elements])

                text = re.sub(r'\s+', ' ', text).strip()
                logger.debug(f"Extracted {len(text)} characters from {target_url}.")
                return text
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to retrieve or parse {target_url}: {e}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during _get_page_text for {target_url}: {e}", exc_info=True)
                return None

        main_page_content = _get_page_text(url)
        if main_page_content:
            full_text_content += main_page_content + "\n\n"

        if depth < max_depth:
            try:
                headers = {'User-Agent': user_agent}
                response_for_links = requests.get(url, headers=headers, timeout=15)
                response_for_links.raise_for_status()
                soup_for_links = BeautifulSoup(response_for_links.text, 'html.parser')
                
                base_url_parsed = urlparse(url)
                base_domain = base_url_parsed.netloc
                
                for a_tag in soup_for_links.find_all('a', href=True):
                    href = a_tag['href']
                    linked_url = urljoin(url, href)
                    
                    parsed_linked_url = urlparse(linked_url)
                    
                    if (parsed_linked_url.netloc == base_domain or parsed_linked_url.netloc.endswith(f".{base_domain}")) and \
                       parsed_linked_url.fragment == '' and \
                       linked_url not in visited_urls and \
                       self._is_relevant_path(linked_url, base_url_parsed.path):
                        
                        linked_content = self.scrape_and_extract_content(linked_url, user_agent, depth + 1, max_depth)
                        if linked_content:
                            full_text_content += linked_content + "\n\n"
            except Exception as e:
                logger.error(f"Error during link extraction: {e}", exc_info=True)

        return full_text_content.strip()

    def _is_relevant_path(self, full_url: str, base_path_filter: str) -> bool:
        parsed_url = urlparse(full_url)
        path = parsed_url.path
        
        excluded_extensions = [
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.css', '.js', '.xml', '.txt',
            '.json'
        ]
        if any(path.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        excluded_keywords = [
            '/tag/', '/category/', '/author/', '/feed/', '/rss/', '/sitemap', 
            '/wp-admin/', '/login/', '/register/', '/cart/', '/checkout/', 
            '/account/', '/search/', '/cdn-cgi/', '/_static/', '/assets/',
            '/api/', '/json/', '/xmlrpc.php',
            'sitemaps.xml', 'robots.txt'
        ]
        if any(keyword in path.lower() for keyword in excluded_keywords):
            return False
        
        if len(parsed_url.query) > 100:
            return False

        if base_path_filter and not path.startswith(base_path_filter):
            return False

        return True

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        if not text:
            return []

        chunks = []
        words = text.split()

        if len(words) <= self.chunk_size:
            chunks.append({'text': text, 'metadata': metadata if metadata else {}})
            return chunks

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({'text': chunk_text, 'metadata': metadata if metadata else {}})
        
        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []
        for doc in documents:
            text = doc.get('text')
            metadata = doc.get('metadata')
            if text:
                all_chunks.extend(self.chunk_text(text, metadata))
        return all_chunks