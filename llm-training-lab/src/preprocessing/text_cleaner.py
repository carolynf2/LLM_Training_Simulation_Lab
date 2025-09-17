import re
import unicodedata
import html
from typing import List, Dict, Any, Optional, Callable
import logging
from bs4 import BeautifulSoup
import ftfy


class TextCleaner:
    """Comprehensive text cleaning and preprocessing."""

    def __init__(self,
                 remove_html: bool = True,
                 fix_encoding: bool = True,
                 normalize_unicode: bool = True,
                 normalize_whitespace: bool = True,
                 remove_urls: bool = False,
                 remove_emails: bool = False,
                 min_length: int = 0,
                 max_length: Optional[int] = None,
                 custom_patterns: Optional[Dict[str, str]] = None):

        self.remove_html = remove_html
        self.fix_encoding = fix_encoding
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.min_length = min_length
        self.max_length = max_length
        self.custom_patterns = custom_patterns or {}
        self.logger = logging.getLogger(__name__)

        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.whitespace_pattern = re.compile(r'\s+')
        self.control_chars_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')

    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities."""
        if not self.remove_html:
            return text

        # Decode HTML entities first
        text = html.unescape(text)

        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()

        return text

    def fix_text_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        if not self.fix_encoding:
            return text

        try:
            # Use ftfy to fix encoding issues
            text = ftfy.fix_text(text)
        except Exception as e:
            self.logger.warning(f"Encoding fix failed: {e}")

        return text

    def normalize_unicode_text(self, text: str) -> str:
        """Normalize unicode characters."""
        if not self.normalize_unicode:
            return text

        # Normalize to NFKC form (canonical decomposition, then canonical combining)
        text = unicodedata.normalize('NFKC', text)

        return text

    def remove_control_characters(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        return self.control_chars_pattern.sub('', text)

    def normalize_whitespace_text(self, text: str) -> str:
        """Normalize whitespace."""
        if not self.normalize_whitespace:
            return text

        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def remove_urls_emails(self, text: str) -> str:
        """Remove URLs and email addresses."""
        if self.remove_urls:
            text = self.url_pattern.sub('', text)

        if self.remove_emails:
            text = self.email_pattern.sub('', text)

        return text

    def apply_custom_patterns(self, text: str) -> str:
        """Apply custom regex patterns."""
        for pattern, replacement in self.custom_patterns.items():
            text = re.sub(pattern, replacement, text)

        return text

    def filter_by_length(self, text: str) -> bool:
        """Check if text meets length requirements."""
        length = len(text)

        if length < self.min_length:
            return False

        if self.max_length and length > self.max_length:
            return False

        return True

    def clean_text(self, text: str) -> Optional[str]:
        """Apply all cleaning steps to a single text."""
        if not isinstance(text, str):
            return None

        # Apply cleaning steps in order
        text = self.clean_html(text)
        text = self.fix_text_encoding(text)
        text = self.normalize_unicode_text(text)
        text = self.remove_control_characters(text)
        text = self.remove_urls_emails(text)
        text = self.apply_custom_patterns(text)
        text = self.normalize_whitespace_text(text)

        # Filter by length
        if not self.filter_by_length(text):
            return None

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        cleaned_texts = []

        for text in texts:
            cleaned = self.clean_text(text)
            if cleaned is not None:
                cleaned_texts.append(cleaned)

        return cleaned_texts

    def process_document(self, document: Dict[str, Any],
                        text_fields: List[str] = ['text', 'content']) -> Dict[str, Any]:
        """Process a document with multiple text fields."""
        processed_doc = document.copy()

        for field in text_fields:
            if field in document and isinstance(document[field], str):
                cleaned = self.clean_text(document[field])
                if cleaned is not None:
                    processed_doc[field] = cleaned
                else:
                    # Remove document if text doesn't meet criteria
                    return None

        return processed_doc

    def get_cleaning_stats(self, original_texts: List[str], cleaned_texts: List[str]) -> Dict[str, Any]:
        """Get statistics about the cleaning process."""
        original_count = len(original_texts)
        cleaned_count = len(cleaned_texts)

        if original_count == 0:
            return {'error': 'No input texts provided'}

        original_lengths = [len(text) for text in original_texts]
        cleaned_lengths = [len(text) for text in cleaned_texts]

        stats = {
            'original_count': original_count,
            'cleaned_count': cleaned_count,
            'filtered_out': original_count - cleaned_count,
            'filter_rate': (original_count - cleaned_count) / original_count,
            'original_total_chars': sum(original_lengths),
            'cleaned_total_chars': sum(cleaned_lengths),
            'char_reduction_rate': 1 - (sum(cleaned_lengths) / sum(original_lengths)) if sum(original_lengths) > 0 else 0,
            'avg_original_length': sum(original_lengths) / len(original_lengths) if original_lengths else 0,
            'avg_cleaned_length': sum(cleaned_lengths) / len(cleaned_lengths) if cleaned_lengths else 0
        }

        return stats


class LanguageDetector:
    """Language detection and filtering."""

    def __init__(self, target_languages: Optional[List[str]] = None):
        self.target_languages = target_languages or ['en']
        self.logger = logging.getLogger(__name__)

        try:
            from langdetect import detect, LangDetectError
            self.detect = detect
            self.LangDetectError = LangDetectError
            self.available = True
        except ImportError:
            self.logger.warning("langdetect not available, language filtering disabled")
            self.available = False

    def detect_language(self, text: str) -> Optional[str]:
        """Detect the language of a text."""
        if not self.available or len(text.strip()) < 50:
            return None

        try:
            return self.detect(text)
        except self.LangDetectError:
            return None

    def filter_by_language(self, text: str) -> bool:
        """Check if text is in target language(s)."""
        if not self.available:
            return True

        detected_lang = self.detect_language(text)
        return detected_lang in self.target_languages if detected_lang else False

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of texts with language detection."""
        results = []

        for text in texts:
            lang = self.detect_language(text)
            keep = self.filter_by_language(text)

            results.append({
                'text': text,
                'detected_language': lang,
                'keep': keep
            })

        return results