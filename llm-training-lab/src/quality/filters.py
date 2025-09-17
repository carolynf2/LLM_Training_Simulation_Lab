from typing import List, Dict, Any, Optional, Callable, Tuple
import re
import logging
from abc import ABC, abstractmethod


class BaseFilter(ABC):
    """Base class for all filters."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Return True if text should be kept, False if filtered out."""
        pass

    def filter_documents(self, documents: List[Dict[str, Any]],
                        text_field: str = 'text') -> List[Dict[str, Any]]:
        """Filter list of documents."""
        filtered = []
        for doc in documents:
            if text_field in doc:
                if self.filter(doc[text_field], doc):
                    filtered.append(doc)
            else:
                # Keep documents without text field
                filtered.append(doc)

        return filtered

    def get_filter_stats(self, documents: List[Dict[str, Any]],
                        text_field: str = 'text') -> Dict[str, Any]:
        """Get filtering statistics."""
        original_count = len(documents)
        filtered_docs = self.filter_documents(documents, text_field)
        filtered_count = len(filtered_docs)

        return {
            'filter_name': self.name,
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': original_count - filtered_count,
            'removal_rate': (original_count - filtered_count) / original_count if original_count > 0 else 0
        }


class LengthFilter(BaseFilter):
    """Filter based on text length."""

    def __init__(self, min_length: int = 0, max_length: Optional[int] = None,
                 unit: str = 'characters'):
        super().__init__(f"LengthFilter({unit})")
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit.lower()

    def _get_length(self, text: str) -> int:
        """Get length in specified unit."""
        if self.unit == 'characters':
            return len(text)
        elif self.unit == 'words':
            return len(re.findall(r'\b\w+\b', text))
        elif self.unit == 'sentences':
            return len(re.split(r'[.!?]+', text.strip()))
        else:
            raise ValueError(f"Unknown unit: {self.unit}")

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        length = self._get_length(text)

        if length < self.min_length:
            return False

        if self.max_length is not None and length > self.max_length:
            return False

        return True


class LanguageFilter(BaseFilter):
    """Filter based on detected language."""

    def __init__(self, target_languages: List[str], confidence_threshold: float = 0.7):
        super().__init__("LanguageFilter")
        self.target_languages = [lang.lower() for lang in target_languages]
        self.confidence_threshold = confidence_threshold

        try:
            from langdetect import detect, detect_langs, LangDetectError
            self.detect = detect
            self.detect_langs = detect_langs
            self.LangDetectError = LangDetectError
            self.available = True
        except ImportError:
            self.logger.warning("langdetect not available, language filtering disabled")
            self.available = False

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if not self.available:
            return True

        if len(text.strip()) < 50:  # Skip very short texts
            return True

        try:
            # Get detailed language detection
            lang_probs = self.detect_langs(text)
            for lang_prob in lang_probs:
                if (lang_prob.lang.lower() in self.target_languages and
                    lang_prob.prob >= self.confidence_threshold):
                    return True

            return False

        except self.LangDetectError:
            # If detection fails, be conservative and keep the text
            return True


class QualityFilter(BaseFilter):
    """Filter based on quality metrics."""

    def __init__(self, min_quality_score: float = 0.5,
                 quality_calculator=None):
        super().__init__("QualityFilter")
        self.min_quality_score = min_quality_score

        if quality_calculator is None:
            from .metrics import QualityMetricsCalculator
            self.quality_calculator = QualityMetricsCalculator()
        else:
            self.quality_calculator = quality_calculator

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        metrics = self.quality_calculator.calculate_all_metrics(text)
        quality_score = self.quality_calculator.get_quality_score(metrics)

        return quality_score >= self.min_quality_score


class ToxicityFilter(BaseFilter):
    """Filter based on toxicity detection."""

    def __init__(self, max_toxicity_score: float = 0.3):
        super().__init__("ToxicityFilter")
        self.max_toxicity_score = max_toxicity_score

        # Simple word-based toxicity detection
        self.toxic_words = self._load_toxic_words()

    def _load_toxic_words(self) -> List[str]:
        """Load list of toxic words (simplified implementation)."""
        # In a real implementation, this would load from a comprehensive list
        return [
            'hate', 'racist', 'sexist', 'offensive', 'discrimination',
            'violence', 'threat', 'harassment', 'bullying', 'abuse'
        ]

    def _calculate_toxicity_score(self, text: str) -> float:
        """Calculate simple toxicity score based on word matching."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        toxic_count = sum(1 for word in words if word in self.toxic_words)
        return toxic_count / len(words)

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        toxicity_score = self._calculate_toxicity_score(text)
        return toxicity_score <= self.max_toxicity_score


class ContentTypeFilter(BaseFilter):
    """Filter based on content type classification."""

    def __init__(self, allowed_types: List[str], min_confidence: float = 0.6):
        super().__init__("ContentTypeFilter")
        self.allowed_types = [t.lower() for t in allowed_types]
        self.min_confidence = min_confidence

    def _classify_content_type(self, text: str) -> Tuple[str, float]:
        """Simple content type classification."""
        text_lower = text.lower()

        # Simple heuristics for content type detection
        if re.search(r'\b(recipe|ingredients|cooking|bake|cook)\b', text_lower):
            return 'recipe', 0.8
        elif re.search(r'\b(news|reported|according to|sources)\b', text_lower):
            return 'news', 0.7
        elif re.search(r'\b(abstract|introduction|methodology|conclusion)\b', text_lower):
            return 'academic', 0.8
        elif re.search(r'\b(review|rating|stars|recommend)\b', text_lower):
            return 'review', 0.7
        elif re.search(r'\b(tutorial|how to|step|guide)\b', text_lower):
            return 'tutorial', 0.8
        else:
            return 'general', 0.5

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        content_type, confidence = self._classify_content_type(text)

        return (content_type in self.allowed_types and
                confidence >= self.min_confidence)


class PerplexityFilter(BaseFilter):
    """Filter based on text perplexity (complexity)."""

    def __init__(self, max_perplexity: float = 1000.0):
        super().__init__("PerplexityFilter")
        self.max_perplexity = max_perplexity

    def _calculate_perplexity(self, text: str) -> float:
        """Calculate simple perplexity estimate."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return float('inf')

        # Simple unigram perplexity based on word frequency
        from collections import Counter
        word_counts = Counter(words)
        total_words = len(words)

        log_prob_sum = 0.0
        for word in words:
            probability = word_counts[word] / total_words
            log_prob_sum += -math.log2(probability)

        return 2 ** (log_prob_sum / total_words)

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        import math
        perplexity = self._calculate_perplexity(text)
        return perplexity <= self.max_perplexity


class DuplicateFilter(BaseFilter):
    """Filter out duplicate content."""

    def __init__(self, similarity_threshold: float = 0.9):
        super().__init__("DuplicateFilter")
        self.similarity_threshold = similarity_threshold
        self.seen_texts = []

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        # Check against previously seen texts
        for seen_text in self.seen_texts:
            similarity = self._calculate_similarity(text, seen_text)
            if similarity >= self.similarity_threshold:
                return False

        # Add to seen texts if unique
        self.seen_texts.append(text)
        return True

    def reset(self):
        """Reset the filter's memory."""
        self.seen_texts = []


class CompositeFilter:
    """Combine multiple filters with AND/OR logic."""

    def __init__(self, filters: List[BaseFilter], logic: str = 'and'):
        self.filters = filters
        self.logic = logic.lower()
        self.logger = logging.getLogger(__name__)

        if self.logic not in ['and', 'or']:
            raise ValueError("Logic must be 'and' or 'or'")

    def filter(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Apply composite filtering."""
        if not self.filters:
            return True

        results = [f.filter(text, metadata) for f in self.filters]

        if self.logic == 'and':
            return all(results)
        else:  # or
            return any(results)

    def filter_documents(self, documents: List[Dict[str, Any]],
                        text_field: str = 'text') -> List[Dict[str, Any]]:
        """Filter documents using composite logic."""
        filtered = []
        for doc in documents:
            if text_field in doc:
                if self.filter(doc[text_field], doc):
                    filtered.append(doc)
            else:
                filtered.append(doc)

        return filtered

    def get_detailed_stats(self, documents: List[Dict[str, Any]],
                          text_field: str = 'text') -> Dict[str, Any]:
        """Get detailed filtering statistics for each filter."""
        stats = {
            'composite_logic': self.logic,
            'total_filters': len(self.filters),
            'filter_stats': []
        }

        # Get stats for each individual filter
        for filter_obj in self.filters:
            filter_stats = filter_obj.get_filter_stats(documents, text_field)
            stats['filter_stats'].append(filter_stats)

        # Get overall composite stats
        original_count = len(documents)
        filtered_docs = self.filter_documents(documents, text_field)
        final_count = len(filtered_docs)

        stats['overall'] = {
            'original_count': original_count,
            'final_count': final_count,
            'total_removed': original_count - final_count,
            'total_removal_rate': (original_count - final_count) / original_count if original_count > 0 else 0
        }

        return stats


class FilterPipeline:
    """Pipeline for applying multiple filters in sequence."""

    def __init__(self):
        self.filters = []
        self.logger = logging.getLogger(__name__)

    def add_filter(self, filter_obj: BaseFilter):
        """Add a filter to the pipeline."""
        self.filters.append(filter_obj)

    def process(self, documents: List[Dict[str, Any]],
               text_field: str = 'text') -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process documents through the filter pipeline."""
        current_docs = documents.copy()
        stats = {
            'original_count': len(documents),
            'filter_stats': [],
            'final_count': 0,
            'total_removed': 0,
            'total_removal_rate': 0.0
        }

        self.logger.info(f"Starting filter pipeline with {len(documents)} documents")

        for i, filter_obj in enumerate(self.filters):
            before_count = len(current_docs)
            current_docs = filter_obj.filter_documents(current_docs, text_field)
            after_count = len(current_docs)

            removed = before_count - after_count
            removal_rate = removed / before_count if before_count > 0 else 0

            filter_stats = {
                'step': i + 1,
                'filter_name': filter_obj.name,
                'before_count': before_count,
                'after_count': after_count,
                'removed_count': removed,
                'removal_rate': removal_rate
            }

            stats['filter_stats'].append(filter_stats)

            self.logger.info(f"Filter {i+1} ({filter_obj.name}): {before_count} -> {after_count} "
                           f"(removed {removed}, {removal_rate:.2%})")

        stats['final_count'] = len(current_docs)
        stats['total_removed'] = len(documents) - len(current_docs)
        stats['total_removal_rate'] = stats['total_removed'] / len(documents) if len(documents) > 0 else 0

        self.logger.info(f"Filter pipeline complete: {len(documents)} -> {len(current_docs)} "
                        f"(total removed: {stats['total_removed']}, {stats['total_removal_rate']:.2%})")

        return current_docs, stats