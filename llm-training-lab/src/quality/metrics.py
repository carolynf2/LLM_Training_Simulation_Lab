import re
import math
from typing import Dict, List, Any, Optional
import logging
from collections import Counter
import statistics


class TextComplexityMetrics:
    """Calculate various text complexity metrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score."""
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        syllables = self._count_syllables(text)

        if sentences == 0 or words == 0:
            return 0.0

        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score))

    def flesch_kincaid_grade(self, text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level."""
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        syllables = self._count_syllables(text)

        if sentences == 0 or words == 0:
            return 0.0

        grade = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
        return max(0.0, grade)

    def smog_index(self, text: str) -> float:
        """Calculate SMOG (Simple Measure of Gobbledygook) index."""
        sentences = self._count_sentences(text)
        polysyllables = self._count_polysyllables(text)

        if sentences < 3:
            return 0.0

        smog = 1.0430 * math.sqrt(polysyllables * (30 / sentences)) + 3.1291
        return max(0.0, smog)

    def automated_readability_index(self, text: str) -> float:
        """Calculate Automated Readability Index (ARI)."""
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        characters = len(re.sub(r'\s+', '', text))

        if sentences == 0 or words == 0:
            return 0.0

        ari = (4.71 * (characters / words)) + (0.5 * (words / sentences)) - 21.43
        return max(0.0, ari)

    def gunning_fog_index(self, text: str) -> float:
        """Calculate Gunning Fog Index."""
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        complex_words = self._count_complex_words(text)

        if sentences == 0 or words == 0:
            return 0.0

        fog = 0.4 * ((words / sentences) + (100 * (complex_words / words)))
        return max(0.0, fog)

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    def _count_syllables(self, text: str) -> int:
        """Estimate syllable count in text."""
        words = re.findall(r'\b\w+\b', text.lower())
        total_syllables = 0

        for word in words:
            syllables = self._syllables_in_word(word)
            total_syllables += syllables

        return total_syllables

    def _syllables_in_word(self, word: str) -> int:
        """Estimate syllables in a single word."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _count_polysyllables(self, text: str) -> int:
        """Count words with 3+ syllables."""
        words = re.findall(r'\b\w+\b', text.lower())
        return sum(1 for word in words if self._syllables_in_word(word) >= 3)

    def _count_complex_words(self, text: str) -> int:
        """Count complex words (3+ syllables, not proper nouns or common suffixes)."""
        words = re.findall(r'\b\w+\b', text)
        complex_count = 0

        for word in words:
            if len(word) <= 2:
                continue

            # Skip if starts with capital (proper noun)
            if word[0].isupper():
                continue

            # Skip common suffixes
            if word.lower().endswith(('es', 'ed', 'ing')):
                continue

            if self._syllables_in_word(word.lower()) >= 3:
                complex_count += 1

        return complex_count


class DiversityMetrics:
    """Calculate text diversity metrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def vocabulary_richness(self, text: str) -> float:
        """Calculate type-token ratio (vocabulary richness)."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)

        return unique_words / total_words

    def lexical_diversity(self, text: str, k: int = 10000) -> float:
        """Calculate lexical diversity using MTLD (Measure of Textual Lexical Diversity)."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 50:
            return 0.0

        def ttr_at_point(word_list: List[str], start: int, threshold: float = 0.72) -> int:
            types = set()
            for i, word in enumerate(word_list[start:], start):
                types.add(word)
                ttr = len(types) / (i - start + 1)
                if ttr <= threshold:
                    return i - start + 1
            return len(word_list) - start

        forward_mtld = []
        i = 0
        while i < len(words):
            segment_length = ttr_at_point(words, i)
            if segment_length >= 10:
                forward_mtld.append(segment_length)
            i += max(1, segment_length)

        if not forward_mtld:
            return 0.0

        return statistics.mean(forward_mtld)

    def ngram_diversity(self, text: str, n: int = 2) -> float:
        """Calculate n-gram diversity."""
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < n:
            return 0.0

        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        return unique_ngrams / total_ngrams

    def repetitiveness_score(self, text: str) -> float:
        """Calculate repetitiveness score (lower is better)."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 1.0

        word_counts = Counter(words)
        total_words = len(words)

        # Calculate entropy
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            entropy -= probability * math.log2(probability)

        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(word_counts))
        if max_entropy == 0:
            return 1.0

        normalized_entropy = entropy / max_entropy
        repetitiveness = 1.0 - normalized_entropy

        return max(0.0, min(1.0, repetitiveness))


class LanguageQualityMetrics:
    """Calculate language quality metrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def spelling_error_ratio(self, text: str) -> float:
        """Estimate spelling error ratio using simple heuristics."""
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0

        # Simple heuristics for potential spelling errors
        error_count = 0
        for word in words:
            # Check for obvious errors
            if self._likely_spelling_error(word):
                error_count += 1

        return error_count / len(words)

    def _likely_spelling_error(self, word: str) -> bool:
        """Simple heuristics to detect potential spelling errors."""
        word = word.lower()

        # Very short or very long words
        if len(word) < 2 or len(word) > 20:
            return True

        # Too many consecutive consonants or vowels
        if re.search(r'[bcdfghjklmnpqrstvwxz]{4,}', word) or re.search(r'[aeiou]{4,}', word):
            return True

        # Numbers mixed with letters in unexpected ways
        if re.search(r'\d.*[a-z].*\d', word):
            return True

        return False

    def punctuation_quality(self, text: str) -> float:
        """Assess punctuation quality."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        quality_score = 0.0
        total_checks = 0

        for sentence in sentences:
            # Check if sentence starts with capital letter
            if sentence and sentence[0].isupper():
                quality_score += 1
            total_checks += 1

            # Check for excessive punctuation
            punct_ratio = len(re.findall(r'[^\w\s]', sentence)) / max(1, len(sentence))
            if punct_ratio < 0.1:  # Not too much punctuation
                quality_score += 1
            total_checks += 1

        return quality_score / total_checks if total_checks > 0 else 0.0

    def coherence_score(self, text: str) -> float:
        """Simple coherence score based on sentence structure."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5

        coherence_score = 0.0
        comparisons = 0

        for i in range(len(sentences) - 1):
            sent1_words = set(re.findall(r'\b\w+\b', sentences[i].lower()))
            sent2_words = set(re.findall(r'\b\w+\b', sentences[i + 1].lower()))

            if sent1_words and sent2_words:
                overlap = len(sent1_words.intersection(sent2_words))
                union = len(sent1_words.union(sent2_words))
                similarity = overlap / union if union > 0 else 0
                coherence_score += similarity
                comparisons += 1

        return coherence_score / comparisons if comparisons > 0 else 0.0


class QualityMetricsCalculator:
    """Main class for calculating all quality metrics."""

    def __init__(self):
        self.complexity_metrics = TextComplexityMetrics()
        self.diversity_metrics = DiversityMetrics()
        self.language_metrics = LanguageQualityMetrics()
        self.logger = logging.getLogger(__name__)

    def calculate_all_metrics(self, text: str) -> Dict[str, float]:
        """Calculate all quality metrics for a text."""
        if not text or not text.strip():
            return self._empty_metrics()

        metrics = {}

        try:
            # Complexity metrics
            metrics['flesch_reading_ease'] = self.complexity_metrics.flesch_reading_ease(text)
            metrics['flesch_kincaid_grade'] = self.complexity_metrics.flesch_kincaid_grade(text)
            metrics['smog_index'] = self.complexity_metrics.smog_index(text)
            metrics['ari'] = self.complexity_metrics.automated_readability_index(text)
            metrics['gunning_fog'] = self.complexity_metrics.gunning_fog_index(text)

            # Diversity metrics
            metrics['vocabulary_richness'] = self.diversity_metrics.vocabulary_richness(text)
            metrics['lexical_diversity'] = self.diversity_metrics.lexical_diversity(text)
            metrics['bigram_diversity'] = self.diversity_metrics.ngram_diversity(text, 2)
            metrics['trigram_diversity'] = self.diversity_metrics.ngram_diversity(text, 3)
            metrics['repetitiveness'] = self.diversity_metrics.repetitiveness_score(text)

            # Language quality metrics
            metrics['spelling_error_ratio'] = self.language_metrics.spelling_error_ratio(text)
            metrics['punctuation_quality'] = self.language_metrics.punctuation_quality(text)
            metrics['coherence_score'] = self.language_metrics.coherence_score(text)

            # Basic metrics
            metrics['word_count'] = len(re.findall(r'\b\w+\b', text))
            metrics['char_count'] = len(text)
            metrics['sentence_count'] = len(re.split(r'[.!?]+', text))
            metrics['avg_word_length'] = self._average_word_length(text)
            metrics['avg_sentence_length'] = self._average_sentence_length(text)

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return self._empty_metrics()

        return metrics

    def _average_word_length(self, text: str) -> float:
        """Calculate average word length."""
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)

    def _average_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        total_words = 0
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            total_words += len(words)

        return total_words / len(sentences)

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'smog_index': 0.0,
            'ari': 0.0,
            'gunning_fog': 0.0,
            'vocabulary_richness': 0.0,
            'lexical_diversity': 0.0,
            'bigram_diversity': 0.0,
            'trigram_diversity': 0.0,
            'repetitiveness': 0.0,
            'spelling_error_ratio': 0.0,
            'punctuation_quality': 0.0,
            'coherence_score': 0.0,
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0
        }

    def get_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics."""
        if not metrics:
            return 0.0

        # Weights for different metric categories
        weights = {
            'readability': 0.3,
            'diversity': 0.3,
            'language_quality': 0.4
        }

        # Normalize and weight metrics
        readability_score = self._normalize_readability_metrics(metrics)
        diversity_score = self._normalize_diversity_metrics(metrics)
        language_score = self._normalize_language_metrics(metrics)

        overall_score = (
            weights['readability'] * readability_score +
            weights['diversity'] * diversity_score +
            weights['language_quality'] * language_score
        )

        return max(0.0, min(1.0, overall_score))

    def _normalize_readability_metrics(self, metrics: Dict[str, float]) -> float:
        """Normalize readability metrics to 0-1 scale."""
        # Flesch Reading Ease: higher is better, scale 0-100
        flesch_score = metrics.get('flesch_reading_ease', 0) / 100.0

        # Grade levels: target around 8-12, normalize accordingly
        fk_grade = metrics.get('flesch_kincaid_grade', 0)
        fk_score = max(0, 1 - abs(fk_grade - 10) / 10)

        return (flesch_score + fk_score) / 2

    def _normalize_diversity_metrics(self, metrics: Dict[str, float]) -> float:
        """Normalize diversity metrics to 0-1 scale."""
        vocab_richness = metrics.get('vocabulary_richness', 0)
        bigram_diversity = metrics.get('bigram_diversity', 0)
        repetitiveness = 1 - metrics.get('repetitiveness', 1)  # Invert repetitiveness

        return (vocab_richness + bigram_diversity + repetitiveness) / 3

    def _normalize_language_metrics(self, metrics: Dict[str, float]) -> float:
        """Normalize language quality metrics to 0-1 scale."""
        spelling_quality = 1 - metrics.get('spelling_error_ratio', 1)  # Invert error ratio
        punctuation_quality = metrics.get('punctuation_quality', 0)
        coherence_score = metrics.get('coherence_score', 0)

        return (spelling_quality + punctuation_quality + coherence_score) / 3