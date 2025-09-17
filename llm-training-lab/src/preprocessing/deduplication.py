import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
import logging
from collections import defaultdict
import re


class ExactDeduplicator:
    """Exact deduplication using hashing."""

    def __init__(self, hash_algorithm: str = 'sha256'):
        self.hash_algorithm = hash_algorithm
        self.seen_hashes: Set[str] = set()
        self.logger = logging.getLogger(__name__)

    def get_hash(self, text: str) -> str:
        """Get hash of text."""
        hasher = hashlib.new(self.hash_algorithm)
        hasher.update(text.encode('utf-8'))
        return hasher.hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate."""
        text_hash = self.get_hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """Remove exact duplicates from list of texts."""
        unique_texts = []
        duplicates_count = 0

        for text in texts:
            if not self.is_duplicate(text):
                unique_texts.append(text)
            else:
                duplicates_count += 1

        self.logger.info(f"Removed {duplicates_count} exact duplicates")
        return unique_texts

    def deduplicate_documents(self, documents: List[Dict[str, Any]],
                            text_field: str = 'text') -> List[Dict[str, Any]]:
        """Remove exact duplicates from list of documents."""
        unique_docs = []
        duplicates_count = 0

        for doc in documents:
            if text_field in doc:
                if not self.is_duplicate(doc[text_field]):
                    unique_docs.append(doc)
                else:
                    duplicates_count += 1
            else:
                unique_docs.append(doc)

        self.logger.info(f"Removed {duplicates_count} duplicate documents")
        return unique_docs

    def get_statistics(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            'unique_hashes': len(self.seen_hashes),
            'hash_algorithm': self.hash_algorithm
        }


class MinHashDeduplicator:
    """MinHash-based near-duplicate detection."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.9, shingle_size: int = 3):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.logger = logging.getLogger(__name__)

        try:
            from datasketch import MinHashLSH, MinHash
            self.MinHashLSH = MinHashLSH
            self.MinHash = MinHash
            self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self.available = True
        except ImportError:
            self.logger.error("datasketch not available for MinHash deduplication")
            self.available = False

    def get_shingles(self, text: str) -> Set[str]:
        """Get character n-grams (shingles) from text."""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.lower().strip())

        # Generate character shingles
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            shingles.add(shingle)

        return shingles

    def get_minhash(self, text: str) -> Optional['MinHash']:
        """Get MinHash signature for text."""
        if not self.available:
            return None

        shingles = self.get_shingles(text)
        if not shingles:
            return None

        minhash = self.MinHash(num_perm=self.num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))

        return minhash

    def add_document(self, doc_id: str, text: str) -> bool:
        """Add document to LSH index. Returns True if added, False if duplicate."""
        if not self.available:
            return True

        minhash = self.get_minhash(text)
        if not minhash:
            return True

        # Check for duplicates
        duplicates = self.lsh.query(minhash)
        if duplicates:
            self.logger.debug(f"Document {doc_id} is similar to {duplicates}")
            return False

        # Add to index
        self.lsh.insert(doc_id, minhash)
        return True

    def deduplicate_texts(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """Deduplicate texts and return unique texts and indices."""
        if not self.available:
            return texts, list(range(len(texts)))

        unique_texts = []
        unique_indices = []
        duplicates_count = 0

        for i, text in enumerate(texts):
            doc_id = f"doc_{i}"
            if self.add_document(doc_id, text):
                unique_texts.append(text)
                unique_indices.append(i)
            else:
                duplicates_count += 1

        self.logger.info(f"Removed {duplicates_count} near-duplicates using MinHash")
        return unique_texts, unique_indices

    def deduplicate_documents(self, documents: List[Dict[str, Any]],
                            text_field: str = 'text') -> List[Dict[str, Any]]:
        """Deduplicate documents based on text field."""
        if not self.available:
            return documents

        unique_docs = []
        duplicates_count = 0

        for i, doc in enumerate(documents):
            if text_field in doc:
                doc_id = f"doc_{i}"
                if self.add_document(doc_id, doc[text_field]):
                    unique_docs.append(doc)
                else:
                    duplicates_count += 1
            else:
                unique_docs.append(doc)

        self.logger.info(f"Removed {duplicates_count} duplicate documents using MinHash")
        return unique_docs

    def get_similarity(self, text1: str, text2: str) -> float:
        """Get Jaccard similarity between two texts."""
        if not self.available:
            return 0.0

        minhash1 = self.get_minhash(text1)
        minhash2 = self.get_minhash(text2)

        if not minhash1 or not minhash2:
            return 0.0

        return minhash1.jaccard(minhash2)


class ContentBasedDeduplicator:
    """Content-based deduplication using various heuristics."""

    def __init__(self, min_length_ratio: float = 0.8, max_common_ratio: float = 0.9):
        self.min_length_ratio = min_length_ratio
        self.max_common_ratio = max_common_ratio
        self.logger = logging.getLogger(__name__)

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def get_word_overlap_ratio(self, text1: str, text2: str) -> float:
        """Get word overlap ratio between two texts."""
        words1 = set(self.normalize_text(text1).split())
        words2 = set(self.normalize_text(text2).split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def are_near_duplicates(self, text1: str, text2: str) -> bool:
        """Check if two texts are near duplicates."""
        # Length ratio check
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return len1 == len2

        length_ratio = min(len1, len2) / max(len1, len2)
        if length_ratio < self.min_length_ratio:
            return False

        # Word overlap check
        overlap_ratio = self.get_word_overlap_ratio(text1, text2)
        return overlap_ratio > self.max_common_ratio

    def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """Remove near-duplicate texts."""
        if not texts:
            return []

        unique_texts = [texts[0]]
        duplicates_count = 0

        for text in texts[1:]:
            is_duplicate = False
            for unique_text in unique_texts:
                if self.are_near_duplicates(text, unique_text):
                    is_duplicate = True
                    duplicates_count += 1
                    break

            if not is_duplicate:
                unique_texts.append(text)

        self.logger.info(f"Removed {duplicates_count} near-duplicates using content analysis")
        return unique_texts


class DeduplicationPipeline:
    """Combined deduplication pipeline."""

    def __init__(self, methods: List[str] = None, **kwargs):
        self.methods = methods or ['exact', 'minhash']
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)

        # Initialize deduplicators
        self.exact_dedup = ExactDeduplicator()

        if 'minhash' in self.methods:
            self.minhash_dedup = MinHashDeduplicator(**kwargs)
        else:
            self.minhash_dedup = None

        if 'content' in self.methods:
            self.content_dedup = ContentBasedDeduplicator(**kwargs)
        else:
            self.content_dedup = None

    def deduplicate(self, documents: List[Dict[str, Any]],
                   text_field: str = 'text') -> List[Dict[str, Any]]:
        """Apply deduplication pipeline."""
        original_count = len(documents)
        current_docs = documents.copy()

        self.logger.info(f"Starting deduplication of {original_count} documents")

        # Apply exact deduplication
        if 'exact' in self.methods:
            current_docs = self.exact_dedup.deduplicate_documents(current_docs, text_field)
            self.logger.info(f"After exact dedup: {len(current_docs)} documents")

        # Apply MinHash deduplication
        if 'minhash' in self.methods and self.minhash_dedup:
            current_docs = self.minhash_dedup.deduplicate_documents(current_docs, text_field)
            self.logger.info(f"After MinHash dedup: {len(current_docs)} documents")

        # Apply content-based deduplication
        if 'content' in self.methods and self.content_dedup:
            texts = [doc[text_field] for doc in current_docs if text_field in doc]
            unique_texts = self.content_dedup.deduplicate_texts(texts)

            # Rebuild documents
            text_to_doc = {doc[text_field]: doc for doc in current_docs if text_field in doc}
            current_docs = [text_to_doc[text] for text in unique_texts if text in text_to_doc]
            self.logger.info(f"After content dedup: {len(current_docs)} documents")

        final_count = len(current_docs)
        removed_count = original_count - final_count
        removal_rate = removed_count / original_count if original_count > 0 else 0

        self.logger.info(f"Deduplication complete: removed {removed_count} documents ({removal_rate:.2%})")

        return current_docs

    def get_deduplication_stats(self, original_docs: List[Dict[str, Any]],
                              deduplicated_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get deduplication statistics."""
        original_count = len(original_docs)
        final_count = len(deduplicated_docs)

        return {
            'original_count': original_count,
            'final_count': final_count,
            'removed_count': original_count - final_count,
            'removal_rate': (original_count - final_count) / original_count if original_count > 0 else 0,
            'methods_used': self.methods
        }