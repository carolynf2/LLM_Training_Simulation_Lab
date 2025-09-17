import os
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import json


class BaseTokenizer:
    """Base tokenizer interface."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.logger = logging.getLogger(__name__)

    def train(self, texts: List[str], output_path: str):
        """Train tokenizer on texts."""
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        raise NotImplementedError

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens."""
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def save(self, path: str):
        """Save tokenizer."""
        raise NotImplementedError

    def load(self, path: str):
        """Load tokenizer."""
        raise NotImplementedError


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece tokenizer wrapper."""

    def __init__(self, vocab_size: int = 32000, model_type: str = 'bpe',
                 character_coverage: float = 0.9995):
        super().__init__(vocab_size)
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.model = None

        try:
            import sentencepiece as spm
            self.spm = spm
            self.available = True
        except ImportError:
            self.logger.error("sentencepiece not available")
            self.available = False

    def train(self, texts: List[str], output_path: str):
        """Train SentencePiece model."""
        if not self.available:
            raise RuntimeError("sentencepiece not available")

        # Create temporary training file
        temp_file = f"{output_path}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')

        # Train model
        self.spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=output_path,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=self.character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )

        # Load trained model
        self.model = self.spm.SentencePieceProcessor()
        self.model.load(f"{output_path}.model")

        # Clean up temp file
        os.remove(temp_file)

        self.logger.info(f"Trained SentencePiece model: {output_path}")

    def load(self, model_path: str):
        """Load trained model."""
        if not self.available:
            raise RuntimeError("sentencepiece not available")

        self.model = self.spm.SentencePieceProcessor()
        self.model.load(model_path)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.decode(token_ids)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to tokens."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.encode(text, out_type=str)


class SimpleTokenizer(BaseTokenizer):
    """Simple whitespace + punctuation tokenizer."""

    def __init__(self, vocab_size: int = 32000):
        super().__init__(vocab_size)
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts."""
        import re
        from collections import Counter

        # Simple tokenization: split on whitespace and punctuation
        token_pattern = re.compile(r'\w+|[^\w\s]')
        counter = Counter()

        for text in texts:
            tokens = token_pattern.findall(text.lower())
            counter.update(tokens)

        # Build vocab with most frequent tokens
        vocab_tokens = counter.most_common(self.vocab_size - len(self.special_tokens))

        # Start with special tokens
        self.vocab = self.special_tokens.copy()
        self.reverse_vocab = {v: k for k, v in self.special_tokens.items()}

        # Add frequent tokens
        for token, _ in vocab_tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token

    def train(self, texts: List[str], output_path: str):
        """Train simple tokenizer."""
        self.build_vocab(texts)
        self.save(output_path)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        import re
        token_pattern = re.compile(r'\w+|[^\w\s]')
        return token_pattern.findall(text.lower())

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.reverse_vocab.get(id, '<unk>') for id in token_ids]
        return ' '.join(tokens)

    def save(self, path: str):
        """Save tokenizer."""
        save_data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(f"{path}.json", 'w') as f:
            json.dump(save_data, f, indent=2)

    def load(self, path: str):
        """Load tokenizer."""
        with open(path, 'r') as f:
            save_data = json.load(f)

        self.vocab = save_data['vocab']
        self.vocab_size = save_data['vocab_size']
        self.special_tokens = save_data['special_tokens']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}


class TokenizerFactory:
    """Factory for creating tokenizers."""

    @staticmethod
    def create_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
        """Create tokenizer of specified type."""
        if tokenizer_type.lower() == 'sentencepiece':
            return SentencePieceTokenizer(**kwargs)
        elif tokenizer_type.lower() == 'simple':
            return SimpleTokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


class TokenizationPipeline:
    """Pipeline for text tokenization with statistics."""

    def __init__(self, tokenizer: BaseTokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process single text with tokenization."""
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)

        # Truncate if max_length specified
        if self.max_length:
            tokens = tokens[:self.max_length]
            token_ids = token_ids[:self.max_length]

        return {
            'original_text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'num_tokens': len(tokens),
            'truncated': len(self.tokenizer.tokenize(text)) > len(tokens) if self.max_length else False
        }

    def process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process batch of texts."""
        return [self.process_text(text) for text in texts]

    def get_tokenization_stats(self, texts: List[str]) -> Dict[str, Any]:
        """Get tokenization statistics."""
        results = self.process_batch(texts)

        token_counts = [r['num_tokens'] for r in results]
        truncated_count = sum(1 for r in results if r['truncated'])

        stats = {
            'total_texts': len(texts),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_text': sum(token_counts) / len(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'truncated_texts': truncated_count,
            'truncation_rate': truncated_count / len(texts) if texts else 0,
            'vocab_size': self.tokenizer.get_vocab_size()
        }

        return stats

    def create_training_data(self, texts: List[str], format: str = 'ids') -> List[Union[List[int], List[str]]]:
        """Create training data in specified format."""
        if format == 'ids':
            return [self.tokenizer.encode(text) for text in texts]
        elif format == 'tokens':
            return [self.tokenizer.tokenize(text) for text in texts]
        else:
            raise ValueError(f"Unknown format: {format}")

    def estimate_dataset_tokens(self, texts: List[str], sample_size: int = 1000) -> int:
        """Estimate total tokens in dataset by sampling."""
        if len(texts) <= sample_size:
            sample = texts
        else:
            import random
            sample = random.sample(texts, sample_size)

        sample_stats = self.get_tokenization_stats(sample)
        avg_tokens = sample_stats['avg_tokens_per_text']

        return int(avg_tokens * len(texts))