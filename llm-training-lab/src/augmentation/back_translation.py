import random
import re
from typing import List, Dict, Any, Optional, Tuple
import logging


class BackTranslationSimulator:
    """Simulate back-translation for text augmentation."""

    def __init__(self, target_languages: Optional[List[str]] = None):
        self.target_languages = target_languages or ['es', 'fr', 'de', 'it', 'pt']
        self.logger = logging.getLogger(__name__)

        # Load language-specific transformation patterns
        self.language_patterns = self._load_language_patterns()
        self.grammar_patterns = self._load_grammar_patterns()

    def _load_language_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load language-specific transformation patterns."""
        return {
            'es': {  # Spanish
                'word_order': 'svo_flexible',
                'articles': {'the': 'el/la', 'a': 'un/una'},
                'common_words': {
                    'very': 'muy',
                    'good': 'bueno',
                    'bad': 'malo',
                    'big': 'grande',
                    'small': 'pequeño'
                },
                'grammar_quirks': [
                    'adjective_after_noun',
                    'double_negative',
                    'ser_vs_estar'
                ]
            },
            'fr': {  # French
                'word_order': 'svo',
                'articles': {'the': 'le/la/les', 'a': 'un/une'},
                'common_words': {
                    'very': 'très',
                    'good': 'bon',
                    'bad': 'mauvais',
                    'big': 'grand',
                    'small': 'petit'
                },
                'grammar_quirks': [
                    'adjective_agreement',
                    'liaison_effects',
                    'formal_informal'
                ]
            },
            'de': {  # German
                'word_order': 'svo_v2',
                'articles': {'the': 'der/die/das', 'a': 'ein/eine'},
                'common_words': {
                    'very': 'sehr',
                    'good': 'gut',
                    'bad': 'schlecht',
                    'big': 'groß',
                    'small': 'klein'
                },
                'grammar_quirks': [
                    'verb_second_position',
                    'compound_words',
                    'case_system'
                ]
            }
        }

    def _load_grammar_patterns(self) -> Dict[str, List[Dict[str, str]]]:
        """Load grammar transformation patterns that simulate translation effects."""
        return {
            'word_order_changes': [
                {
                    'pattern': r'(\w+) (is|are) (very|really|quite) (\w+)',
                    'replacement': r'\1 \2 \4 \3',  # "is very good" -> "is good very"
                    'probability': 0.3
                },
                {
                    'pattern': r'(the|a|an) (\w+) (red|blue|green|big|small) (\w+)',
                    'replacement': r'\1 \3 \2 \4',  # "the big house" -> "the house big"
                    'probability': 0.4
                }
            ],
            'article_changes': [
                {
                    'pattern': r'\ba\b(?=\s+[aeiou])',  # "a" before vowels
                    'replacement': 'an',
                    'probability': 0.2
                },
                {
                    'pattern': r'\ban\b(?=\s+[^aeiou])',  # "an" before consonants
                    'replacement': 'a',
                    'probability': 0.2
                },
                {
                    'pattern': r'\bthe\b',  # Sometimes remove "the"
                    'replacement': '',
                    'probability': 0.1
                }
            ],
            'preposition_changes': [
                {
                    'pattern': r'\bon\b',
                    'replacement': 'upon',
                    'probability': 0.2
                },
                {
                    'pattern': r'\bin\b',
                    'replacement': 'within',
                    'probability': 0.15
                },
                {
                    'pattern': r'\bwith\b',
                    'replacement': 'using',
                    'probability': 0.2
                }
            ],
            'verb_form_changes': [
                {
                    'pattern': r'\b(\w+)ing\b',  # -ing forms
                    'replacement': r'to \1',  # "walking" -> "to walk"
                    'probability': 0.1
                },
                {
                    'pattern': r'\bhave (\w+ed)\b',  # perfect tense
                    'replacement': r'\1',  # "have walked" -> "walked"
                    'probability': 0.15
                }
            ]
        }

    def simulate_translation_chain(self, text: str,
                                 intermediate_language: Optional[str] = None,
                                 intensity: float = 0.3) -> str:
        """Simulate translation through an intermediate language and back."""
        if intermediate_language is None:
            intermediate_language = random.choice(self.target_languages)

        # First "translation" to intermediate language
        intermediate_text = self._apply_language_effects(text, intermediate_language, intensity)

        # "Translation" back to English
        back_translated = self._apply_back_translation_effects(intermediate_text, intensity)

        return back_translated

    def _apply_language_effects(self, text: str, language: str, intensity: float) -> str:
        """Apply language-specific effects to simulate translation."""
        if language not in self.language_patterns:
            language = 'es'  # Default to Spanish patterns

        patterns = self.language_patterns[language]
        result = text

        # Apply grammar quirks based on the target language
        for quirk in patterns.get('grammar_quirks', []):
            if random.random() < intensity:
                result = self._apply_grammar_quirk(result, quirk)

        # Apply word order changes
        if patterns.get('word_order') in ['svo_flexible', 'svo_v2']:
            result = self._apply_word_order_changes(result, intensity)

        return result

    def _apply_grammar_quirk(self, text: str, quirk: str) -> str:
        """Apply specific grammar quirks."""
        if quirk == 'adjective_after_noun':
            # Move adjectives after nouns (Spanish/French style)
            pattern = r'\b(a|an|the)\s+(\w+)\s+(big|small|red|blue|green|good|bad)\s+(\w+)\b'
            replacement = r'\1 \4 \2 \3'  # "the big house" -> "the house big"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        elif quirk == 'double_negative':
            # Add double negatives (Spanish style)
            text = re.sub(r'\bnot\s+(\w+)\b', r'not \1 either', text)

        elif quirk == 'formal_informal':
            # Change contractions to formal forms
            contractions = {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not",
                "it's": "it is",
                "that's": "that is"
            }
            for contraction, formal in contractions.items():
                text = text.replace(contraction, formal)

        elif quirk == 'verb_second_position':
            # German V2 word order simulation
            pattern = r'^(\w+)\s+(is|are|was|were)\s+'
            if re.match(pattern, text):
                text = re.sub(pattern, r'\2 \1 ', text)

        return text

    def _apply_word_order_changes(self, text: str, intensity: float) -> str:
        """Apply word order changes."""
        grammar_patterns = self.grammar_patterns['word_order_changes']

        for pattern_dict in grammar_patterns:
            if random.random() < pattern_dict['probability'] * intensity:
                text = re.sub(
                    pattern_dict['pattern'],
                    pattern_dict['replacement'],
                    text,
                    flags=re.IGNORECASE
                )

        return text

    def _apply_back_translation_effects(self, text: str, intensity: float) -> str:
        """Apply effects that occur when translating back to English."""
        result = text

        # Apply various grammatical changes
        for category in ['article_changes', 'preposition_changes', 'verb_form_changes']:
            patterns = self.grammar_patterns.get(category, [])
            for pattern_dict in patterns:
                if random.random() < pattern_dict['probability'] * intensity:
                    result = re.sub(
                        pattern_dict['pattern'],
                        pattern_dict['replacement'],
                        result,
                        flags=re.IGNORECASE
                    )

        # Clean up any double spaces
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def generate_back_translations(self, text: str,
                                 num_variations: int = 3,
                                 intensity_range: Tuple[float, float] = (0.2, 0.5)) -> List[str]:
        """Generate multiple back-translation variations."""
        variations = []

        for _ in range(num_variations):
            # Random intensity for each variation
            intensity = random.uniform(*intensity_range)

            # Random intermediate language
            intermediate_lang = random.choice(self.target_languages)

            # Generate variation
            variation = self.simulate_translation_chain(text, intermediate_lang, intensity)

            # Avoid exact duplicates
            if variation not in variations and variation != text:
                variations.append(variation)

        return variations


class AdvancedBackTranslationSimulator:
    """More sophisticated back-translation simulation with linguistic rules."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.morphological_changes = self._load_morphological_patterns()
        self.syntactic_changes = self._load_syntactic_patterns()
        self.lexical_changes = self._load_lexical_patterns()

    def _load_morphological_patterns(self) -> Dict[str, List[Dict]]:
        """Load morphological transformation patterns."""
        return {
            'tense_changes': [
                {
                    'pattern': r'\b(\w+)ed\b',  # Past tense
                    'replacement': r'did \1',  # Analytic past
                    'probability': 0.1
                },
                {
                    'pattern': r'\bis (\w+)ing\b',  # Progressive
                    'replacement': r'is in the process of \1ing',
                    'probability': 0.05
                }
            ],
            'number_changes': [
                {
                    'pattern': r'\bmany\b',
                    'replacement': 'a lot of',
                    'probability': 0.3
                },
                {
                    'pattern': r'\bfew\b',
                    'replacement': 'a small number of',
                    'probability': 0.2
                }
            ]
        }

    def _load_syntactic_patterns(self) -> Dict[str, List[Dict]]:
        """Load syntactic transformation patterns."""
        return {
            'passive_voice': [
                {
                    'pattern': r'(\w+) (\w+ed) (\w+)',
                    'replacement': r'\3 was \2 by \1',
                    'probability': 0.1
                }
            ],
            'clause_reordering': [
                {
                    'pattern': r'(.+),\s*(because|since|as|when)\s*(.+)',
                    'replacement': r'\2 \3, \1',
                    'probability': 0.2
                }
            ]
        }

    def _load_lexical_patterns(self) -> Dict[str, str]:
        """Load lexical substitution patterns."""
        return {
            'utilize': 'use',
            'demonstrate': 'show',
            'facilitate': 'help',
            'subsequently': 'then',
            'commence': 'start',
            'terminate': 'end',
            'acquire': 'get',
            'assist': 'help',
            'endeavor': 'try',
            'obtain': 'get'
        }

    def apply_morphological_changes(self, text: str, intensity: float = 0.3) -> str:
        """Apply morphological changes."""
        result = text

        for category, patterns in self.morphological_changes.items():
            for pattern_dict in patterns:
                if random.random() < pattern_dict['probability'] * intensity:
                    result = re.sub(
                        pattern_dict['pattern'],
                        pattern_dict['replacement'],
                        result,
                        flags=re.IGNORECASE
                    )

        return result

    def apply_syntactic_changes(self, text: str, intensity: float = 0.3) -> str:
        """Apply syntactic changes."""
        result = text

        for category, patterns in self.syntactic_changes.items():
            for pattern_dict in patterns:
                if random.random() < pattern_dict['probability'] * intensity:
                    result = re.sub(
                        pattern_dict['pattern'],
                        pattern_dict['replacement'],
                        result,
                        flags=re.IGNORECASE
                    )

        return result

    def apply_lexical_changes(self, text: str, intensity: float = 0.3) -> str:
        """Apply lexical substitutions."""
        result = text
        words = text.split()

        num_changes = max(1, int(len(words) * intensity * 0.1))  # Change up to 10% of words
        changeable_words = [(i, word) for i, word in enumerate(words)
                           if word.lower() in self.lexical_changes]

        if changeable_words:
            words_to_change = random.sample(changeable_words, min(num_changes, len(changeable_words)))

            for idx, original_word in words_to_change:
                new_word = self.lexical_changes[original_word.lower()]

                # Preserve capitalization
                if original_word[0].isupper():
                    new_word = new_word.capitalize()

                words[idx] = new_word

            result = ' '.join(words)

        return result

    def simulate_advanced_back_translation(self, text: str, intensity: float = 0.3) -> str:
        """Apply advanced back-translation simulation."""
        result = text

        # Apply changes in order
        result = self.apply_lexical_changes(result, intensity)
        result = self.apply_morphological_changes(result, intensity)
        result = self.apply_syntactic_changes(result, intensity)

        # Final cleanup
        result = re.sub(r'\s+', ' ', result).strip()

        return result


class BackTranslationAugmenter:
    """Main class for back-translation augmentation."""

    def __init__(self, languages: Optional[List[str]] = None):
        self.basic_simulator = BackTranslationSimulator(languages)
        self.advanced_simulator = AdvancedBackTranslationSimulator()
        self.logger = logging.getLogger(__name__)

    def augment_text(self, text: str, method: str = 'basic',
                    intensity: float = 0.3, num_variations: int = 2) -> List[str]:
        """Generate back-translation augmented versions of text."""
        if method == 'basic':
            return self.basic_simulator.generate_back_translations(text, num_variations, (intensity, intensity + 0.2))
        elif method == 'advanced':
            variations = []
            for _ in range(num_variations):
                variation = self.advanced_simulator.simulate_advanced_back_translation(text, intensity)
                if variation != text and variation not in variations:
                    variations.append(variation)
            return variations
        else:
            raise ValueError(f"Unknown method: {method}")

    def augment_dataset(self, documents: List[Dict[str, Any]],
                       text_field: str = 'text',
                       method: str = 'basic',
                       augmentation_ratio: float = 0.5,
                       variations_per_doc: int = 2) -> List[Dict[str, Any]]:
        """Augment dataset with back-translation variations."""
        augmented_docs = documents.copy()

        # Determine how many documents to augment
        num_to_augment = int(len(documents) * augmentation_ratio)
        docs_to_augment = random.sample(documents, min(num_to_augment, len(documents)))

        self.logger.info(f"Augmenting {len(docs_to_augment)} documents with back-translation")

        for doc in docs_to_augment:
            if text_field not in doc:
                continue

            original_text = doc[text_field]
            variations = self.augment_text(original_text, method, 0.3, variations_per_doc)

            # Add variations as new documents
            for i, variation in enumerate(variations):
                augmented_doc = doc.copy()
                augmented_doc[text_field] = variation
                augmented_doc['is_back_translation'] = True
                augmented_doc['original_index'] = documents.index(doc)
                augmented_doc['variation_index'] = i

                augmented_docs.append(augmented_doc)

        self.logger.info(f"Added {len(augmented_docs) - len(documents)} back-translation variations")

        return augmented_docs

    def get_augmentation_stats(self, original_docs: List[Dict[str, Any]],
                             augmented_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the augmentation process."""
        back_translation_docs = [doc for doc in augmented_docs if doc.get('is_back_translation', False)]

        return {
            'original_count': len(original_docs),
            'augmented_count': len(augmented_docs),
            'back_translation_count': len(back_translation_docs),
            'augmentation_ratio': len(back_translation_docs) / len(original_docs) if original_docs else 0,
            'total_increase': (len(augmented_docs) - len(original_docs)) / len(original_docs) if original_docs else 0
        }