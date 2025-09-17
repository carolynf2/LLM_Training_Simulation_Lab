import re
import random
from typing import List, Dict, Any, Optional
import logging


class RuleBasedParaphraser:
    """Rule-based text paraphrasing using linguistic transformations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.synonym_dict = self._load_synonyms()
        self.phrase_patterns = self._load_phrase_patterns()

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym dictionary for word replacement."""
        return {
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite'],
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'dreadful'],
            'fast': ['quick', 'rapid', 'speedy', 'swift', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'glad'],
            'sad': ['unhappy', 'sorrowful', 'melancholy', 'dejected', 'gloomy'],
            'important': ['significant', 'crucial', 'vital', 'essential', 'key'],
            'easy': ['simple', 'effortless', 'straightforward', 'uncomplicated'],
            'difficult': ['challenging', 'hard', 'tough', 'complex', 'demanding'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive', 'pretty'],
            'ugly': ['hideous', 'unattractive', 'unsightly', 'repulsive'],
            'smart': ['intelligent', 'clever', 'brilliant', 'wise', 'bright'],
            'stupid': ['foolish', 'dumb', 'ignorant', 'senseless', 'idiotic']
        }

    def _load_phrase_patterns(self) -> List[Dict[str, str]]:
        """Load phrase transformation patterns."""
        return [
            {'pattern': r'\bIn order to\b', 'replacement': 'To'},
            {'pattern': r'\bDue to the fact that\b', 'replacement': 'Because'},
            {'pattern': r'\bIt is important to note that\b', 'replacement': 'Note that'},
            {'pattern': r'\bA large number of\b', 'replacement': 'Many'},
            {'pattern': r'\bA small number of\b', 'replacement': 'Few'},
            {'pattern': r'\bAt this point in time\b', 'replacement': 'Now'},
            {'pattern': r'\bIn the near future\b', 'replacement': 'Soon'},
            {'pattern': r'\bWith regard to\b', 'replacement': 'Regarding'},
            {'pattern': r'\bFor the purpose of\b', 'replacement': 'To'},
            {'pattern': r'\bBy means of\b', 'replacement': 'By'},
        ]

    def synonym_replacement(self, text: str, replacement_rate: float = 0.1) -> str:
        """Replace words with synonyms."""
        words = text.split()
        num_replacements = max(1, int(len(words) * replacement_rate))

        replaceable_indices = []
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.synonym_dict:
                replaceable_indices.append(i)

        if not replaceable_indices:
            return text

        # Randomly select words to replace
        indices_to_replace = random.sample(
            replaceable_indices,
            min(num_replacements, len(replaceable_indices))
        )

        new_words = words.copy()
        for idx in indices_to_replace:
            original_word = words[idx]
            clean_word = re.sub(r'[^\w]', '', original_word.lower())

            if clean_word in self.synonym_dict:
                synonyms = self.synonym_dict[clean_word]
                new_synonym = random.choice(synonyms)

                # Preserve capitalization and punctuation
                if original_word[0].isupper():
                    new_synonym = new_synonym.capitalize()

                # Preserve punctuation at the end
                punctuation = re.findall(r'[^\w]+$', original_word)
                if punctuation:
                    new_synonym += punctuation[0]

                new_words[idx] = new_synonym

        return ' '.join(new_words)

    def phrase_transformation(self, text: str) -> str:
        """Transform phrases using predefined patterns."""
        result = text
        for pattern_dict in self.phrase_patterns:
            pattern = pattern_dict['pattern']
            replacement = pattern_dict['replacement']
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def sentence_restructuring(self, text: str) -> str:
        """Restructure sentences by changing voice or order."""
        sentences = re.split(r'[.!?]+', text)
        restructured = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Try passive to active voice conversion
            restructured_sentence = self._passive_to_active(sentence)
            if restructured_sentence == sentence:
                # Try other restructuring
                restructured_sentence = self._reorder_clauses(sentence)

            restructured.append(restructured_sentence)

        return '. '.join(restructured) + '.'

    def _passive_to_active(self, sentence: str) -> str:
        """Convert passive voice to active voice (simplified)."""
        # Simple pattern matching for "was/were + past participle + by"
        passive_pattern = r'(\w+)\s+(was|were)\s+(\w+ed|done|taken|given|made)\s+by\s+(\w+)'
        match = re.search(passive_pattern, sentence, re.IGNORECASE)

        if match:
            subject = match.group(1)
            verb = match.group(2)
            past_participle = match.group(3)
            agent = match.group(4)

            # Simple conversion (this is very basic)
            active_verb = self._past_participle_to_active(past_participle)
            if active_verb:
                new_sentence = sentence.replace(
                    match.group(0),
                    f"{agent} {active_verb} {subject}"
                )
                return new_sentence

        return sentence

    def _past_participle_to_active(self, past_participle: str) -> Optional[str]:
        """Convert past participle to active verb form."""
        conversions = {
            'taken': 'took',
            'given': 'gave',
            'made': 'made',
            'done': 'did',
            'written': 'wrote',
            'spoken': 'spoke',
            'broken': 'broke'
        }
        return conversions.get(past_participle.lower())

    def _reorder_clauses(self, sentence: str) -> str:
        """Reorder clauses in a sentence."""
        # Look for sentences with conjunctions
        conjunctions = ['and', 'but', 'or', 'because', 'although', 'while', 'when']

        for conj in conjunctions:
            if f' {conj} ' in sentence.lower():
                parts = re.split(f'\\s+{re.escape(conj)}\\s+', sentence, 1, re.IGNORECASE)
                if len(parts) == 2:
                    # Swap the order
                    return f"{parts[1].strip()}, {conj} {parts[0].strip()}"

        return sentence

    def paraphrase(self, text: str, methods: Optional[List[str]] = None,
                  intensity: float = 0.3) -> str:
        """Generate paraphrase using specified methods."""
        if methods is None:
            methods = ['synonym', 'phrase', 'structure']

        result = text

        if 'synonym' in methods:
            result = self.synonym_replacement(result, intensity)

        if 'phrase' in methods:
            result = self.phrase_transformation(result)

        if 'structure' in methods and random.random() < intensity:
            result = self.sentence_restructuring(result)

        return result

    def generate_multiple_paraphrases(self, text: str, num_paraphrases: int = 3,
                                    methods: Optional[List[str]] = None) -> List[str]:
        """Generate multiple paraphrases of the same text."""
        paraphrases = []

        for _ in range(num_paraphrases):
            # Vary the intensity for each paraphrase
            intensity = random.uniform(0.2, 0.5)
            paraphrase = self.paraphrase(text, methods, intensity)

            # Avoid exact duplicates
            if paraphrase not in paraphrases and paraphrase != text:
                paraphrases.append(paraphrase)

        return paraphrases


class TemplateBasedParaphraser:
    """Template-based paraphrasing for structured text."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.question_templates = self._load_question_templates()
        self.instruction_templates = self._load_instruction_templates()

    def _load_question_templates(self) -> List[Dict[str, str]]:
        """Load question paraphrasing templates."""
        return [
            {
                'pattern': r'^What is (.+)\?$',
                'templates': [
                    'Can you explain what {} is?',
                    'Define {}.',
                    'What does {} mean?',
                    'Could you tell me about {}?'
                ]
            },
            {
                'pattern': r'^How do you (.+)\?$',
                'templates': [
                    'What is the process to {}?',
                    'Can you describe how to {}?',
                    'What are the steps to {}?',
                    'How can I {}?'
                ]
            },
            {
                'pattern': r'^Why (.+)\?$',
                'templates': [
                    'What is the reason {}?',
                    'Can you explain why {}?',
                    'What causes {}?',
                    'For what purpose {}?'
                ]
            }
        ]

    def _load_instruction_templates(self) -> List[Dict[str, str]]:
        """Load instruction paraphrasing templates."""
        return [
            {
                'pattern': r'^Please (.+)$',
                'templates': [
                    'Could you {}?',
                    'I need you to {}.',
                    'Can you {}?',
                    '{}, please.'
                ]
            },
            {
                'pattern': r'^Create (.+)$',
                'templates': [
                    'Generate {}.',
                    'Make {}.',
                    'Build {}.',
                    'Develop {}.'
                ]
            }
        ]

    def paraphrase_question(self, question: str) -> Optional[str]:
        """Paraphrase a question using templates."""
        for template_dict in self.question_templates:
            pattern = template_dict['pattern']
            templates = template_dict['templates']

            match = re.match(pattern, question, re.IGNORECASE)
            if match:
                content = match.group(1)
                template = random.choice(templates)
                return template.format(content)

        return None

    def paraphrase_instruction(self, instruction: str) -> Optional[str]:
        """Paraphrase an instruction using templates."""
        for template_dict in self.instruction_templates:
            pattern = template_dict['pattern']
            templates = template_dict['templates']

            match = re.match(pattern, instruction, re.IGNORECASE)
            if match:
                content = match.group(1)
                template = random.choice(templates)
                return template.format(content)

        return None

    def paraphrase_text(self, text: str) -> Optional[str]:
        """Paraphrase text by detecting type and applying appropriate templates."""
        text = text.strip()

        # Try question paraphrasing
        if text.endswith('?'):
            result = self.paraphrase_question(text)
            if result:
                return result

        # Try instruction paraphrasing
        instruction_starters = ['please', 'create', 'make', 'generate', 'write', 'explain']
        if any(text.lower().startswith(starter) for starter in instruction_starters):
            result = self.paraphrase_instruction(text)
            if result:
                return result

        return None


class BackTranslationSimulator:
    """Simulate back-translation effects for paraphrasing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.translation_effects = self._load_translation_effects()

    def _load_translation_effects(self) -> Dict[str, List[str]]:
        """Load simulated translation effects."""
        return {
            'word_order_changes': [
                r'(\w+) (\w+) (very|really|quite) (\w+)',  # "very good" -> "good very"
                r'(the|a|an) (\w+) (of|for|to) (\w+)',      # "the book of John" -> "John's book"
            ],
            'article_changes': [
                r'\ba\b',  # Remove/change articles
                r'\ban\b',
                r'\bthe\b'
            ],
            'preposition_changes': {
                'on': ['upon', 'at'],
                'in': ['within', 'inside'],
                'with': ['using', 'by means of'],
                'for': ['in order to', 'to']
            },
            'formality_changes': {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not",
                "it's": "it is",
                "that's": "that is"
            }
        }

    def simulate_back_translation(self, text: str, intensity: float = 0.3) -> str:
        """Simulate back-translation effects."""
        result = text

        # Apply formality changes
        if random.random() < intensity:
            result = self._apply_formality_changes(result)

        # Apply preposition changes
        if random.random() < intensity:
            result = self._apply_preposition_changes(result)

        # Apply article changes (be conservative)
        if random.random() < intensity * 0.5:
            result = self._apply_article_changes(result)

        return result

    def _apply_formality_changes(self, text: str) -> str:
        """Apply formality level changes."""
        formality_changes = self.translation_effects['formality_changes']

        for informal, formal in formality_changes.items():
            if random.random() < 0.5:  # 50% chance to apply each change
                text = re.sub(r'\b' + re.escape(informal) + r'\b', formal, text, flags=re.IGNORECASE)

        return text

    def _apply_preposition_changes(self, text: str) -> str:
        """Apply preposition changes."""
        preposition_changes = self.translation_effects['preposition_changes']

        for original, alternatives in preposition_changes.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))

            for match in matches:
                if random.random() < 0.3:  # 30% chance to change each preposition
                    replacement = random.choice(alternatives)
                    text = text[:match.start()] + replacement + text[match.end():]

        return text

    def _apply_article_changes(self, text: str) -> str:
        """Apply article changes (very conservative)."""
        # Only remove articles in specific contexts where it's grammatically acceptable
        # This is a simplified implementation

        # Remove articles before proper nouns (very basic detection)
        text = re.sub(r'\b(the|a|an)\s+([A-Z]\w+)\b', r'\2', text)

        return text


class CompositeParaphraser:
    """Combine multiple paraphrasing techniques."""

    def __init__(self):
        self.rule_based = RuleBasedParaphraser()
        self.template_based = TemplateBasedParaphraser()
        self.back_translation = BackTranslationSimulator()
        self.logger = logging.getLogger(__name__)

    def paraphrase(self, text: str, methods: Optional[List[str]] = None,
                  intensity: float = 0.3) -> str:
        """Generate paraphrase using multiple methods."""
        if methods is None:
            methods = ['rule_based', 'template', 'back_translation']

        result = text

        # Apply template-based first (for structured text)
        if 'template' in methods:
            template_result = self.template_based.paraphrase_text(result)
            if template_result:
                result = template_result

        # Apply rule-based transformations
        if 'rule_based' in methods:
            result = self.rule_based.paraphrase(result, intensity=intensity)

        # Apply back-translation simulation
        if 'back_translation' in methods:
            result = self.back_translation.simulate_back_translation(result, intensity)

        return result

    def generate_paraphrase_dataset(self, documents: List[Dict[str, Any]],
                                  text_field: str = 'text',
                                  paraphrases_per_doc: int = 2,
                                  methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate paraphrased versions of documents."""
        augmented_docs = []

        for doc in documents:
            if text_field not in doc:
                augmented_docs.append(doc)
                continue

            original_text = doc[text_field]

            # Add original document
            augmented_docs.append(doc)

            # Generate paraphrases
            for i in range(paraphrases_per_doc):
                paraphrased_text = self.paraphrase(original_text, methods)

                # Create new document with paraphrased text
                paraphrased_doc = doc.copy()
                paraphrased_doc[text_field] = paraphrased_text
                paraphrased_doc['is_paraphrase'] = True
                paraphrased_doc['original_index'] = len(augmented_docs) - paraphrases_per_doc

                augmented_docs.append(paraphrased_doc)

        self.logger.info(f"Generated {len(augmented_docs) - len(documents)} paraphrases from {len(documents)} documents")

        return augmented_docs