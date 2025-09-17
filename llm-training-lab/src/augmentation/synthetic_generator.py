import random
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
import json


class TemplateEngine:
    """Template-based synthetic data generation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._load_templates()
        self.vocabularies = self._load_vocabularies()

    def _load_templates(self) -> Dict[str, List[str]]:
        """Load text generation templates."""
        return {
            'qa_pairs': [
                "What is {concept}? {concept} is {definition}.",
                "Define {concept}. {concept} refers to {definition}.",
                "Explain {concept}. {concept} can be described as {definition}.",
                "What does {concept} mean? {concept} means {definition}.",
                "Can you tell me about {concept}? {concept} is {definition}."
            ],
            'instructions': [
                "Please {action} {object}.",
                "I need you to {action} {object}.",
                "Can you {action} {object}?",
                "Help me {action} {object}.",
                "{action} {object}, please."
            ],
            'explanations': [
                "To {action} {object}, you need to {steps}.",
                "The process of {action} {object} involves {steps}.",
                "Here's how to {action} {object}: {steps}.",
                "{action} {object} requires the following steps: {steps}.",
                "Follow these steps to {action} {object}: {steps}."
            ],
            'comparisons': [
                "{item1} is {comparison} than {item2} because {reason}.",
                "Compared to {item2}, {item1} is {comparison} due to {reason}.",
                "The difference between {item1} and {item2} is that {item1} is {comparison}.",
                "{item1} differs from {item2} in that it is {comparison}."
            ],
            'descriptions': [
                "{subject} is characterized by {features}.",
                "{subject} typically has {features}.",
                "Key features of {subject} include {features}.",
                "{subject} is known for {features}.",
                "Notable aspects of {subject} are {features}."
            ]
        }

    def _load_vocabularies(self) -> Dict[str, List[str]]:
        """Load vocabulary lists for template filling."""
        return {
            'concepts': [
                'machine learning', 'artificial intelligence', 'data science',
                'neural networks', 'deep learning', 'natural language processing',
                'computer vision', 'robotics', 'algorithms', 'programming',
                'software engineering', 'web development', 'mobile apps',
                'cloud computing', 'cybersecurity', 'blockchain'
            ],
            'actions': [
                'create', 'build', 'develop', 'design', 'implement',
                'analyze', 'optimize', 'improve', 'test', 'debug',
                'deploy', 'maintain', 'configure', 'install', 'setup'
            ],
            'objects': [
                'a website', 'an application', 'a database', 'a model',
                'a system', 'a program', 'a script', 'a function',
                'an algorithm', 'a solution', 'a framework', 'a tool'
            ],
            'features': [
                'high performance', 'scalability', 'reliability', 'security',
                'user-friendly interface', 'efficiency', 'flexibility',
                'modularity', 'maintainability', 'robustness'
            ],
            'comparisons': [
                'faster', 'slower', 'more efficient', 'less complex',
                'more reliable', 'easier to use', 'more scalable',
                'more secure', 'more flexible', 'more robust'
            ],
            'reasons': [
                'it uses advanced algorithms',
                'it has better optimization',
                'it follows best practices',
                'it has modern architecture',
                'it uses efficient data structures',
                'it implements caching mechanisms'
            ]
        }

    def fill_template(self, template: str, variables: Optional[Dict[str, str]] = None) -> str:
        """Fill a template with provided or random variables."""
        if variables is None:
            variables = self._generate_random_variables(template)

        try:
            return template.format(**variables)
        except KeyError as e:
            self.logger.warning(f"Missing variable {e} in template")
            return template

    def _generate_random_variables(self, template: str) -> Dict[str, str]:
        """Generate random variables for template placeholders."""
        # Extract placeholder names from template
        placeholders = re.findall(r'\{(\w+)\}', template)
        variables = {}

        for placeholder in placeholders:
            if placeholder in self.vocabularies:
                variables[placeholder] = random.choice(self.vocabularies[placeholder])
            else:
                # Try to infer from placeholder name
                if 'concept' in placeholder:
                    variables[placeholder] = random.choice(self.vocabularies['concepts'])
                elif 'action' in placeholder:
                    variables[placeholder] = random.choice(self.vocabularies['actions'])
                elif 'object' in placeholder:
                    variables[placeholder] = random.choice(self.vocabularies['objects'])
                elif 'feature' in placeholder:
                    variables[placeholder] = random.choice(self.vocabularies['features'])
                elif 'comparison' in placeholder:
                    variables[placeholder] = random.choice(self.vocabularies['comparisons'])
                elif 'reason' in placeholder:
                    variables[placeholder] = random.choice(self.vocabularies['reasons'])
                else:
                    variables[placeholder] = f"[{placeholder}]"

        return variables

    def generate_from_template_type(self, template_type: str,
                                   count: int = 1,
                                   variables: Optional[Dict[str, str]] = None) -> List[str]:
        """Generate text samples from a specific template type."""
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")

        templates = self.templates[template_type]
        generated = []

        for _ in range(count):
            template = random.choice(templates)
            text = self.fill_template(template, variables)
            generated.append(text)

        return generated

    def generate_mixed_content(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate mixed content using different template types."""
        content = []

        for _ in range(count):
            template_type = random.choice(list(self.templates.keys()))
            text = self.generate_from_template_type(template_type, 1)[0]

            content.append({
                'text': text,
                'template_type': template_type,
                'generated': True,
                'timestamp': datetime.now().isoformat()
            })

        return content


class InstructionResponseGenerator:
    """Generate instruction-response pairs for training."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.instruction_patterns = self._load_instruction_patterns()
        self.response_patterns = self._load_response_patterns()

    def _load_instruction_patterns(self) -> Dict[str, List[str]]:
        """Load instruction generation patterns."""
        return {
            'how_to': [
                "How do I {action} {object}?",
                "What's the best way to {action} {object}?",
                "Can you explain how to {action} {object}?",
                "I need help {action} {object}. What should I do?",
                "Walk me through the process of {action} {object}."
            ],
            'what_is': [
                "What is {concept}?",
                "Can you define {concept}?",
                "Explain {concept} to me.",
                "I don't understand {concept}. Can you help?",
                "What does {concept} mean?"
            ],
            'why': [
                "Why is {concept} important?",
                "What are the benefits of {concept}?",
                "Why should I use {concept}?",
                "What makes {concept} useful?",
                "Why do people choose {concept}?"
            ],
            'compare': [
                "What's the difference between {item1} and {item2}?",
                "Compare {item1} vs {item2}.",
                "Which is better: {item1} or {item2}?",
                "How do {item1} and {item2} differ?",
                "Should I choose {item1} or {item2}?"
            ],
            'troubleshoot': [
                "I'm having trouble with {problem}. How can I fix it?",
                "My {object} isn't working. What should I do?",
                "How do I solve {problem}?",
                "I'm getting an error with {problem}. Help!",
                "What's wrong when {problem} happens?"
            ]
        }

    def _load_response_patterns(self) -> Dict[str, List[str]]:
        """Load response generation patterns."""
        return {
            'how_to': [
                "To {action} {object}, follow these steps:\n1. {step1}\n2. {step2}\n3. {step3}",
                "Here's how you can {action} {object}:\n\nFirst, {step1}. Then, {step2}. Finally, {step3}.",
                "The process to {action} {object} involves:\n- {step1}\n- {step2}\n- {step3}",
                "{action} {object} is straightforward. Start by {step1}, then {step2}, and finish with {step3}."
            ],
            'what_is': [
                "{concept} is {definition}. It's commonly used for {use_case} and has the following key features: {features}.",
                "{concept} refers to {definition}. This technology/concept is important because {importance}.",
                "Simply put, {concept} is {definition}. You'll typically encounter it when {use_case}.",
                "{concept} can be defined as {definition}. Its main advantages include {features}."
            ],
            'why': [
                "{concept} is important because {reason1}. Additionally, it offers {benefit1} and {benefit2}.",
                "The key benefits of {concept} include: {benefit1}, {benefit2}, and {benefit3}.",
                "You should consider {concept} because it {reason1} and {reason2}.",
                "{concept} is valuable due to its {benefit1} and ability to {benefit2}."
            ],
            'compare': [
                "The main differences between {item1} and {item2} are:\n{item1}: {feature1}\n{item2}: {feature2}",
                "{item1} is generally better for {use_case1}, while {item2} excels at {use_case2}.",
                "Choose {item1} if you need {feature1}. Choose {item2} if you prioritize {feature2}.",
                "Both have their strengths: {item1} offers {feature1}, whereas {item2} provides {feature2}."
            ],
            'troubleshoot': [
                "To fix {problem}, try these solutions:\n1. {solution1}\n2. {solution2}\n3. {solution3}",
                "This issue usually occurs when {cause}. You can resolve it by {solution1}.",
                "Common fixes for {problem} include: {solution1}, {solution2}, and {solution3}.",
                "Start troubleshooting by {solution1}. If that doesn't work, try {solution2}."
            ]
        }

    def generate_instruction_response_pair(self, instruction_type: str,
                                         variables: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Generate a single instruction-response pair."""
        if instruction_type not in self.instruction_patterns:
            raise ValueError(f"Unknown instruction type: {instruction_type}")

        # Generate instruction
        instruction_template = random.choice(self.instruction_patterns[instruction_type])
        response_template = random.choice(self.response_patterns[instruction_type])

        if variables is None:
            variables = self._generate_variables_for_type(instruction_type)

        instruction = instruction_template.format(**variables)
        response = response_template.format(**variables)

        return {
            'instruction': instruction,
            'response': response,
            'type': instruction_type,
            'variables': variables
        }

    def _generate_variables_for_type(self, instruction_type: str) -> Dict[str, str]:
        """Generate appropriate variables for instruction type."""
        variables = {}

        # Common variables
        concepts = ['machine learning', 'web development', 'data analysis', 'programming', 'software testing']
        actions = ['implement', 'optimize', 'debug', 'deploy', 'design']
        objects = ['a web application', 'a database', 'an API', 'a machine learning model', 'a mobile app']

        if instruction_type == 'how_to':
            variables.update({
                'action': random.choice(actions),
                'object': random.choice(objects),
                'step1': 'first step description',
                'step2': 'second step description',
                'step3': 'third step description'
            })

        elif instruction_type == 'what_is':
            variables.update({
                'concept': random.choice(concepts),
                'definition': 'a detailed definition',
                'use_case': 'specific use cases',
                'features': 'key features and benefits',
                'importance': 'why it matters'
            })

        elif instruction_type == 'why':
            variables.update({
                'concept': random.choice(concepts),
                'reason1': 'primary reason',
                'reason2': 'secondary reason',
                'benefit1': 'first benefit',
                'benefit2': 'second benefit',
                'benefit3': 'third benefit'
            })

        elif instruction_type == 'compare':
            items = ['React', 'Vue', 'Angular', 'Python', 'JavaScript', 'Java']
            selected_items = random.sample(items, 2)
            variables.update({
                'item1': selected_items[0],
                'item2': selected_items[1],
                'feature1': 'feature of first item',
                'feature2': 'feature of second item',
                'use_case1': 'use case for first item',
                'use_case2': 'use case for second item'
            })

        elif instruction_type == 'troubleshoot':
            problems = ['installation error', 'performance issue', 'connection problem', 'compilation error']
            variables.update({
                'problem': random.choice(problems),
                'object': random.choice(objects),
                'cause': 'common cause of the issue',
                'solution1': 'first solution to try',
                'solution2': 'second solution to try',
                'solution3': 'third solution to try'
            })

        return variables

    def generate_dataset(self, count: int = 100,
                        instruction_types: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Generate a dataset of instruction-response pairs."""
        if instruction_types is None:
            instruction_types = list(self.instruction_patterns.keys())

        dataset = []

        for _ in range(count):
            instruction_type = random.choice(instruction_types)
            pair = self.generate_instruction_response_pair(instruction_type)
            dataset.append(pair)

        self.logger.info(f"Generated {len(dataset)} instruction-response pairs")
        return dataset


class ConversationGenerator:
    """Generate multi-turn conversations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversation_flows = self._load_conversation_flows()

    def _load_conversation_flows(self) -> Dict[str, List[Dict]]:
        """Load conversation flow templates."""
        return {
            'technical_help': [
                {'speaker': 'user', 'template': "I need help with {problem}."},
                {'speaker': 'assistant', 'template': "I'd be happy to help with {problem}. Can you provide more details about {specific_aspect}?"},
                {'speaker': 'user', 'template': "Sure, {details}."},
                {'speaker': 'assistant', 'template': "Based on what you've described, here's what I suggest: {solution}"},
                {'speaker': 'user', 'template': "That makes sense. How do I {follow_up_question}?"},
                {'speaker': 'assistant', 'template': "To {follow_up_question}, you should {detailed_steps}."}
            ],
            'explanation_request': [
                {'speaker': 'user', 'template': "Can you explain {concept}?"},
                {'speaker': 'assistant', 'template': "{concept} is {basic_explanation}. Would you like me to elaborate on any specific aspect?"},
                {'speaker': 'user', 'template': "Yes, I'm particularly interested in {specific_interest}."},
                {'speaker': 'assistant', 'template': "Great question! {specific_interest} works by {detailed_explanation}. This is important because {importance}."},
                {'speaker': 'user', 'template': "That's helpful. Are there any common {related_question}?"},
                {'speaker': 'assistant', 'template': "Yes, common {related_question} include {examples}. I'd recommend {recommendation}."}
            ],
            'comparison_discussion': [
                {'speaker': 'user', 'template': "What's the difference between {option1} and {option2}?"},
                {'speaker': 'assistant', 'template': "Both {option1} and {option2} have their strengths. {option1} is better for {use_case1}, while {option2} excels at {use_case2}."},
                {'speaker': 'user', 'template': "I'm working on {project_context}. Which would you recommend?"},
                {'speaker': 'assistant', 'template': "For {project_context}, I'd lean towards {recommendation} because {reasoning}."},
                {'speaker': 'user', 'template': "What about {concern}? Is that something I should worry about?"},
                {'speaker': 'assistant', 'template': "{concern} is {assessment}. You can address this by {mitigation_strategy}."}
            ]
        }

    def generate_conversation(self, flow_type: str,
                            variables: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """Generate a conversation following a specific flow."""
        if flow_type not in self.conversation_flows:
            raise ValueError(f"Unknown flow type: {flow_type}")

        if variables is None:
            variables = self._generate_conversation_variables(flow_type)

        conversation = []
        flow = self.conversation_flows[flow_type]

        for turn in flow:
            message = {
                'speaker': turn['speaker'],
                'message': turn['template'].format(**variables),
                'turn_number': len(conversation) + 1
            }
            conversation.append(message)

        return conversation

    def _generate_conversation_variables(self, flow_type: str) -> Dict[str, str]:
        """Generate variables for conversation flows."""
        base_variables = {
            'problem': random.choice(['database connectivity', 'performance optimization', 'deployment issues']),
            'concept': random.choice(['machine learning', 'microservices', 'blockchain']),
            'option1': random.choice(['React', 'Python', 'PostgreSQL']),
            'option2': random.choice(['Vue', 'JavaScript', 'MongoDB']),
        }

        if flow_type == 'technical_help':
            base_variables.update({
                'specific_aspect': 'the error messages you\'re seeing',
                'details': 'I keep getting timeout errors',
                'solution': 'check your connection settings and increase the timeout value',
                'follow_up_question': 'implement this solution',
                'detailed_steps': 'modify the configuration file and restart the service'
            })

        elif flow_type == 'explanation_request':
            base_variables.update({
                'basic_explanation': 'a complex system with multiple components',
                'specific_interest': 'how the different parts work together',
                'detailed_explanation': 'using well-defined interfaces and protocols',
                'importance': 'it enables scalability and maintainability',
                'related_question': 'challenges',
                'examples': 'integration complexity and debugging difficulties',
                'recommendation': 'starting with a simple architecture'
            })

        elif flow_type == 'comparison_discussion':
            base_variables.update({
                'use_case1': 'rapid prototyping',
                'use_case2': 'large-scale applications',
                'project_context': 'a data-intensive application',
                'recommendation': base_variables['option2'],
                'reasoning': 'it handles large datasets more efficiently',
                'concern': 'learning curve',
                'assessment': 'a valid consideration',
                'mitigation_strategy': 'starting with online tutorials and practice projects'
            })

        return base_variables

    def generate_conversation_dataset(self, count: int = 50,
                                    flow_types: Optional[List[str]] = None) -> List[List[Dict[str, str]]]:
        """Generate multiple conversations."""
        if flow_types is None:
            flow_types = list(self.conversation_flows.keys())

        conversations = []

        for _ in range(count):
            flow_type = random.choice(flow_types)
            conversation = self.generate_conversation(flow_type)
            conversations.append(conversation)

        self.logger.info(f"Generated {len(conversations)} conversations")
        return conversations


class SyntheticDataGenerator:
    """Main synthetic data generation orchestrator."""

    def __init__(self):
        self.template_engine = TemplateEngine()
        self.instruction_generator = InstructionResponseGenerator()
        self.conversation_generator = ConversationGenerator()
        self.logger = logging.getLogger(__name__)

    def generate_mixed_dataset(self, total_count: int = 1000,
                             composition: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Generate mixed synthetic dataset with different types of content."""
        if composition is None:
            composition = {
                'templates': 0.4,
                'instructions': 0.4,
                'conversations': 0.2
            }

        # Ensure composition sums to 1.0
        total = sum(composition.values())
        if abs(total - 1.0) > 0.01:
            composition = {k: v/total for k, v in composition.items()}

        dataset = []

        # Generate template-based content
        template_count = int(total_count * composition['templates'])
        template_data = self.template_engine.generate_mixed_content(template_count)
        dataset.extend(template_data)

        # Generate instruction-response pairs
        instruction_count = int(total_count * composition['instructions'])
        instruction_data = self.instruction_generator.generate_dataset(instruction_count)
        for item in instruction_data:
            item['content_type'] = 'instruction_response'
        dataset.extend(instruction_data)

        # Generate conversations
        conversation_count = int(total_count * composition['conversations'])
        conversations = self.conversation_generator.generate_conversation_dataset(conversation_count)
        for i, conv in enumerate(conversations):
            dataset.append({
                'conversation': conv,
                'content_type': 'conversation',
                'conversation_id': i,
                'generated': True,
                'timestamp': datetime.now().isoformat()
            })

        # Shuffle the dataset
        random.shuffle(dataset)

        self.logger.info(f"Generated {len(dataset)} synthetic data samples")
        return dataset

    def export_dataset(self, dataset: List[Dict[str, Any]],
                      output_path: str, format: str = 'jsonl'):
        """Export synthetic dataset to file."""
        if format.lower() == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        elif format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Exported {len(dataset)} samples to {output_path}")

    def get_generation_stats(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about generated dataset."""
        content_types = {}
        for item in dataset:
            content_type = item.get('content_type', item.get('template_type', 'unknown'))
            content_types[content_type] = content_types.get(content_type, 0) + 1

        return {
            'total_samples': len(dataset),
            'content_type_distribution': content_types,
            'generation_timestamp': datetime.now().isoformat()
        }