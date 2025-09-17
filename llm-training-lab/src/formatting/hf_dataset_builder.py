from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
import logging
import json
import tempfile
import shutil


class HuggingFaceDatasetBuilder:
    """Build HuggingFace datasets from processed data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        try:
            from datasets import Dataset, DatasetDict, Features, Value, Sequence
            self.Dataset = Dataset
            self.DatasetDict = DatasetDict
            self.Features = Features
            self.Value = Value
            self.Sequence = Sequence
            self.available = True
        except ImportError:
            self.logger.error("datasets library not available")
            self.available = False

    def create_dataset(self, documents: List[Dict[str, Any]],
                      features: Optional['Features'] = None,
                      split_name: str = 'train') -> 'Dataset':
        """Create HuggingFace Dataset from documents."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        if not documents:
            raise ValueError("No documents provided")

        # Infer features if not provided
        if features is None:
            features = self._infer_features(documents)

        # Create dataset
        dataset = self.Dataset.from_list(documents, features=features)

        self.logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset

    def _infer_features(self, documents: List[Dict[str, Any]]) -> 'Features':
        """Infer dataset features from documents."""
        if not documents:
            return self.Features({})

        # Analyze first few documents to infer schema
        sample_docs = documents[:min(10, len(documents))]
        field_types = {}

        for doc in sample_docs:
            for field, value in doc.items():
                if field not in field_types:
                    field_types[field] = set()

                field_types[field].add(type(value).__name__)

        # Convert to HuggingFace features
        features = {}
        for field, types in field_types.items():
            if len(types) == 1:
                type_name = list(types)[0]
                features[field] = self._python_type_to_hf_feature(type_name, sample_docs, field)
            else:
                # Mixed types - default to string
                features[field] = self.Value('string')

        return self.Features(features)

    def _python_type_to_hf_feature(self, type_name: str, sample_docs: List[Dict[str, Any]], field: str):
        """Convert Python type to HuggingFace feature."""
        if type_name == 'str':
            return self.Value('string')
        elif type_name == 'int':
            return self.Value('int64')
        elif type_name == 'float':
            return self.Value('float64')
        elif type_name == 'bool':
            return self.Value('bool')
        elif type_name == 'list':
            # Try to infer list element type
            list_values = [doc.get(field, []) for doc in sample_docs if isinstance(doc.get(field), list)]
            if list_values and all(list_values):
                element_types = set()
                for lst in list_values:
                    for item in lst:
                        element_types.add(type(item).__name__)

                if len(element_types) == 1:
                    element_type = list(element_types)[0]
                    if element_type == 'str':
                        return self.Sequence(self.Value('string'))
                    elif element_type == 'int':
                        return self.Sequence(self.Value('int64'))
                    elif element_type == 'float':
                        return self.Sequence(self.Value('float64'))

            # Default to sequence of strings
            return self.Sequence(self.Value('string'))
        else:
            # Default to string for complex types
            return self.Value('string')

    def create_dataset_dict(self, datasets: Dict[str, List[Dict[str, Any]]],
                           features: Optional['Features'] = None) -> 'DatasetDict':
        """Create HuggingFace DatasetDict from multiple splits."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        dataset_dict = {}

        # Use same features for all splits if provided
        if features is None and datasets:
            # Infer from first non-empty split
            for split_data in datasets.values():
                if split_data:
                    features = self._infer_features(split_data)
                    break

        for split_name, documents in datasets.items():
            if documents:
                dataset_dict[split_name] = self.create_dataset(documents, features, split_name)
            else:
                self.logger.warning(f"Empty split: {split_name}")

        return self.DatasetDict(dataset_dict)

    def split_dataset(self, documents: List[Dict[str, Any]],
                     split_ratios: Dict[str, float] = None,
                     stratify_field: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Split documents into train/validation/test sets."""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'validation': 0.1, 'test': 0.1}

        # Validate split ratios
        total_ratio = sum(split_ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        if stratify_field:
            return self._stratified_split(documents, split_ratios, stratify_field)
        else:
            return self._random_split(documents, split_ratios)

    def _random_split(self, documents: List[Dict[str, Any]], split_ratios: Dict[str, float]) -> Dict[str, List[Dict[str, Any]]]:
        """Random split of documents."""
        import random

        # Shuffle documents
        shuffled_docs = documents.copy()
        random.shuffle(shuffled_docs)

        splits = {}
        start_idx = 0

        for split_name, ratio in split_ratios.items():
            split_size = int(len(documents) * ratio)
            end_idx = start_idx + split_size

            # Handle last split to include remaining documents
            if split_name == list(split_ratios.keys())[-1]:
                splits[split_name] = shuffled_docs[start_idx:]
            else:
                splits[split_name] = shuffled_docs[start_idx:end_idx]

            start_idx = end_idx

        self.logger.info(f"Split {len(documents)} documents: {[(k, len(v)) for k, v in splits.items()]}")
        return splits

    def _stratified_split(self, documents: List[Dict[str, Any]], split_ratios: Dict[str, float],
                         stratify_field: str) -> Dict[str, List[Dict[str, Any]]]:
        """Stratified split based on a field value."""
        from collections import defaultdict
        import random

        # Group documents by stratify field value
        groups = defaultdict(list)
        for doc in documents:
            if stratify_field in doc:
                groups[doc[stratify_field]].append(doc)
            else:
                groups['_missing_'].append(doc)

        splits = {split_name: [] for split_name in split_ratios.keys()}

        # Split each group proportionally
        for group_value, group_docs in groups.items():
            random.shuffle(group_docs)
            start_idx = 0

            for split_name, ratio in split_ratios.items():
                split_size = int(len(group_docs) * ratio)
                end_idx = start_idx + split_size

                # Handle last split
                if split_name == list(split_ratios.keys())[-1]:
                    splits[split_name].extend(group_docs[start_idx:])
                else:
                    splits[split_name].extend(group_docs[start_idx:end_idx])

                start_idx = end_idx

        self.logger.info(f"Stratified split on '{stratify_field}': {[(k, len(v)) for k, v in splits.items()]}")
        return splits

    def save_dataset(self, dataset: Union['Dataset', 'DatasetDict'], output_path: str) -> Dict[str, Any]:
        """Save dataset to disk."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            dataset.save_to_disk(str(output_path))

            stats = {
                'output_path': str(output_path),
                'dataset_type': type(dataset).__name__
            }

            if isinstance(dataset, self.DatasetDict):
                stats['splits'] = {split: len(ds) for split, ds in dataset.items()}
                stats['total_examples'] = sum(stats['splits'].values())
            else:
                stats['total_examples'] = len(dataset)

            self.logger.info(f"Saved dataset to {output_path}")
            return stats

        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")
            raise

    def load_dataset(self, dataset_path: str) -> Union['Dataset', 'DatasetDict']:
        """Load dataset from disk."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        try:
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
            self.logger.info(f"Loaded dataset from {dataset_path}")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def push_to_hub(self, dataset: Union['Dataset', 'DatasetDict'],
                   repo_id: str, private: bool = False,
                   token: Optional[str] = None) -> Dict[str, Any]:
        """Push dataset to HuggingFace Hub."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        try:
            dataset.push_to_hub(
                repo_id=repo_id,
                private=private,
                token=token
            )

            stats = {
                'repo_id': repo_id,
                'private': private,
                'dataset_type': type(dataset).__name__
            }

            if isinstance(dataset, self.DatasetDict):
                stats['splits'] = {split: len(ds) for split, ds in dataset.items()}
            else:
                stats['total_examples'] = len(dataset)

            self.logger.info(f"Pushed dataset to hub: {repo_id}")
            return stats

        except Exception as e:
            self.logger.error(f"Error pushing to hub: {e}")
            raise

    def apply_preprocessing(self, dataset: 'Dataset',
                          preprocess_fn: Callable,
                          batched: bool = False,
                          batch_size: int = 1000,
                          num_proc: Optional[int] = None) -> 'Dataset':
        """Apply preprocessing function to dataset."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        try:
            processed_dataset = dataset.map(
                preprocess_fn,
                batched=batched,
                batch_size=batch_size,
                num_proc=num_proc
            )

            self.logger.info(f"Applied preprocessing to {len(dataset)} examples")
            return processed_dataset

        except Exception as e:
            self.logger.error(f"Error applying preprocessing: {e}")
            raise

    def filter_dataset(self, dataset: 'Dataset',
                      filter_fn: Callable,
                      num_proc: Optional[int] = None) -> 'Dataset':
        """Filter dataset using a function."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        try:
            filtered_dataset = dataset.filter(filter_fn, num_proc=num_proc)

            self.logger.info(f"Filtered dataset: {len(dataset)} -> {len(filtered_dataset)} examples")
            return filtered_dataset

        except Exception as e:
            self.logger.error(f"Error filtering dataset: {e}")
            raise

    def get_dataset_info(self, dataset: Union['Dataset', 'DatasetDict']) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = {
            'dataset_type': type(dataset).__name__
        }

        if isinstance(dataset, self.DatasetDict):
            info['splits'] = {}
            total_examples = 0

            for split_name, split_dataset in dataset.items():
                split_info = {
                    'num_examples': len(split_dataset),
                    'features': list(split_dataset.features.keys()),
                    'num_columns': len(split_dataset.column_names)
                }
                info['splits'][split_name] = split_info
                total_examples += len(split_dataset)

            info['total_examples'] = total_examples

        else:
            info.update({
                'num_examples': len(dataset),
                'features': list(dataset.features.keys()),
                'num_columns': len(dataset.column_names),
                'features_details': {name: str(feature) for name, feature in dataset.features.items()}
            })

        return info

    def create_from_files(self, file_paths: List[str],
                         file_format: str = 'json',
                         split_config: Optional[Dict[str, float]] = None) -> 'DatasetDict':
        """Create dataset from files."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        from datasets import load_dataset

        try:
            # Load dataset from files
            if file_format.lower() == 'json':
                dataset = load_dataset('json', data_files=file_paths, split='train')
            elif file_format.lower() == 'csv':
                dataset = load_dataset('csv', data_files=file_paths, split='train')
            elif file_format.lower() == 'parquet':
                dataset = load_dataset('parquet', data_files=file_paths, split='train')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Convert to list for splitting if needed
            if split_config:
                documents = [dict(example) for example in dataset]
                split_docs = self.split_dataset(documents, split_config)
                return self.create_dataset_dict(split_docs)
            else:
                return self.DatasetDict({'train': dataset})

        except Exception as e:
            self.logger.error(f"Error creating dataset from files: {e}")
            raise

    def convert_to_streaming(self, dataset_path: str) -> 'Dataset':
        """Convert dataset to streaming format."""
        if not self.available:
            raise RuntimeError("datasets library not available")

        try:
            from datasets import load_dataset
            streaming_dataset = load_dataset(dataset_path, streaming=True)
            self.logger.info(f"Created streaming dataset from {dataset_path}")
            return streaming_dataset

        except Exception as e:
            self.logger.error(f"Error creating streaming dataset: {e}")
            raise