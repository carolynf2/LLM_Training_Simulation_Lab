from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from ..utils.config import ConfigManager
from ..utils.logger import setup_logging, TimedContext
from ..utils.monitoring import MonitoringManager
from ..versioning.dataset_tracker import DatasetTracker
from ..versioning.checkpointing import PipelineCheckpointer

from ..ingestion.file_reader import FileReader
from ..ingestion.web_scraper import WebScraper
from ..ingestion.api_connector import APIConnector

from ..preprocessing.text_cleaner import TextCleaner, LanguageDetector
from ..preprocessing.tokenizer import TokenizerFactory, TokenizationPipeline
from ..preprocessing.deduplication import DeduplicationPipeline

from ..quality.metrics import QualityMetricsCalculator
from ..quality.filters import FilterPipeline, LengthFilter, LanguageFilter, QualityFilter, ToxicityFilter
from ..quality.validators import DataQualityValidator

from ..augmentation.paraphraser import CompositeParaphraser
from ..augmentation.back_translation import BackTranslationAugmenter
from ..augmentation.synthetic_generator import SyntheticDataGenerator

from ..formatting.jsonl_formatter import JSONLFormatter
from ..formatting.parquet_formatter import ParquetFormatter
from ..formatting.hf_dataset_builder import HuggingFaceDatasetBuilder


class DataPipeline:
    """Main data processing pipeline orchestrator."""

    def __init__(self, config: Union[str, Dict[str, Any], ConfigManager]):
        """Initialize the data pipeline."""
        # Setup configuration
        if isinstance(config, str):
            self.config = ConfigManager(config)
        elif isinstance(config, dict):
            self.config = ConfigManager()
            self.config.config = config
        else:
            self.config = config

        # Validate configuration
        issues = self.config.validate_config()
        if issues:
            raise ValueError(f"Configuration validation failed: {issues}")

        # Setup logging
        monitoring_config = self.config.get_monitoring_config()
        self.logger_manager = setup_logging({
            'log_level': monitoring_config.log_level,
            'log_file': monitoring_config.log_file,
            'log_format': monitoring_config.log_format
        })
        self.logger = self.logger_manager.get_logger('pipeline')

        # Setup monitoring
        self.monitoring = MonitoringManager({
            'enable_system_monitoring': monitoring_config.enable_performance_monitoring
        })

        # Setup versioning
        project_config = self.config.get_project_config()
        self.dataset_tracker = DatasetTracker(
            project_config.name,
            project_config.tracking_dir
        )

        # Setup checkpointing
        self.checkpointer = PipelineCheckpointer(
            f"{project_config.name}_pipeline",
            project_config.checkpoint_dir
        )

        # Initialize pipeline steps
        self.pipeline_steps = [
            "data_ingestion",
            "preprocessing",
            "quality_assessment",
            "data_augmentation",
            "format_conversion"
        ]
        self.checkpointer.register_steps(self.pipeline_steps)

        self.logger.info(f"Initialized data pipeline for project: {project_config.name}")

    def run(self, input_paths: Optional[List[str]] = None,
           output_path: Optional[str] = None,
           validate: bool = True,
           resume: bool = True) -> Dict[str, Any]:
        """Run the complete data processing pipeline."""
        with self.monitoring:
            self.logger.info("Starting data pipeline execution")

            # Define processing functions
            processing_functions = {
                "data_ingestion": self._run_ingestion,
                "preprocessing": self._run_preprocessing,
                "quality_assessment": self._run_quality_assessment,
                "data_augmentation": self._run_augmentation,
                "format_conversion": self._run_format_conversion
            }

            # Prepare initial data
            initial_data = {
                'input_paths': input_paths or [],
                'output_path': output_path,
                'validate': validate,
                'pipeline_start_time': datetime.now().isoformat()
            }

            try:
                # Run pipeline with checkpointing
                final_result = self.checkpointer.run_with_checkpoints(
                    processing_functions,
                    initial_data,
                    resume=resume
                )

                # Create dataset version
                if 'processed_documents' in final_result:
                    version_id = self.dataset_tracker.create_version(
                        final_result['processed_documents'],
                        description="Pipeline execution result",
                        preprocessing_steps=self.pipeline_steps,
                        source_info=final_result.get('source_info', {})
                    )
                    final_result['dataset_version'] = version_id

                self.logger.info("Pipeline execution completed successfully")
                return final_result

            except Exception as e:
                self.logger.error(f"Pipeline execution failed: {e}")
                raise

    def _run_ingestion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run data ingestion step."""
        with TimedContext("data_ingestion", self.logger):
            ingestion_config = self.config.get_ingestion_config()

            documents = []
            source_info = {'sources': [], 'total_documents': 0}

            # Process configured sources
            for source_config in ingestion_config.sources:
                source_type = source_config['type']

                if source_type == 'file':
                    docs = self._ingest_files(source_config)
                elif source_type == 'web':
                    docs = self._ingest_web(source_config)
                elif source_type == 'api':
                    docs = self._ingest_api(source_config)
                else:
                    self.logger.warning(f"Unknown source type: {source_type}")
                    continue

                documents.extend(docs)
                source_info['sources'].append({
                    'type': source_type,
                    'config': source_config,
                    'document_count': len(docs)
                })

            # Process input paths if provided
            if data.get('input_paths'):
                file_reader = FileReader()
                for input_path in data['input_paths']:
                    docs = list(file_reader.read_directory(input_path))
                    documents.extend(docs)

            source_info['total_documents'] = len(documents)

            self.logger.info(f"Ingested {len(documents)} documents from {len(source_info['sources'])} sources")

            data.update({
                'raw_documents': documents,
                'source_info': source_info
            })
            return data

    def _ingest_files(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ingest data from files."""
        file_reader = FileReader()
        path = config.get('path', '')

        if '*' in path or '?' in path:
            # Glob pattern
            from glob import glob
            files = glob(path)
            documents = []
            for file_path in files:
                documents.extend(list(file_reader.read_file(file_path)))
        else:
            # Single file or directory
            path_obj = Path(path)
            if path_obj.is_file():
                documents = list(file_reader.read_file(path))
            else:
                documents = list(file_reader.read_directory(path))

        return documents

    def _ingest_web(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ingest data from web sources."""
        scraper = WebScraper(
            delay=config.get('delay', 1.0),
            timeout=config.get('timeout', 30)
        )

        urls = config.get('urls', [])
        documents = []

        for result in scraper.scrape_urls(urls):
            if 'text' in result and result['text']:
                documents.append({
                    'text': result['text'],
                    'source': result['url'],
                    'title': result.get('title', ''),
                    'metadata': {
                        'content_type': 'web_scrape',
                        'scraped_at': result.get('scraped_at', ''),
                        'content_length': result.get('content_length', 0)
                    }
                })

        return documents

    def _ingest_api(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Ingest data from APIs."""
        api_connector = APIConnector()
        api_type = config.get('api_type', '')

        documents = []

        if api_type == 'wikipedia':
            query = config.get('query', '')
            limit = config.get('limit', 10)

            for result in api_connector.wikipedia_search(query, limit):
                documents.append({
                    'text': result['extract'],
                    'source': result['url'],
                    'title': result['title'],
                    'metadata': {
                        'content_type': 'wikipedia',
                        'timestamp': result.get('timestamp', ''),
                        'content_length': result.get('content_length', 0)
                    }
                })

        elif api_type == 'arxiv':
            query = config.get('query', '')
            max_results = config.get('max_results', 100)

            for result in api_connector.arxiv_search(query, max_results):
                documents.append({
                    'text': result['abstract'],
                    'source': result['url'],
                    'title': result['title'],
                    'metadata': {
                        'content_type': 'arxiv',
                        'authors': result.get('authors', []),
                        'published': result.get('published', '')
                    }
                })

        return documents

    def _run_preprocessing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run preprocessing step."""
        with TimedContext("preprocessing", self.logger):
            preprocessing_config = self.config.get_preprocessing_config()
            documents = data['raw_documents']

            # Text cleaning
            cleaning_config = preprocessing_config.cleaning
            text_cleaner = TextCleaner(
                remove_html=cleaning_config.get('remove_html', True),
                normalize_unicode=cleaning_config.get('normalize_unicode', True),
                min_length=cleaning_config.get('min_length', 10),
                max_length=cleaning_config.get('max_length', 10000)
            )

            # Language detection if configured
            target_languages = cleaning_config.get('target_languages', ['en'])
            if target_languages:
                language_detector = LanguageDetector(target_languages)

            cleaned_documents = []
            for doc in documents:
                if 'text' in doc:
                    cleaned_text = text_cleaner.clean_text(doc['text'])
                    if cleaned_text:
                        # Language filtering
                        if target_languages and not language_detector.filter_by_language(cleaned_text):
                            continue

                        doc = doc.copy()
                        doc['text'] = cleaned_text
                        cleaned_documents.append(doc)

            # Deduplication
            dedup_config = preprocessing_config.deduplication
            dedup_pipeline = DeduplicationPipeline(
                methods=[dedup_config.get('method', 'exact')],
                threshold=dedup_config.get('threshold', 0.9)
            )

            deduplicated_documents = dedup_pipeline.deduplicate(cleaned_documents)

            # Tokenization (optional - for analysis)
            tokenizer_config = preprocessing_config.tokenizer
            tokenizer = TokenizerFactory.create_tokenizer(
                tokenizer_config.get('type', 'simple'),
                vocab_size=tokenizer_config.get('vocab_size', 32000)
            )

            tokenization_pipeline = TokenizationPipeline(tokenizer)
            stats = tokenization_pipeline.get_tokenization_stats(
                [doc['text'] for doc in deduplicated_documents if 'text' in doc]
            )

            self.logger.info(f"Preprocessing: {len(documents)} -> {len(deduplicated_documents)} documents")
            self.logger.info(f"Tokenization stats: {stats}")

            data.update({
                'preprocessed_documents': deduplicated_documents,
                'preprocessing_stats': {
                    'original_count': len(documents),
                    'cleaned_count': len(cleaned_documents),
                    'final_count': len(deduplicated_documents),
                    'tokenization_stats': stats
                }
            })
            return data

    def _run_quality_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run quality assessment step."""
        with TimedContext("quality_assessment", self.logger):
            quality_config = self.config.get_quality_config()
            documents = data['preprocessed_documents']

            # Setup quality filters
            filter_pipeline = FilterPipeline()

            for filter_config in quality_config.filters:
                filter_type = filter_config['type']

                if filter_type == 'length':
                    filter_obj = LengthFilter(
                        min_length=filter_config.get('min_length', 10),
                        max_length=filter_config.get('max_length'),
                        unit=filter_config.get('unit', 'characters')
                    )
                elif filter_type == 'language':
                    filter_obj = LanguageFilter(
                        target_languages=filter_config.get('languages', ['en']),
                        confidence_threshold=filter_config.get('confidence', 0.7)
                    )
                elif filter_type == 'quality':
                    filter_obj = QualityFilter(
                        min_quality_score=filter_config.get('min_score', 0.5)
                    )
                elif filter_type == 'toxicity':
                    filter_obj = ToxicityFilter(
                        max_toxicity_score=filter_config.get('max_score', 0.3)
                    )
                else:
                    self.logger.warning(f"Unknown filter type: {filter_type}")
                    continue

                filter_pipeline.add_filter(filter_obj)

            # Apply filters
            filtered_documents, filter_stats = filter_pipeline.process(documents)

            # Data validation if enabled
            validation_results = {}
            if quality_config.enable_validators:
                validator = DataQualityValidator()
                validation_results = validator.comprehensive_validation(filtered_documents)

            # Calculate quality metrics
            quality_calculator = QualityMetricsCalculator()
            sample_docs = filtered_documents[:100] if len(filtered_documents) > 100 else filtered_documents
            quality_metrics = {}

            for i, doc in enumerate(sample_docs):
                if 'text' in doc:
                    metrics = quality_calculator.calculate_all_metrics(doc['text'])
                    for metric_name, value in metrics.items():
                        if metric_name not in quality_metrics:
                            quality_metrics[metric_name] = []
                        quality_metrics[metric_name].append(value)

            # Aggregate metrics
            aggregated_metrics = {}
            for metric_name, values in quality_metrics.items():
                if values:
                    aggregated_metrics[metric_name] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }

            self.logger.info(f"Quality assessment: {len(documents)} -> {len(filtered_documents)} documents")

            data.update({
                'quality_filtered_documents': filtered_documents,
                'quality_stats': {
                    'filter_stats': filter_stats,
                    'validation_results': validation_results,
                    'quality_metrics': aggregated_metrics
                }
            })
            return data

    def _run_augmentation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run data augmentation step."""
        with TimedContext("data_augmentation", self.logger):
            augmentation_config = self.config.get_augmentation_config()
            documents = data['quality_filtered_documents']

            if not augmentation_config.enabled:
                self.logger.info("Data augmentation disabled")
                data['augmented_documents'] = documents
                return data

            augmented_documents = documents.copy()

            # Paraphrasing
            if 'paraphrase' in augmentation_config.techniques:
                paraphraser = CompositeParaphraser()
                paraphrased_docs = paraphraser.generate_paraphrase_dataset(
                    documents,
                    paraphrases_per_doc=1,
                    methods=['rule_based', 'template']
                )
                augmented_documents.extend([doc for doc in paraphrased_docs if doc.get('is_paraphrase')])

            # Back translation
            if 'back_translation' in augmentation_config.techniques:
                back_translator = BackTranslationAugmenter()
                bt_docs = back_translator.augment_dataset(
                    documents,
                    augmentation_ratio=augmentation_config.augmentation_ratio,
                    variations_per_doc=1
                )
                augmented_documents.extend([doc for doc in bt_docs if doc.get('is_back_translation')])

            # Synthetic generation
            if 'synthetic' in augmentation_config.techniques:
                synthetic_generator = SyntheticDataGenerator()
                synthetic_count = int(len(documents) * augmentation_config.augmentation_ratio)
                synthetic_docs = synthetic_generator.generate_mixed_dataset(synthetic_count)

                # Convert to standard format
                for doc in synthetic_docs:
                    if 'text' not in doc and 'instruction' in doc:
                        doc['text'] = f"Instruction: {doc['instruction']}\nResponse: {doc.get('response', '')}"

                augmented_documents.extend(synthetic_docs)

            self.logger.info(f"Data augmentation: {len(documents)} -> {len(augmented_documents)} documents")

            data.update({
                'augmented_documents': augmented_documents,
                'augmentation_stats': {
                    'original_count': len(documents),
                    'augmented_count': len(augmented_documents),
                    'techniques_used': augmentation_config.techniques
                }
            })
            return data

    def _run_format_conversion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run format conversion step."""
        with TimedContext("format_conversion", self.logger):
            output_config = self.config.get_output_config()
            documents = data['augmented_documents']

            # Split data if ratios specified
            if output_config.split_ratios:
                hf_builder = HuggingFaceDatasetBuilder()
                split_documents = hf_builder.split_dataset(documents, output_config.split_ratios)
            else:
                split_documents = {'train': documents}

            # Determine output paths
            base_output_path = data.get('output_path') or output_config.path
            output_paths = {}

            # Convert to specified format
            for split_name, split_docs in split_documents.items():
                if output_config.format.lower() == 'jsonl':
                    output_path = f"{base_output_path}/{split_name}.jsonl"
                    if output_config.compression and output_config.compression != 'none':
                        output_path += f".{output_config.compression}"

                    formatter = JSONLFormatter(output_path, output_config.compression)
                    stats = formatter.write_documents(split_docs, output_config.schema_mapping)

                elif output_config.format.lower() == 'parquet':
                    output_path = f"{base_output_path}/{split_name}.parquet"
                    formatter = ParquetFormatter(output_path, output_config.compression)
                    stats = formatter.write_documents(split_docs, output_config.schema_mapping)

                elif output_config.format.lower() == 'hf_dataset':
                    output_path = f"{base_output_path}/{split_name}"
                    if not hasattr(self, '_hf_builder'):
                        self._hf_builder = HuggingFaceDatasetBuilder()

                    dataset = self._hf_builder.create_dataset(split_docs)
                    stats = self._hf_builder.save_dataset(dataset, output_path)

                else:
                    raise ValueError(f"Unsupported output format: {output_config.format}")

                output_paths[split_name] = output_path

            self.logger.info(f"Format conversion completed: {output_config.format}")

            data.update({
                'processed_documents': documents,
                'output_paths': output_paths,
                'format_stats': {
                    'output_format': output_config.format,
                    'splits': list(split_documents.keys()),
                    'total_documents': len(documents)
                }
            })
            return data

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'project_name': self.config.get_project_config().name,
            'pipeline_status': self.checkpointer.get_pipeline_status(),
            'monitoring_summary': self.monitoring.get_summary(),
            'dataset_versions': len(self.dataset_tracker.list_versions())
        }

    def cleanup(self, keep_checkpoints: int = 3, keep_versions: int = 5):
        """Clean up old checkpoints and versions."""
        self.checkpointer.cleanup_pipeline_checkpoints(keep_checkpoints)
        self.dataset_tracker.cleanup_old_versions(keep_versions)
        self.logger.info("Pipeline cleanup completed")

    def __enter__(self):
        self.monitoring.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitoring.stop()