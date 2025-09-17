import yaml
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import os
from dataclasses import dataclass, asdict
import logging


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    sources: List[Dict[str, Any]]
    batch_size: int = 1000
    max_workers: int = 4
    timeout: int = 30
    rate_limit: float = 1.0


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    cleaning: Dict[str, Any]
    tokenizer: Dict[str, Any]
    deduplication: Dict[str, Any]


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    filters: List[Dict[str, Any]]
    min_quality_score: float = 0.5
    enable_validators: bool = True


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    enabled: bool = True
    techniques: List[str]
    augmentation_ratio: float = 0.2
    intensity: float = 0.3


@dataclass
class OutputConfig:
    """Configuration for output formatting."""
    format: str = "jsonl"
    path: str = "/outputs/processed/"
    split_ratios: Dict[str, float] = None
    compression: str = "none"
    schema_mapping: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.split_ratios is None:
            self.split_ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "standard"
    metrics_export: bool = True
    dashboard: bool = False
    enable_performance_monitoring: bool = True


@dataclass
class ProjectConfig:
    """Project-level configuration."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    output_dir: str = "./outputs"
    tracking_dir: str = "./tracking"
    checkpoint_dir: str = "./checkpoints"


class ConfigManager:
    """Manage configuration for LLM Training Lab."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        self.logger = logging.getLogger(__name__)

        # Load configuration if path provided
        if self.config_path and self.config_path.exists():
            self.load_config(self.config_path)

    def load_config(self, config_path: Union[str, Path]):
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            self.config_path = config_path
            self.logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def save_config(self, output_path: Optional[Union[str, Path]] = None):
        """Save configuration to file."""
        if output_path is None:
            output_path = self.config_path

        if output_path is None:
            raise ValueError("No output path specified")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                elif output_path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported config format: {output_path.suffix}")

            self.logger.info(f"Saved configuration to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def get_project_config(self) -> ProjectConfig:
        """Get project configuration."""
        project_data = self.config.get('project', {})
        return ProjectConfig(**project_data)

    def get_ingestion_config(self) -> IngestionConfig:
        """Get ingestion configuration."""
        ingestion_data = self.config.get('ingestion', {})
        return IngestionConfig(**ingestion_data)

    def get_preprocessing_config(self) -> PreprocessingConfig:
        """Get preprocessing configuration."""
        preprocessing_data = self.config.get('preprocessing', {})

        # Set defaults for nested configs
        cleaning = preprocessing_data.get('cleaning', {})
        tokenizer = preprocessing_data.get('tokenizer', {'type': 'simple', 'vocab_size': 32000})
        deduplication = preprocessing_data.get('deduplication', {'method': 'exact', 'threshold': 0.9})

        return PreprocessingConfig(
            cleaning=cleaning,
            tokenizer=tokenizer,
            deduplication=deduplication
        )

    def get_quality_config(self) -> QualityConfig:
        """Get quality configuration."""
        quality_data = self.config.get('quality', {})
        return QualityConfig(**quality_data)

    def get_augmentation_config(self) -> AugmentationConfig:
        """Get augmentation configuration."""
        augmentation_data = self.config.get('augmentation', {})
        return AugmentationConfig(**augmentation_data)

    def get_output_config(self) -> OutputConfig:
        """Get output configuration."""
        output_data = self.config.get('output', {})
        return OutputConfig(**output_data)

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        monitoring_data = self.config.get('monitoring', {})
        return MonitoringConfig(**monitoring_data)

    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update a configuration section."""
        if section not in self.config:
            self.config[section] = {}

        self.config[section].update(updates)
        self.logger.info(f"Updated configuration section: {section}")

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set_config_value(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config_section = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]

        # Set the value
        config_section[keys[-1]] = value
        self.logger.info(f"Set configuration value: {key_path} = {value}")

    def merge_config(self, other_config: Dict[str, Any]):
        """Merge another configuration into this one."""
        def deep_merge(base: Dict, update: Dict) -> Dict:
            """Recursively merge dictionaries."""
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        self.config = deep_merge(self.config, other_config)
        self.logger.info("Merged external configuration")

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate project config
        project = self.config.get('project', {})
        if not project.get('name'):
            issues.append("Project name is required")

        # Validate ingestion config
        ingestion = self.config.get('ingestion', {})
        sources = ingestion.get('sources', [])
        if not sources:
            issues.append("At least one ingestion source is required")

        for i, source in enumerate(sources):
            if 'type' not in source:
                issues.append(f"Ingestion source {i} missing 'type' field")

        # Validate output config
        output = self.config.get('output', {})
        split_ratios = output.get('split_ratios', {})
        if split_ratios:
            total_ratio = sum(split_ratios.values())
            if abs(total_ratio - 1.0) > 0.01:
                issues.append(f"Split ratios must sum to 1.0, got {total_ratio}")

        # Validate paths
        paths_to_check = [
            ('output.path', 'Output path'),
            ('project.output_dir', 'Project output directory'),
            ('project.tracking_dir', 'Tracking directory'),
            ('project.checkpoint_dir', 'Checkpoint directory')
        ]

        for path_key, description in paths_to_check:
            path_value = self.get_config_value(path_key)
            if path_value:
                try:
                    Path(path_value).parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"{description} is invalid: {e}")

        return issues

    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration."""
        return {
            'project': asdict(ProjectConfig(name="my_training_dataset")),
            'ingestion': asdict(IngestionConfig(sources=[
                {'type': 'file', 'path': 'data/raw/*.jsonl'}
            ])),
            'preprocessing': {
                'cleaning': {
                    'remove_html': True,
                    'normalize_unicode': True,
                    'min_length': 10,
                    'max_length': 10000
                },
                'tokenizer': {
                    'type': 'sentencepiece',
                    'vocab_size': 32000
                },
                'deduplication': {
                    'method': 'minhash',
                    'threshold': 0.9
                }
            },
            'quality': asdict(QualityConfig(filters=[
                {'type': 'language', 'languages': ['en']},
                {'type': 'toxicity', 'max_score': 0.3}
            ])),
            'augmentation': asdict(AugmentationConfig()),
            'output': asdict(OutputConfig()),
            'monitoring': asdict(MonitoringConfig())
        }

    def export_config_template(self, output_path: str, format: str = 'yaml'):
        """Export a configuration template."""
        template_config = self.create_default_config()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(template_config, f, default_flow_style=False, allow_unicode=True)
                elif format.lower() == 'json':
                    json.dump(template_config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Exported configuration template to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting template: {e}")
            raise

    def load_from_env(self, prefix: str = "LLM_LAB_"):
        """Load configuration values from environment variables."""
        env_config = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert env var name to config key
                config_key = key[len(prefix):].lower().replace('_', '.')

                # Try to parse as JSON for complex values
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value

                self.set_config_value(config_key, parsed_value)

        self.logger.info(f"Loaded configuration from environment variables with prefix '{prefix}'")

    def get_resolved_path(self, path_key: str, base_dir: Optional[str] = None) -> Path:
        """Get resolved absolute path from configuration."""
        path_value = self.get_config_value(path_key)

        if not path_value:
            raise ValueError(f"Configuration key '{path_key}' not found or empty")

        path = Path(path_value)

        # Make relative paths absolute based on base_dir or config file location
        if not path.is_absolute():
            if base_dir:
                path = Path(base_dir) / path
            elif self.config_path:
                path = self.config_path.parent / path
            else:
                path = Path.cwd() / path

        return path.resolve()

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to configuration."""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting of configuration."""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator for configuration."""
        return key in self.config

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()


def load_config(config_path: str) -> ConfigManager:
    """Load configuration from file path."""
    return ConfigManager(config_path)


def create_default_config(output_path: str, format: str = 'yaml'):
    """Create a default configuration file."""
    config_manager = ConfigManager()
    config_manager.export_config_template(output_path, format)