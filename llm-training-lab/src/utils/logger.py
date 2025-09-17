import logging
import logging.handlers
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Add context information to log records."""

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        if self.context:
            if not hasattr(record, 'extra_fields'):
                record.extra_fields = {}
            record.extra_fields.update(self.context)
        return True

    def update_context(self, new_context: Dict[str, Any]):
        """Update the context."""
        self.context.update(new_context)


class ProgressFilter(logging.Filter):
    """Filter to handle progress messages."""

    def __init__(self):
        super().__init__()
        self.last_progress_time = 0
        self.min_interval = 1.0  # Minimum seconds between progress logs

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter progress messages to avoid spam."""
        if hasattr(record, 'is_progress') and record.is_progress:
            current_time = record.created
            if current_time - self.last_progress_time < self.min_interval:
                return False
            self.last_progress_time = current_time
        return True


class LLMTrainingLabLogger:
    """Main logger for the LLM Training Lab."""

    def __init__(self, name: str = "llm_training_lab",
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: str = "standard",
                 enable_console: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):

        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        self.log_format = log_format
        self.enable_console = enable_console
        self.max_file_size = max_file_size
        self.backup_count = backup_count

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # Add context filter
        self.context_filter = ContextFilter()
        self.logger.addFilter(self.context_filter)

        # Add progress filter
        self.progress_filter = ProgressFilter()
        self.logger.addFilter(self.progress_filter)

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup logging handlers."""
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)

            if self.log_format == "json":
                console_formatter = JSONFormatter()
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )

            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)

            if self.log_format == "json":
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )

            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self, component: Optional[str] = None) -> logging.Logger:
        """Get logger instance for a component."""
        if component:
            logger_name = f"{self.name}.{component}"
            component_logger = logging.getLogger(logger_name)
            component_logger.setLevel(self.log_level)

            # Inherit handlers from parent if not already set
            if not component_logger.handlers:
                component_logger.parent = self.logger

            return component_logger

        return self.logger

    def set_context(self, **context):
        """Set logging context."""
        self.context_filter.update_context(context)

    def log_progress(self, message: str, **kwargs):
        """Log progress message."""
        logger = self.get_logger()
        record = logger.makeRecord(
            logger.name, logging.INFO, "", 0, message, (), None
        )
        record.is_progress = True
        if kwargs:
            record.extra_fields = kwargs
        logger.handle(record)

    def log_metrics(self, metrics: Dict[str, Any], component: str = "metrics"):
        """Log metrics data."""
        logger = self.get_logger(component)
        record = logger.makeRecord(
            logger.name, logging.INFO, "", 0, "Metrics update", (), None
        )
        record.extra_fields = {
            'metrics': metrics,
            'metric_type': 'performance'
        }
        logger.handle(record)

    def log_pipeline_step(self, step_name: str, status: str, **metadata):
        """Log pipeline step execution."""
        logger = self.get_logger("pipeline")
        message = f"Pipeline step '{step_name}': {status}"

        record = logger.makeRecord(
            logger.name, logging.INFO, "", 0, message, (), None
        )
        record.extra_fields = {
            'step_name': step_name,
            'status': status,
            'pipeline_metadata': metadata
        }
        logger.handle(record)

    def log_data_quality(self, quality_metrics: Dict[str, Any], dataset_info: Dict[str, Any]):
        """Log data quality metrics."""
        logger = self.get_logger("quality")
        message = "Data quality assessment completed"

        record = logger.makeRecord(
            logger.name, logging.INFO, "", 0, message, (), None
        )
        record.extra_fields = {
            'quality_metrics': quality_metrics,
            'dataset_info': dataset_info,
            'assessment_type': 'data_quality'
        }
        logger.handle(record)

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        logger = self.get_logger("error")
        message = f"Error occurred: {str(error)}"

        record = logger.makeRecord(
            logger.name, logging.ERROR, "", 0, message, (), (type(error), error, error.__traceback__)
        )
        record.extra_fields = {
            'error_context': context,
            'error_type': type(error).__name__
        }
        logger.handle(record)

    def create_performance_logger(self) -> 'PerformanceLogger':
        """Create a performance logger."""
        return PerformanceLogger(self.get_logger("performance"))


class PerformanceLogger:
    """Logger for performance monitoring."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}

    def start_timer(self, operation: str):
        """Start timing an operation."""
        import time
        self.timers[operation] = time.time()

    def end_timer(self, operation: str, **metadata):
        """End timing and log performance."""
        import time
        if operation not in self.timers:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return

        duration = time.time() - self.timers[operation]
        del self.timers[operation]

        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"Operation '{operation}' completed in {duration:.2f}s", (), None
        )
        record.extra_fields = {
            'operation': operation,
            'duration_seconds': duration,
            'performance_metadata': metadata
        }
        self.logger.handle(record)

    def log_throughput(self, operation: str, items_processed: int, duration: float, **metadata):
        """Log throughput metrics."""
        throughput = items_processed / duration if duration > 0 else 0

        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"Throughput for '{operation}': {throughput:.2f} items/second", (), None
        )
        record.extra_fields = {
            'operation': operation,
            'items_processed': items_processed,
            'duration_seconds': duration,
            'throughput_per_second': throughput,
            'throughput_metadata': metadata
        }
        self.logger.handle(record)

    def log_memory_usage(self, operation: str, **metadata):
        """Log memory usage."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            record = self.logger.makeRecord(
                self.logger.name, logging.INFO, "", 0,
                f"Memory usage for '{operation}': {memory_info.rss / 1024 / 1024:.2f} MB", (), None
            )
            record.extra_fields = {
                'operation': operation,
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_metadata': metadata
            }
            self.logger.handle(record)

        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")


def setup_logging(config: Optional[Dict[str, Any]] = None) -> LLMTrainingLabLogger:
    """Setup logging with configuration."""
    if config is None:
        config = {}

    logger_config = {
        'name': config.get('name', 'llm_training_lab'),
        'log_level': config.get('log_level', 'INFO'),
        'log_file': config.get('log_file'),
        'log_format': config.get('log_format', 'standard'),
        'enable_console': config.get('enable_console', True),
        'max_file_size': config.get('max_file_size', 10 * 1024 * 1024),
        'backup_count': config.get('backup_count', 5)
    }

    return LLMTrainingLabLogger(**logger_config)


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    if name is None:
        name = "llm_training_lab"

    return logging.getLogger(name)


class TimedContext:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(f"Completed operation '{self.operation_name}' in {duration:.2f}s")
        else:
            self.logger.error(f"Operation '{self.operation_name}' failed after {duration:.2f}s: {exc_val}")


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling function: {func.__name__}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise

    return wrapper


def log_exceptions(logger: Optional[logging.Logger] = None):
    """Decorator to log exceptions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator