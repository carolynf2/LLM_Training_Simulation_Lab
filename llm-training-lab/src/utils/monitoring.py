import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from pathlib import Path


class MetricsCollector:
    """Collect and manage performance metrics."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics = defaultdict(deque)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None,
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if timestamp is None:
            timestamp = time.time()

        metric_entry = {
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        }

        with self.lock:
            self.metrics[name].append(metric_entry)

            # Maintain history size
            if len(self.metrics[name]) > self.history_size:
                self.metrics[name].popleft()

    def get_metric_history(self, name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metric history."""
        with self.lock:
            history = list(self.metrics[name])

        if limit:
            history = history[-limit:]

        return history

    def get_metric_stats(self, name: str, time_window: Optional[float] = None) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        history = self.get_metric_history(name)

        if not history:
            return {}

        # Filter by time window if specified
        if time_window:
            cutoff_time = time.time() - time_window
            history = [entry for entry in history if entry['timestamp'] >= cutoff_time]

        if not history:
            return {}

        values = [entry['value'] for entry in history]

        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1] if values else 0
        }

        # Calculate median
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            stats['median'] = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            stats['median'] = sorted_values[n//2]

        return stats

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all metrics."""
        all_stats = {}
        with self.lock:
            metric_names = list(self.metrics.keys())

        for name in metric_names:
            all_stats[name] = self.get_metric_stats(name)

        return all_stats

    def clear_metrics(self, name: Optional[str] = None):
        """Clear metrics."""
        with self.lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
            else:
                self.metrics.clear()


class SystemMonitor:
    """Monitor system resources."""

    def __init__(self, metrics_collector: MetricsCollector,
                 interval: float = 5.0, enabled: bool = True):
        self.metrics_collector = metrics_collector
        self.interval = interval
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)

        self._stop_event = threading.Event()
        self._monitor_thread = None

    def start(self):
        """Start system monitoring."""
        if not self.enabled:
            return

        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self.logger.info("Started system monitoring")

    def stop(self):
        """Stop system monitoring."""
        if self._monitor_thread:
            self._stop_event.set()
            self._monitor_thread.join(timeout=self.interval + 1)

        self.logger.info("Stopped system monitoring")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")

    def _collect_system_metrics(self):
        """Collect system metrics."""
        timestamp = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.metrics_collector.record_metric('system.cpu.percent', cpu_percent, timestamp)

        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric('system.memory.percent', memory.percent, timestamp)
        self.metrics_collector.record_metric('system.memory.used_gb', memory.used / (1024**3), timestamp)
        self.metrics_collector.record_metric('system.memory.available_gb', memory.available / (1024**3), timestamp)

        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics_collector.record_metric('system.disk.percent', disk.percent, timestamp)
        self.metrics_collector.record_metric('system.disk.used_gb', disk.used / (1024**3), timestamp)
        self.metrics_collector.record_metric('system.disk.free_gb', disk.free / (1024**3), timestamp)

        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        self.metrics_collector.record_metric('process.memory.rss_mb', process_memory.rss / (1024**2), timestamp)
        self.metrics_collector.record_metric('process.memory.vms_mb', process_memory.vms / (1024**2), timestamp)
        self.metrics_collector.record_metric('process.cpu.percent', process.cpu_percent(), timestamp)


class PerformanceTracker:
    """Track performance of operations."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_operations = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"

        operation_info = {
            'name': operation_name,
            'start_time': time.time(),
            'metadata': metadata or {}
        }

        with self.lock:
            self.active_operations[operation_id] = operation_info

        return operation_id

    def end_operation(self, operation_id: str, items_processed: Optional[int] = None,
                     error: Optional[str] = None):
        """End tracking an operation."""
        with self.lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Unknown operation ID: {operation_id}")
                return

            operation_info = self.active_operations.pop(operation_id)

        duration = time.time() - operation_info['start_time']
        operation_name = operation_info['name']

        # Record duration
        self.metrics_collector.record_metric(
            f'operation.{operation_name}.duration',
            duration,
            tags={'status': 'error' if error else 'success'}
        )

        # Record throughput if items processed
        if items_processed is not None:
            throughput = items_processed / duration if duration > 0 else 0
            self.metrics_collector.record_metric(
                f'operation.{operation_name}.throughput',
                throughput,
                tags={'unit': 'items_per_second'}
            )

        # Record success/error
        self.metrics_collector.record_metric(
            f'operation.{operation_name}.count',
            1,
            tags={'status': 'error' if error else 'success'}
        )

        self.logger.info(f"Operation {operation_name} completed in {duration:.2f}s")

    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active operations."""
        with self.lock:
            active = {}
            current_time = time.time()

            for op_id, op_info in self.active_operations.items():
                active[op_id] = {
                    'name': op_info['name'],
                    'duration': current_time - op_info['start_time'],
                    'metadata': op_info['metadata']
                }

        return active


class DataQualityMonitor:
    """Monitor data quality metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)

    def record_quality_metrics(self, dataset_name: str, metrics: Dict[str, float]):
        """Record data quality metrics."""
        timestamp = time.time()

        for metric_name, value in metrics.items():
            full_metric_name = f'quality.{dataset_name}.{metric_name}'
            self.metrics_collector.record_metric(full_metric_name, value, timestamp)

    def record_processing_stats(self, step_name: str, stats: Dict[str, Any]):
        """Record processing step statistics."""
        timestamp = time.time()

        numeric_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float))}

        for stat_name, value in numeric_stats.items():
            metric_name = f'processing.{step_name}.{stat_name}'
            self.metrics_collector.record_metric(metric_name, value, timestamp)

    def record_filter_stats(self, filter_name: str, original_count: int,
                           filtered_count: int, removed_count: int):
        """Record filtering statistics."""
        timestamp = time.time()
        removal_rate = removed_count / original_count if original_count > 0 else 0

        self.metrics_collector.record_metric(f'filter.{filter_name}.removal_rate', removal_rate, timestamp)
        self.metrics_collector.record_metric(f'filter.{filter_name}.removed_count', removed_count, timestamp)
        self.metrics_collector.record_metric(f'filter.{filter_name}.filtered_count', filtered_count, timestamp)


class AlertManager:
    """Manage alerts based on metrics."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.alert_handlers = []
        self.logger = logging.getLogger(__name__)

    def add_alert_rule(self, metric_name: str, condition: str, threshold: float,
                      message: str, severity: str = "warning"):
        """Add an alert rule."""
        rule = {
            'metric_name': metric_name,
            'condition': condition,  # 'gt', 'lt', 'eq'
            'threshold': threshold,
            'message': message,
            'severity': severity
        }
        self.alert_rules.append(rule)

    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    def check_alerts(self):
        """Check all alert rules and trigger alerts if needed."""
        for rule in self.alert_rules:
            metric_name = rule['metric_name']
            stats = self.metrics_collector.get_metric_stats(metric_name)

            if not stats:
                continue

            latest_value = stats.get('latest', 0)
            threshold = rule['threshold']
            condition = rule['condition']

            triggered = False
            if condition == 'gt' and latest_value > threshold:
                triggered = True
            elif condition == 'lt' and latest_value < threshold:
                triggered = True
            elif condition == 'eq' and abs(latest_value - threshold) < 0.001:
                triggered = True

            if triggered:
                alert = {
                    'metric_name': metric_name,
                    'current_value': latest_value,
                    'threshold': threshold,
                    'condition': condition,
                    'message': rule['message'],
                    'severity': rule['severity'],
                    'timestamp': time.time()
                }

                self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert."""
        self.logger.warning(f"ALERT: {alert['message']} (value: {alert['current_value']})")

        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")


class MonitoringDashboard:
    """Simple monitoring dashboard."""

    def __init__(self, metrics_collector: MetricsCollector,
                 performance_tracker: PerformanceTracker,
                 system_monitor: SystemMonitor):
        self.metrics_collector = metrics_collector
        self.performance_tracker = performance_tracker
        self.system_monitor = system_monitor
        self.logger = logging.getLogger(__name__)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self._get_system_metrics(),
            'performance_metrics': self._get_performance_metrics(),
            'active_operations': self.performance_tracker.get_active_operations(),
            'quality_metrics': self._get_quality_metrics()
        }

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics summary."""
        system_metrics = {}

        for metric_name in ['system.cpu.percent', 'system.memory.percent', 'system.disk.percent']:
            stats = self.metrics_collector.get_metric_stats(metric_name, time_window=300)  # Last 5 minutes
            if stats:
                system_metrics[metric_name] = stats

        return system_metrics

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        performance_metrics = {}

        all_metrics = self.metrics_collector.get_all_metrics()
        for metric_name, stats in all_metrics.items():
            if metric_name.startswith('operation.'):
                performance_metrics[metric_name] = stats

        return performance_metrics

    def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics summary."""
        quality_metrics = {}

        all_metrics = self.metrics_collector.get_all_metrics()
        for metric_name, stats in all_metrics.items():
            if metric_name.startswith('quality.'):
                quality_metrics[metric_name] = stats

        return quality_metrics

    def export_metrics(self, output_path: str, format: str = 'json'):
        """Export metrics to file."""
        dashboard_data = self.get_dashboard_data()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(dashboard_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Exported metrics to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            raise


class MonitoringManager:
    """Main monitoring manager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize components
        self.metrics_collector = MetricsCollector(
            history_size=self.config.get('history_size', 1000)
        )

        self.system_monitor = SystemMonitor(
            self.metrics_collector,
            interval=self.config.get('system_monitor_interval', 5.0),
            enabled=self.config.get('enable_system_monitoring', True)
        )

        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.quality_monitor = DataQualityMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)

        self.dashboard = MonitoringDashboard(
            self.metrics_collector,
            self.performance_tracker,
            self.system_monitor
        )

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start monitoring."""
        self.system_monitor.start()
        self.logger.info("Monitoring started")

    def stop(self):
        """Stop monitoring."""
        self.system_monitor.stop()
        self.logger.info("Monitoring stopped")

    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            'active_operations': len(self.performance_tracker.get_active_operations()),
            'total_metrics': len(self.metrics_collector.get_all_metrics()),
            'system_monitoring': self.system_monitor.enabled,
            'monitoring_config': self.config
        }

    def setup_default_alerts(self):
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            'system.cpu.percent', 'gt', 90.0,
            'High CPU usage detected', 'warning'
        )

        # High memory usage
        self.alert_manager.add_alert_rule(
            'system.memory.percent', 'gt', 90.0,
            'High memory usage detected', 'warning'
        )

        # Low disk space
        self.alert_manager.add_alert_rule(
            'system.disk.percent', 'gt', 90.0,
            'Low disk space detected', 'critical'
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()