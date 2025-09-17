import pickle
import json
import gzip
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import logging
import hashlib
import time


class ProcessingCheckpoint:
    """Represents a processing checkpoint."""

    def __init__(self, checkpoint_id: str, step_name: str, data: Any, metadata: Dict[str, Any]):
        self.checkpoint_id = checkpoint_id
        self.step_name = step_name
        self.data = data
        self.metadata = metadata
        self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding data)."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'step_name': self.step_name,
            'metadata': self.metadata,
            'created_at': self.created_at
        }


class CheckpointManager:
    """Manage processing checkpoints for long-running operations."""

    def __init__(self, checkpoint_dir: str = "./checkpoints", compression: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.compression = compression
        self.logger = logging.getLogger(__name__)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Index file to track checkpoints
        self.index_file = self.checkpoint_dir / "checkpoint_index.json"
        self.checkpoint_index = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load checkpoint index."""
        if not self.index_file.exists():
            return {}

        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading checkpoint index: {e}")
            return {}

    def _save_index(self):
        """Save checkpoint index."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.checkpoint_index, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint index: {e}")
            raise

    def _generate_checkpoint_id(self, step_name: str, data: Any) -> str:
        """Generate unique checkpoint ID."""
        timestamp = str(int(time.time() * 1000))  # milliseconds
        data_hash = self._compute_data_hash(data)
        return f"{step_name}_{timestamp}_{data_hash[:8]}"

    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of data for identification."""
        try:
            if isinstance(data, (list, dict)):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)

            hasher = hashlib.md5()
            hasher.update(data_str.encode('utf-8'))
            return hasher.hexdigest()

        except Exception:
            # Fallback for non-serializable data
            return hashlib.md5(str(type(data)).encode()).hexdigest()

    def save_checkpoint(self, step_name: str, data: Any,
                       metadata: Optional[Dict[str, Any]] = None,
                       checkpoint_id: Optional[str] = None) -> str:
        """Save a processing checkpoint."""
        if checkpoint_id is None:
            checkpoint_id = self._generate_checkpoint_id(step_name, data)

        if metadata is None:
            metadata = {}

        # Add automatic metadata
        metadata.update({
            'data_type': type(data).__name__,
            'step_name': step_name,
            'created_at': datetime.now().isoformat()
        })

        if isinstance(data, (list, dict)):
            metadata['data_size'] = len(data)

        # Create checkpoint object
        checkpoint = ProcessingCheckpoint(checkpoint_id, step_name, data, metadata)

        # Save data to file
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        self._save_checkpoint_data(checkpoint_file, data)

        # Update index
        self.checkpoint_index[checkpoint_id] = checkpoint.to_dict()
        self._save_index()

        self.logger.info(f"Saved checkpoint {checkpoint_id} for step '{step_name}'")
        return checkpoint_id

    def _save_checkpoint_data(self, file_path: Path, data: Any):
        """Save checkpoint data to file."""
        try:
            if self.compression:
                with gzip.open(file_path.with_suffix('.pkl.gz'), 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)

        except Exception as e:
            self.logger.error(f"Error saving checkpoint data: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str) -> Any:
        """Load data from a checkpoint."""
        if checkpoint_id not in self.checkpoint_index:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        # Determine file path
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        compressed_file = checkpoint_file.with_suffix('.pkl.gz')

        if compressed_file.exists():
            file_path = compressed_file
            use_compression = True
        elif checkpoint_file.exists():
            file_path = checkpoint_file
            use_compression = False
        else:
            raise FileNotFoundError(f"Checkpoint data file not found for {checkpoint_id}")

        try:
            if use_compression:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

            self.logger.info(f"Loaded checkpoint {checkpoint_id}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            raise

    def list_checkpoints(self, step_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        checkpoints = list(self.checkpoint_index.values())

        if step_name:
            checkpoints = [cp for cp in checkpoints if cp['step_name'] == step_name]

        # Sort by creation time (newest first)
        checkpoints.sort(key=lambda cp: cp['created_at'], reverse=True)
        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint."""
        if checkpoint_id not in self.checkpoint_index:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        # Delete data files
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        compressed_file = checkpoint_file.with_suffix('.pkl.gz')

        for file_path in [checkpoint_file, compressed_file]:
            if file_path.exists():
                file_path.unlink()

        # Remove from index
        del self.checkpoint_index[checkpoint_id]
        self._save_index()

        self.logger.info(f"Deleted checkpoint {checkpoint_id}")

    def cleanup_old_checkpoints(self, keep_latest: int = 10,
                               max_age_days: Optional[int] = None):
        """Clean up old checkpoints."""
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return

        deleted_count = 0

        # Delete by age if specified
        if max_age_days:
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)

            for checkpoint in checkpoints:
                created_at = datetime.fromisoformat(checkpoint['created_at']).timestamp()
                if created_at < cutoff_time:
                    try:
                        self.delete_checkpoint(checkpoint['checkpoint_id'])
                        deleted_count += 1
                    except Exception as e:
                        self.logger.error(f"Error deleting checkpoint: {e}")

        # Keep only latest N checkpoints per step
        if keep_latest > 0:
            steps = {}
            for checkpoint in checkpoints:
                step_name = checkpoint['step_name']
                if step_name not in steps:
                    steps[step_name] = []
                steps[step_name].append(checkpoint)

            for step_name, step_checkpoints in steps.items():
                # Sort by creation time and keep latest
                step_checkpoints.sort(key=lambda cp: cp['created_at'], reverse=True)

                for checkpoint in step_checkpoints[keep_latest:]:
                    try:
                        self.delete_checkpoint(checkpoint['checkpoint_id'])
                        deleted_count += 1
                    except Exception as e:
                        self.logger.error(f"Error deleting checkpoint: {e}")

        self.logger.info(f"Cleaned up {deleted_count} old checkpoints")

    def get_latest_checkpoint(self, step_name: str) -> Optional[str]:
        """Get the latest checkpoint for a step."""
        checkpoints = self.list_checkpoints(step_name)

        if not checkpoints:
            return None

        return checkpoints[0]['checkpoint_id']  # Already sorted by creation time

    def checkpoint_exists(self, step_name: str) -> bool:
        """Check if any checkpoint exists for a step."""
        return len(self.list_checkpoints(step_name)) > 0

    def get_checkpoint_metadata(self, checkpoint_id: str) -> Dict[str, Any]:
        """Get metadata for a checkpoint."""
        if checkpoint_id not in self.checkpoint_index:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        return self.checkpoint_index[checkpoint_id].copy()

    def export_checkpoint(self, checkpoint_id: str, export_path: str, format: str = 'json'):
        """Export checkpoint data to external format."""
        data = self.load_checkpoint(checkpoint_id)
        metadata = self.get_checkpoint_metadata(checkpoint_id)

        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == 'json':
            export_data = {
                'checkpoint_metadata': metadata,
                'data': data
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

        elif format.lower() == 'pickle':
            with open(export_path, 'wb') as f:
                pickle.dump({
                    'checkpoint_metadata': metadata,
                    'data': data
                }, f)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")


class PipelineCheckpointer:
    """Checkpoint manager for data processing pipelines."""

    def __init__(self, pipeline_name: str, checkpoint_dir: str = "./checkpoints"):
        self.pipeline_name = pipeline_name
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.pipeline_steps = []
        self.current_step = 0
        self.logger = logging.getLogger(__name__)

    def register_steps(self, steps: List[str]):
        """Register pipeline steps."""
        self.pipeline_steps = steps
        self.current_step = 0

    def save_step_checkpoint(self, step_name: str, data: Any,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint for a pipeline step."""
        if metadata is None:
            metadata = {}

        # Add pipeline context to metadata
        metadata.update({
            'pipeline_name': self.pipeline_name,
            'step_index': self.pipeline_steps.index(step_name) if step_name in self.pipeline_steps else -1,
            'total_steps': len(self.pipeline_steps)
        })

        checkpoint_id = self.checkpoint_manager.save_checkpoint(
            f"{self.pipeline_name}_{step_name}",
            data,
            metadata
        )

        return checkpoint_id

    def load_step_checkpoint(self, step_name: str) -> Any:
        """Load the latest checkpoint for a pipeline step."""
        full_step_name = f"{self.pipeline_name}_{step_name}"
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(full_step_name)

        if latest_checkpoint is None:
            raise ValueError(f"No checkpoint found for step '{step_name}'")

        return self.checkpoint_manager.load_checkpoint(latest_checkpoint)

    def step_has_checkpoint(self, step_name: str) -> bool:
        """Check if a step has a checkpoint."""
        full_step_name = f"{self.pipeline_name}_{step_name}"
        return self.checkpoint_manager.checkpoint_exists(full_step_name)

    def find_resume_point(self) -> Optional[str]:
        """Find the latest completed step to resume from."""
        for i in range(len(self.pipeline_steps) - 1, -1, -1):
            step_name = self.pipeline_steps[i]
            if self.step_has_checkpoint(step_name):
                self.current_step = i
                return step_name

        return None

    def run_with_checkpoints(self, processing_functions: Dict[str, Callable],
                           initial_data: Any,
                           resume: bool = True) -> Any:
        """Run pipeline with automatic checkpointing."""
        if len(processing_functions) != len(self.pipeline_steps):
            raise ValueError("Number of processing functions must match pipeline steps")

        current_data = initial_data

        # Find resume point if requested
        start_step = 0
        if resume:
            resume_step = self.find_resume_point()
            if resume_step:
                self.logger.info(f"Resuming from step: {resume_step}")
                current_data = self.load_step_checkpoint(resume_step)
                start_step = self.pipeline_steps.index(resume_step) + 1

        # Execute remaining steps
        for i in range(start_step, len(self.pipeline_steps)):
            step_name = self.pipeline_steps[i]
            processing_fn = processing_functions[step_name]

            self.logger.info(f"Executing step {i+1}/{len(self.pipeline_steps)}: {step_name}")

            try:
                # Execute processing function
                current_data = processing_fn(current_data)

                # Save checkpoint
                self.save_step_checkpoint(step_name, current_data, {
                    'step_completed': True,
                    'execution_time': datetime.now().isoformat()
                })

                self.current_step = i

            except Exception as e:
                self.logger.error(f"Error in step '{step_name}': {e}")

                # Save error checkpoint
                self.save_step_checkpoint(f"{step_name}_error", current_data, {
                    'step_completed': False,
                    'error': str(e),
                    'execution_time': datetime.now().isoformat()
                })

                raise

        self.logger.info("Pipeline completed successfully")
        return current_data

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of pipeline execution."""
        status = {
            'pipeline_name': self.pipeline_name,
            'total_steps': len(self.pipeline_steps),
            'current_step': self.current_step,
            'steps': []
        }

        for i, step_name in enumerate(self.pipeline_steps):
            step_status = {
                'step_name': step_name,
                'step_index': i,
                'has_checkpoint': self.step_has_checkpoint(step_name),
                'completed': i <= self.current_step
            }

            if step_status['has_checkpoint']:
                try:
                    checkpoint_id = self.checkpoint_manager.get_latest_checkpoint(
                        f"{self.pipeline_name}_{step_name}"
                    )
                    metadata = self.checkpoint_manager.get_checkpoint_metadata(checkpoint_id)
                    step_status['checkpoint_metadata'] = metadata
                except Exception:
                    pass

            status['steps'].append(step_status)

        return status

    def cleanup_pipeline_checkpoints(self, keep_latest: int = 3):
        """Clean up checkpoints for this pipeline."""
        for step_name in self.pipeline_steps:
            full_step_name = f"{self.pipeline_name}_{step_name}"
            checkpoints = self.checkpoint_manager.list_checkpoints(full_step_name)

            # Keep only the latest N checkpoints for each step
            for checkpoint in checkpoints[keep_latest:]:
                try:
                    self.checkpoint_manager.delete_checkpoint(checkpoint['checkpoint_id'])
                except Exception as e:
                    self.logger.error(f"Error deleting checkpoint: {e}")

        self.logger.info(f"Cleaned up pipeline checkpoints, keeping latest {keep_latest} per step")