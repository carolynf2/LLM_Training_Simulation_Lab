import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import shutil


class DatasetVersion:
    """Represents a single dataset version."""

    def __init__(self, version_id: str, metadata: Dict[str, Any]):
        self.version_id = version_id
        self.metadata = metadata
        self.created_at = metadata.get('created_at', datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'version_id': self.version_id,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create from dictionary representation."""
        return cls(
            version_id=data['version_id'],
            metadata=data['metadata']
        )


class DatasetTracker:
    """Track dataset versions and manage dataset lifecycle."""

    def __init__(self, project_name: str, tracking_dir: str = "./tracking"):
        self.project_name = project_name
        self.tracking_dir = Path(tracking_dir)
        self.project_dir = self.tracking_dir / project_name
        self.versions_file = self.project_dir / "versions.json"
        self.logger = logging.getLogger(__name__)

        # Create tracking directories
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.project_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)

        # Load or initialize versions
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict[str, DatasetVersion]:
        """Load versions from disk."""
        if not self.versions_file.exists():
            return {}

        try:
            with open(self.versions_file, 'r') as f:
                data = json.load(f)

            versions = {}
            for version_id, version_data in data.items():
                versions[version_id] = DatasetVersion.from_dict(version_data)

            return versions

        except Exception as e:
            self.logger.error(f"Error loading versions: {e}")
            return {}

    def _save_versions(self):
        """Save versions to disk."""
        try:
            data = {
                version_id: version.to_dict()
                for version_id, version in self.versions.items()
            }

            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving versions: {e}")
            raise

    def compute_dataset_fingerprint(self, documents: List[Dict[str, Any]]) -> str:
        """Compute SHA-256 fingerprint of dataset."""
        # Create stable representation of dataset
        stable_repr = []

        for doc in sorted(documents, key=lambda x: json.dumps(x, sort_keys=True)):
            # Convert document to stable JSON string
            doc_str = json.dumps(doc, sort_keys=True, ensure_ascii=False)
            stable_repr.append(doc_str)

        # Combine all documents
        dataset_str = '\n'.join(stable_repr)

        # Compute hash
        hasher = hashlib.sha256()
        hasher.update(dataset_str.encode('utf-8'))

        return hasher.hexdigest()

    def create_version(self, documents: List[Dict[str, Any]],
                      version_name: Optional[str] = None,
                      description: str = "",
                      tags: Optional[List[str]] = None,
                      preprocessing_steps: Optional[List[str]] = None,
                      source_info: Optional[Dict[str, Any]] = None) -> str:
        """Create a new dataset version."""
        # Compute fingerprint
        fingerprint = self.compute_dataset_fingerprint(documents)

        # Check if this exact dataset already exists
        for existing_version in self.versions.values():
            if existing_version.metadata.get('fingerprint') == fingerprint:
                self.logger.warning(f"Dataset with fingerprint {fingerprint} already exists as version {existing_version.version_id}")
                return existing_version.version_id

        # Generate version ID
        if version_name:
            version_id = f"{version_name}_{int(time.time())}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_id = f"v_{timestamp}"

        # Create metadata
        metadata = {
            'fingerprint': fingerprint,
            'description': description,
            'tags': tags or [],
            'preprocessing_steps': preprocessing_steps or [],
            'source_info': source_info or {},
            'document_count': len(documents),
            'created_at': datetime.now().isoformat(),
            'project_name': self.project_name
        }

        # Calculate dataset statistics
        stats = self._calculate_dataset_stats(documents)
        metadata['statistics'] = stats

        # Save documents
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)

        documents_file = version_dir / "documents.json"
        with open(documents_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Create version object
        version = DatasetVersion(version_id, metadata)
        self.versions[version_id] = version

        # Save versions index
        self._save_versions()

        self.logger.info(f"Created dataset version {version_id} with {len(documents)} documents")
        return version_id

    def _calculate_dataset_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic statistics about the dataset."""
        if not documents:
            return {}

        stats = {
            'total_documents': len(documents),
            'fields': {},
            'text_lengths': [],
            'field_coverage': {}
        }

        # Analyze fields and content
        field_counts = {}
        all_fields = set()

        for doc in documents:
            for field in doc.keys():
                all_fields.add(field)
                field_counts[field] = field_counts.get(field, 0) + 1

                # Analyze text fields
                if isinstance(doc[field], str) and field in ['text', 'content', 'body']:
                    stats['text_lengths'].append(len(doc[field]))

        # Calculate field coverage
        for field in all_fields:
            coverage = field_counts.get(field, 0) / len(documents)
            stats['field_coverage'][field] = coverage

        # Text length statistics
        if stats['text_lengths']:
            stats['text_stats'] = {
                'min_length': min(stats['text_lengths']),
                'max_length': max(stats['text_lengths']),
                'avg_length': sum(stats['text_lengths']) / len(stats['text_lengths']),
                'total_chars': sum(stats['text_lengths'])
            }

        stats['unique_fields'] = list(all_fields)
        return stats

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get a specific version."""
        return self.versions.get(version_id)

    def list_versions(self, tags: Optional[List[str]] = None) -> List[DatasetVersion]:
        """List all versions, optionally filtered by tags."""
        versions = list(self.versions.values())

        if tags:
            filtered_versions = []
            for version in versions:
                version_tags = version.metadata.get('tags', [])
                if any(tag in version_tags for tag in tags):
                    filtered_versions.append(version)
            versions = filtered_versions

        # Sort by creation time (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def load_version_documents(self, version_id: str) -> List[Dict[str, Any]]:
        """Load documents from a specific version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        documents_file = self.versions_dir / version_id / "documents.json"
        if not documents_file.exists():
            raise FileNotFoundError(f"Documents file not found for version {version_id}")

        try:
            with open(documents_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            self.logger.info(f"Loaded {len(documents)} documents from version {version_id}")
            return documents

        except Exception as e:
            self.logger.error(f"Error loading documents from version {version_id}: {e}")
            raise

    def delete_version(self, version_id: str, force: bool = False):
        """Delete a dataset version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version_dir = self.versions_dir / version_id

        if not force:
            # Check if this is the only version
            if len(self.versions) == 1:
                raise ValueError("Cannot delete the only version. Use force=True to override.")

        # Delete version directory
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Remove from versions
        del self.versions[version_id]

        # Save updated versions
        self._save_versions()

        self.logger.info(f"Deleted version {version_id}")

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two dataset versions."""
        if version_id1 not in self.versions or version_id2 not in self.versions:
            raise ValueError("One or both versions not found")

        version1 = self.versions[version_id1]
        version2 = self.versions[version_id2]

        comparison = {
            'version1': {
                'id': version_id1,
                'created_at': version1.created_at,
                'document_count': version1.metadata.get('document_count', 0),
                'fingerprint': version1.metadata.get('fingerprint', '')
            },
            'version2': {
                'id': version_id2,
                'created_at': version2.created_at,
                'document_count': version2.metadata.get('document_count', 0),
                'fingerprint': version2.metadata.get('fingerprint', '')
            }
        }

        # Check if versions are identical
        comparison['identical'] = (
            version1.metadata.get('fingerprint') == version2.metadata.get('fingerprint')
        )

        # Compare statistics if available
        stats1 = version1.metadata.get('statistics', {})
        stats2 = version2.metadata.get('statistics', {})

        if stats1 and stats2:
            comparison['statistics_diff'] = {
                'document_count_diff': stats2.get('total_documents', 0) - stats1.get('total_documents', 0),
                'fields_v1': stats1.get('unique_fields', []),
                'fields_v2': stats2.get('unique_fields', []),
                'new_fields': list(set(stats2.get('unique_fields', [])) - set(stats1.get('unique_fields', []))),
                'removed_fields': list(set(stats1.get('unique_fields', [])) - set(stats2.get('unique_fields', [])))
            }

        return comparison

    def create_branch(self, base_version_id: str, branch_name: str) -> str:
        """Create a new branch from an existing version."""
        if base_version_id not in self.versions:
            raise ValueError(f"Base version {base_version_id} not found")

        # Load base documents
        base_documents = self.load_version_documents(base_version_id)
        base_version = self.versions[base_version_id]

        # Create new version with branch metadata
        branch_version_id = self.create_version(
            documents=base_documents,
            version_name=f"{branch_name}_branch",
            description=f"Branch '{branch_name}' created from version {base_version_id}",
            tags=['branch'] + base_version.metadata.get('tags', []),
            preprocessing_steps=base_version.metadata.get('preprocessing_steps', []),
            source_info={
                'type': 'branch',
                'base_version': base_version_id,
                'branch_name': branch_name
            }
        )

        self.logger.info(f"Created branch '{branch_name}' as version {branch_version_id}")
        return branch_version_id

    def export_version(self, version_id: str, export_path: str, format: str = 'json'):
        """Export a version to external format."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        documents = self.load_version_documents(version_id)
        version = self.versions[version_id]

        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == 'json':
            export_data = {
                'version_info': version.to_dict(),
                'documents': documents
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        elif format.lower() == 'jsonl':
            with open(export_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    json.dump(doc, f, ensure_ascii=False)
                    f.write('\n')

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported version {version_id} to {export_path}")

    def get_project_summary(self) -> Dict[str, Any]:
        """Get summary information about the project."""
        if not self.versions:
            return {
                'project_name': self.project_name,
                'total_versions': 0,
                'tracking_dir': str(self.tracking_dir)
            }

        latest_version = max(self.versions.values(), key=lambda v: v.created_at)

        summary = {
            'project_name': self.project_name,
            'total_versions': len(self.versions),
            'latest_version': latest_version.version_id,
            'latest_version_created': latest_version.created_at,
            'tracking_dir': str(self.tracking_dir),
            'total_documents': latest_version.metadata.get('document_count', 0)
        }

        # Collect all tags across versions
        all_tags = set()
        for version in self.versions.values():
            all_tags.update(version.metadata.get('tags', []))

        summary['all_tags'] = list(all_tags)

        # Calculate storage usage
        try:
            total_size = sum(
                sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                for path in [self.versions_dir]
            )
            summary['storage_size_mb'] = round(total_size / (1024 * 1024), 2)
        except Exception:
            summary['storage_size_mb'] = 'unknown'

        return summary

    def cleanup_old_versions(self, keep_latest: int = 5) -> List[str]:
        """Clean up old versions, keeping only the latest N versions."""
        if len(self.versions) <= keep_latest:
            return []

        # Sort versions by creation time
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )

        # Keep the latest versions
        versions_to_keep = sorted_versions[:keep_latest]
        keep_ids = {v.version_id for v in versions_to_keep}

        # Delete old versions
        deleted_versions = []
        for version_id in list(self.versions.keys()):
            if version_id not in keep_ids:
                try:
                    self.delete_version(version_id, force=True)
                    deleted_versions.append(version_id)
                except Exception as e:
                    self.logger.error(f"Error deleting version {version_id}: {e}")

        self.logger.info(f"Cleaned up {len(deleted_versions)} old versions")
        return deleted_versions