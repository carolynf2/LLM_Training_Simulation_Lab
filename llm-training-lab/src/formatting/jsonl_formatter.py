import json
import gzip
from typing import List, Dict, Any, Optional, Iterator, TextIO
from pathlib import Path
import logging


class JSONLFormatter:
    """Format and export data as JSONL (JSON Lines)."""

    def __init__(self, output_path: str, compression: Optional[str] = None):
        self.output_path = Path(output_path)
        self.compression = compression.lower() if compression else None
        self.logger = logging.getLogger(__name__)

        # Validate compression type
        if self.compression and self.compression not in ['gzip', 'none']:
            raise ValueError(f"Unsupported compression: {compression}")

    def format_document(self, document: Dict[str, Any], schema: Optional[Dict[str, str]] = None) -> str:
        """Format a single document as a JSON line."""
        if schema:
            formatted_doc = self._apply_schema(document, schema)
        else:
            formatted_doc = document

        return json.dumps(formatted_doc, ensure_ascii=False, separators=(',', ':'))

    def _apply_schema(self, document: Dict[str, Any], schema: Dict[str, str]) -> Dict[str, Any]:
        """Apply schema mapping to document."""
        formatted = {}

        for output_field, input_field in schema.items():
            if input_field in document:
                formatted[output_field] = document[input_field]
            else:
                # Handle nested field access with dot notation
                if '.' in input_field:
                    value = self._get_nested_value(document, input_field)
                    if value is not None:
                        formatted[output_field] = value

        return formatted

    def _get_nested_value(self, document: Dict[str, Any], field_path: str) -> Any:
        """Get nested value using dot notation (e.g., 'metadata.source')."""
        keys = field_path.split('.')
        value = document

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None

    def write_documents(self, documents: List[Dict[str, Any]],
                       schema: Optional[Dict[str, str]] = None,
                       batch_size: int = 1000) -> Dict[str, Any]:
        """Write documents to JSONL file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_documents': len(documents),
            'written_documents': 0,
            'skipped_documents': 0,
            'errors': []
        }

        # Determine file opening mode based on compression
        if self.compression == 'gzip':
            file_opener = lambda: gzip.open(self.output_path, 'wt', encoding='utf-8')
        else:
            file_opener = lambda: open(self.output_path, 'w', encoding='utf-8')

        try:
            with file_opener() as f:
                for i, document in enumerate(documents):
                    try:
                        json_line = self.format_document(document, schema)
                        f.write(json_line + '\n')
                        stats['written_documents'] += 1

                        # Flush periodically
                        if (i + 1) % batch_size == 0:
                            f.flush()
                            self.logger.debug(f"Written {i + 1} documents")

                    except Exception as e:
                        stats['skipped_documents'] += 1
                        stats['errors'].append({
                            'document_index': i,
                            'error': str(e)
                        })
                        self.logger.warning(f"Error formatting document {i}: {e}")

        except Exception as e:
            self.logger.error(f"Error writing to file {self.output_path}: {e}")
            raise

        self.logger.info(f"Written {stats['written_documents']} documents to {self.output_path}")
        return stats

    def read_documents(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Read documents from JSONL file."""
        if not self.output_path.exists():
            raise FileNotFoundError(f"File not found: {self.output_path}")

        # Determine file opening mode based on compression
        if self.compression == 'gzip' or self.output_path.suffix == '.gz':
            file_opener = lambda: gzip.open(self.output_path, 'rt', encoding='utf-8')
        else:
            file_opener = lambda: open(self.output_path, 'r', encoding='utf-8')

        count = 0
        try:
            with file_opener() as f:
                for line_num, line in enumerate(f, 1):
                    if limit and count >= limit:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        document = json.loads(line)
                        yield document
                        count += 1
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Error reading file {self.output_path}: {e}")
            raise

    def append_documents(self, documents: List[Dict[str, Any]],
                        schema: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Append documents to existing JSONL file."""
        # Determine file opening mode
        if self.compression == 'gzip':
            # For gzip, we need to read existing content and rewrite
            existing_docs = list(self.read_documents()) if self.output_path.exists() else []
            all_docs = existing_docs + documents
            return self.write_documents(all_docs, schema)
        else:
            # For regular files, we can append
            stats = {
                'total_documents': len(documents),
                'written_documents': 0,
                'skipped_documents': 0,
                'errors': []
            }

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    for i, document in enumerate(documents):
                        try:
                            json_line = self.format_document(document, schema)
                            f.write(json_line + '\n')
                            stats['written_documents'] += 1

                        except Exception as e:
                            stats['skipped_documents'] += 1
                            stats['errors'].append({
                                'document_index': i,
                                'error': str(e)
                            })
                            self.logger.warning(f"Error formatting document {i}: {e}")

            except Exception as e:
                self.logger.error(f"Error appending to file {self.output_path}: {e}")
                raise

            self.logger.info(f"Appended {stats['written_documents']} documents to {self.output_path}")
            return stats

    def split_into_shards(self, documents: List[Dict[str, Any]],
                         shard_size: int = 10000,
                         schema: Optional[Dict[str, str]] = None) -> List[str]:
        """Split documents into multiple JSONL shards."""
        if shard_size <= 0:
            raise ValueError("Shard size must be positive")

        shard_paths = []
        base_path = self.output_path.parent / self.output_path.stem
        extension = self.output_path.suffix

        for i in range(0, len(documents), shard_size):
            shard_documents = documents[i:i + shard_size]
            shard_num = i // shard_size

            shard_path = f"{base_path}_shard_{shard_num:04d}{extension}"
            shard_formatter = JSONLFormatter(shard_path, self.compression)

            shard_formatter.write_documents(shard_documents, schema)
            shard_paths.append(shard_path)

            self.logger.info(f"Created shard {shard_num}: {len(shard_documents)} documents")

        return shard_paths

    def validate_jsonl_file(self) -> Dict[str, Any]:
        """Validate JSONL file format and content."""
        if not self.output_path.exists():
            return {'valid': False, 'error': 'File does not exist'}

        validation_result = {
            'valid': True,
            'total_lines': 0,
            'valid_json_lines': 0,
            'empty_lines': 0,
            'invalid_lines': 0,
            'errors': []
        }

        try:
            for line_num, line in enumerate(self._read_lines(), 1):
                validation_result['total_lines'] += 1

                if not line.strip():
                    validation_result['empty_lines'] += 1
                    continue

                try:
                    json.loads(line)
                    validation_result['valid_json_lines'] += 1
                except json.JSONDecodeError as e:
                    validation_result['invalid_lines'] += 1
                    validation_result['errors'].append({
                        'line': line_num,
                        'error': str(e)
                    })

        except Exception as e:
            validation_result['valid'] = False
            validation_result['file_error'] = str(e)

        # Overall validity check
        if validation_result['invalid_lines'] > 0:
            validation_result['valid'] = False

        return validation_result

    def _read_lines(self) -> Iterator[str]:
        """Read lines from file handling compression."""
        if self.compression == 'gzip' or self.output_path.suffix == '.gz':
            file_opener = lambda: gzip.open(self.output_path, 'rt', encoding='utf-8')
        else:
            file_opener = lambda: open(self.output_path, 'r', encoding='utf-8')

        with file_opener() as f:
            for line in f:
                yield line

    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about the JSONL file."""
        if not self.output_path.exists():
            return {'exists': False}

        stats = {
            'exists': True,
            'path': str(self.output_path),
            'size_bytes': self.output_path.stat().st_size,
            'compression': self.compression,
            'document_count': 0,
            'sample_document': None
        }

        # Count documents and get sample
        try:
            for i, doc in enumerate(self.read_documents(limit=1)):
                if i == 0:
                    stats['sample_document'] = doc
                break

            # Count all documents
            stats['document_count'] = sum(1 for _ in self.read_documents())

        except Exception as e:
            stats['error'] = str(e)

        return stats


class BatchJSONLWriter:
    """Efficient batch writer for large datasets."""

    def __init__(self, output_path: str, batch_size: int = 10000,
                 compression: Optional[str] = None):
        self.formatter = JSONLFormatter(output_path, compression)
        self.batch_size = batch_size
        self.batch_buffer = []
        self.total_written = 0
        self.logger = logging.getLogger(__name__)

    def add_document(self, document: Dict[str, Any]):
        """Add document to batch buffer."""
        self.batch_buffer.append(document)

        if len(self.batch_buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """Write current batch to file."""
        if not self.batch_buffer:
            return

        if self.total_written == 0:
            # First batch - create new file
            stats = self.formatter.write_documents(self.batch_buffer)
        else:
            # Subsequent batches - append
            stats = self.formatter.append_documents(self.batch_buffer)

        self.total_written += len(self.batch_buffer)
        self.logger.info(f"Flushed {len(self.batch_buffer)} documents (total: {self.total_written})")

        self.batch_buffer.clear()

    def close(self):
        """Flush remaining documents and close writer."""
        self.flush()
        self.logger.info(f"Batch writer closed. Total documents written: {self.total_written}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()