import json
import csv
import pandas as pd
import chardet
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Union, List
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging


class FileReader:
    """Handles reading various file formats with automatic encoding detection."""

    def __init__(self, chunk_size: int = 1000, encoding: Optional[str] = None):
        self.chunk_size = chunk_size
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)

    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        if self.encoding:
            return self.encoding

        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Sample first 10KB
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'

    def read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file."""
        encoding = self.detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)

    def read_jsonl(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Read JSONL file line by line."""
        encoding = self.detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue

    def read_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read CSV file with pandas."""
        encoding = self.detect_encoding(file_path)
        return pd.read_csv(file_path, encoding=encoding, **kwargs)

    def read_txt(self, file_path: Path) -> Iterator[str]:
        """Read text file line by line."""
        encoding = self.detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                yield line.strip()

    def read_parquet(self, file_path: Path) -> pd.DataFrame:
        """Read Parquet file."""
        return pd.read_parquet(file_path)

    def read_xml(self, file_path: Path, text_element: str = 'text') -> Iterator[Dict[str, Any]]:
        """Read XML file and extract text elements."""
        encoding = self.detect_encoding(file_path)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            for elem in root.iter():
                if elem.tag == text_element and elem.text:
                    yield {
                        'text': elem.text.strip(),
                        'tag': elem.tag,
                        'attributes': elem.attrib
                    }
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            return

    def read_file(self, file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Auto-detect file format and read accordingly."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower()

        self.logger.info(f"Reading file: {file_path} (format: {file_extension})")

        if file_extension == '.json':
            data = self.read_json(file_path)
            yield {'content': data, 'source': str(file_path), 'format': 'json'}

        elif file_extension == '.jsonl':
            for item in self.read_jsonl(file_path):
                item['source'] = str(file_path)
                item['format'] = 'jsonl'
                yield item

        elif file_extension == '.csv':
            df = self.read_csv(file_path)
            for _, row in df.iterrows():
                yield {
                    'content': row.to_dict(),
                    'source': str(file_path),
                    'format': 'csv'
                }

        elif file_extension == '.txt':
            for line_num, line in enumerate(self.read_txt(file_path), 1):
                if line:
                    yield {
                        'text': line,
                        'source': str(file_path),
                        'format': 'txt',
                        'line_number': line_num
                    }

        elif file_extension == '.parquet':
            df = self.read_parquet(file_path)
            for _, row in df.iterrows():
                yield {
                    'content': row.to_dict(),
                    'source': str(file_path),
                    'format': 'parquet'
                }

        elif file_extension == '.xml':
            for item in self.read_xml(file_path):
                item['source'] = str(file_path)
                item['format'] = 'xml'
                yield item

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def read_directory(self, directory_path: Union[str, Path],
                      pattern: str = "*", recursive: bool = True) -> Iterator[Dict[str, Any]]:
        """Read all files in a directory matching the pattern."""
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if recursive:
            files = directory_path.rglob(pattern)
        else:
            files = directory_path.glob(pattern)

        files = [f for f in files if f.is_file()]

        for file_path in tqdm(files, desc="Processing files"):
            try:
                yield from self.read_file(file_path)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get metadata about a file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()

        return {
            'name': file_path.name,
            'path': str(file_path),
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified_time': stat.st_mtime,
            'extension': file_path.suffix.lower(),
            'encoding': self.detect_encoding(file_path) if file_path.suffix.lower() != '.parquet' else 'binary'
        }