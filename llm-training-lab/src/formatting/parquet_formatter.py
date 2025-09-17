import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any, Optional, Union, Iterator
from pathlib import Path
import logging
import json


class ParquetFormatter:
    """Format and export data as Parquet files."""

    def __init__(self, output_path: str, compression: str = 'snappy'):
        self.output_path = Path(output_path)
        self.compression = compression
        self.logger = logging.getLogger(__name__)

        # Validate compression type
        valid_compressions = ['none', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4']
        if compression not in valid_compressions:
            raise ValueError(f"Unsupported compression: {compression}. Valid options: {valid_compressions}")

    def documents_to_dataframe(self, documents: List[Dict[str, Any]],
                              schema_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Convert list of documents to pandas DataFrame."""
        if not documents:
            return pd.DataFrame()

        # Apply schema mapping if provided
        if schema_mapping:
            mapped_documents = []
            for doc in documents:
                mapped_doc = {}
                for output_col, input_field in schema_mapping.items():
                    mapped_doc[output_col] = self._extract_field_value(doc, input_field)
                mapped_documents.append(mapped_doc)
            documents = mapped_documents

        # Convert to DataFrame
        df = pd.DataFrame(documents)

        # Handle complex nested objects by converting to JSON strings
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check if column contains complex objects
                sample_values = df[column].dropna().head(10)
                if any(isinstance(val, (dict, list)) for val in sample_values):
                    df[column] = df[column].apply(
                        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
                    )

        return df

    def _extract_field_value(self, document: Dict[str, Any], field_path: str) -> Any:
        """Extract field value using dot notation for nested fields."""
        if '.' not in field_path:
            return document.get(field_path)

        keys = field_path.split('.')
        value = document

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None

    def write_documents(self, documents: List[Dict[str, Any]],
                       schema_mapping: Optional[Dict[str, str]] = None,
                       row_group_size: int = 10000) -> Dict[str, Any]:
        """Write documents to Parquet file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = self.documents_to_dataframe(documents, schema_mapping)

        if df.empty:
            self.logger.warning("No data to write")
            return {'total_documents': 0, 'written_documents': 0}

        # Optimize data types
        df = self._optimize_datatypes(df)

        stats = {
            'total_documents': len(documents),
            'written_documents': len(df),
            'columns': list(df.columns),
            'file_size_bytes': 0
        }

        try:
            # Write to Parquet
            df.to_parquet(
                self.output_path,
                compression=self.compression,
                index=False,
                engine='pyarrow',
                row_group_size=row_group_size
            )

            # Get file size
            if self.output_path.exists():
                stats['file_size_bytes'] = self.output_path.stat().st_size
                stats['file_size_mb'] = round(stats['file_size_bytes'] / (1024 * 1024), 2)

            self.logger.info(f"Written {len(df)} documents to {self.output_path}")

        except Exception as e:
            self.logger.error(f"Error writing Parquet file: {e}")
            raise

        return stats

    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for Parquet storage."""
        optimized_df = df.copy()

        for column in optimized_df.columns:
            col_data = optimized_df[column]

            # Skip if column is all NaN
            if col_data.isna().all():
                continue

            # Convert object columns that are actually numeric
            if col_data.dtype == 'object':
                # Try to convert to numeric
                try:
                    numeric_col = pd.to_numeric(col_data, errors='coerce')
                    if not numeric_col.isna().all():
                        # Check if it's integers
                        if numeric_col.dropna().apply(lambda x: x.is_integer()).all():
                            optimized_df[column] = numeric_col.astype('Int64')
                        else:
                            optimized_df[column] = numeric_col.astype('float64')
                        continue
                except (ValueError, TypeError):
                    pass

                # Try to convert to datetime
                try:
                    datetime_col = pd.to_datetime(col_data, errors='coerce')
                    if not datetime_col.isna().all():
                        optimized_df[column] = datetime_col
                        continue
                except (ValueError, TypeError):
                    pass

                # Convert to categorical if many repeated values
                if col_data.nunique() / len(col_data) < 0.5:  # Less than 50% unique values
                    optimized_df[column] = col_data.astype('category')

            # Optimize integer types
            elif col_data.dtype in ['int64', 'int32']:
                min_val, max_val = col_data.min(), col_data.max()
                if min_val >= -128 and max_val <= 127:
                    optimized_df[column] = col_data.astype('int8')
                elif min_val >= -32768 and max_val <= 32767:
                    optimized_df[column] = col_data.astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    optimized_df[column] = col_data.astype('int32')

            # Optimize float types
            elif col_data.dtype == 'float64':
                if col_data.min() >= -3.4e38 and col_data.max() <= 3.4e38:
                    optimized_df[column] = col_data.astype('float32')

        return optimized_df

    def read_documents(self, columns: Optional[List[str]] = None,
                      filters: Optional[List[Tuple]] = None,
                      limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Read documents from Parquet file."""
        if not self.output_path.exists():
            raise FileNotFoundError(f"File not found: {self.output_path}")

        try:
            # Read Parquet file
            df = pd.read_parquet(
                self.output_path,
                columns=columns,
                filters=filters,
                engine='pyarrow'
            )

            # Apply limit if specified
            if limit:
                df = df.head(limit)

            # Convert back to dictionaries
            for _, row in df.iterrows():
                document = row.to_dict()

                # Convert NaN to None
                for key, value in document.items():
                    if pd.isna(value):
                        document[key] = None
                    elif isinstance(value, str) and value.startswith(('[', '{')):
                        # Try to parse JSON strings back to objects
                        try:
                            document[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            pass  # Keep as string

                yield document

        except Exception as e:
            self.logger.error(f"Error reading Parquet file: {e}")
            raise

    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information from Parquet file."""
        if not self.output_path.exists():
            raise FileNotFoundError(f"File not found: {self.output_path}")

        try:
            parquet_file = pq.ParquetFile(self.output_path)
            schema = parquet_file.schema

            schema_info = {
                'num_columns': len(schema),
                'num_rows': parquet_file.metadata.num_rows,
                'columns': [],
                'metadata': dict(parquet_file.metadata.metadata) if parquet_file.metadata.metadata else {}
            }

            for i in range(len(schema)):
                field = schema.field(i)
                column_info = {
                    'name': field.name,
                    'type': str(field.type),
                    'nullable': field.nullable
                }
                schema_info['columns'].append(column_info)

            return schema_info

        except Exception as e:
            self.logger.error(f"Error reading schema: {e}")
            raise

    def create_partitioned_dataset(self, documents: List[Dict[str, Any]],
                                  partition_cols: List[str],
                                  schema_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Create partitioned Parquet dataset."""
        # Convert to DataFrame
        df = self.documents_to_dataframe(documents, schema_mapping)

        if df.empty:
            self.logger.warning("No data to write")
            return {'total_documents': 0, 'written_documents': 0}

        # Validate partition columns
        missing_cols = [col for col in partition_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Partition columns not found in data: {missing_cols}")

        # Optimize data types
        df = self._optimize_datatypes(df)

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_documents': len(documents),
            'written_documents': len(df),
            'partition_columns': partition_cols,
            'partitions_created': 0
        }

        try:
            # Write partitioned dataset
            table = pa.Table.from_pandas(df)

            pq.write_to_dataset(
                table,
                root_path=str(self.output_path),
                partition_cols=partition_cols,
                compression=self.compression,
                existing_data_behavior='overwrite_or_ignore'
            )

            # Count partitions created
            partitions = list(self.output_path.rglob('*.parquet'))
            stats['partitions_created'] = len(partitions)

            self.logger.info(f"Created partitioned dataset with {len(partitions)} partitions")

        except Exception as e:
            self.logger.error(f"Error creating partitioned dataset: {e}")
            raise

        return stats

    def merge_parquet_files(self, input_files: List[str],
                           output_path: Optional[str] = None) -> Dict[str, Any]:
        """Merge multiple Parquet files into one."""
        if output_path is None:
            output_path = self.output_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tables = []
        total_rows = 0

        try:
            # Read all files
            for file_path in input_files:
                if not Path(file_path).exists():
                    self.logger.warning(f"File not found: {file_path}")
                    continue

                table = pq.read_table(file_path)
                tables.append(table)
                total_rows += len(table)

            if not tables:
                raise ValueError("No valid input files found")

            # Concatenate tables
            merged_table = pa.concat_tables(tables)

            # Write merged file
            pq.write_table(
                merged_table,
                output_path,
                compression=self.compression
            )

            stats = {
                'input_files': len(input_files),
                'total_rows': total_rows,
                'output_file': str(output_path),
                'file_size_bytes': output_path.stat().st_size
            }

            self.logger.info(f"Merged {len(input_files)} files into {output_path}")
            return stats

        except Exception as e:
            self.logger.error(f"Error merging Parquet files: {e}")
            raise

    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about the Parquet file."""
        if not self.output_path.exists():
            return {'exists': False}

        try:
            parquet_file = pq.ParquetFile(self.output_path)
            metadata = parquet_file.metadata

            stats = {
                'exists': True,
                'path': str(self.output_path),
                'size_bytes': self.output_path.stat().st_size,
                'size_mb': round(self.output_path.stat().st_size / (1024 * 1024), 2),
                'num_rows': metadata.num_rows,
                'num_columns': len(parquet_file.schema),
                'num_row_groups': metadata.num_row_groups,
                'compression': self.compression,
                'created_by': metadata.created_by,
                'schema': self.get_schema_info()
            }

            # Calculate compression ratio if uncompressed size is available
            if hasattr(metadata, 'serialized_size'):
                uncompressed_size = sum(
                    rg.total_byte_size for rg in metadata.row_groups
                )
                if uncompressed_size > 0:
                    stats['compression_ratio'] = stats['size_bytes'] / uncompressed_size

            return stats

        except Exception as e:
            return {
                'exists': True,
                'path': str(self.output_path),
                'error': str(e)
            }

    def validate_parquet_file(self) -> Dict[str, Any]:
        """Validate Parquet file integrity."""
        if not self.output_path.exists():
            return {'valid': False, 'error': 'File does not exist'}

        try:
            # Try to read the file
            parquet_file = pq.ParquetFile(self.output_path)
            metadata = parquet_file.metadata

            validation_result = {
                'valid': True,
                'num_rows': metadata.num_rows,
                'num_row_groups': metadata.num_row_groups,
                'can_read_metadata': True,
                'can_read_data': False,
                'row_groups_valid': 0
            }

            # Try to read data from each row group
            for i in range(metadata.num_row_groups):
                try:
                    parquet_file.read_row_group(i)
                    validation_result['row_groups_valid'] += 1
                except Exception as e:
                    validation_result['row_group_errors'] = validation_result.get('row_group_errors', [])
                    validation_result['row_group_errors'].append({
                        'row_group': i,
                        'error': str(e)
                    })

            # Check if all row groups are valid
            if validation_result['row_groups_valid'] == metadata.num_row_groups:
                validation_result['can_read_data'] = True

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }