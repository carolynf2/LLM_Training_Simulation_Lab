from typing import List, Dict, Any, Optional, Tuple, Callable
import re
import json
from pathlib import Path
import logging
from jsonschema import validate, ValidationError
import pandas as pd


class DataValidator:
    """Comprehensive data validation for training datasets."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_document_structure(self, document: Dict[str, Any],
                                  required_fields: List[str],
                                  optional_fields: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """Validate document structure against required and optional fields."""
        errors = []
        optional_fields = optional_fields or []

        # Check required fields
        for field in required_fields:
            if field not in document:
                errors.append(f"Missing required field: {field}")
            elif document[field] is None:
                errors.append(f"Required field is None: {field}")

        # Check for unexpected fields
        expected_fields = set(required_fields + optional_fields)
        actual_fields = set(document.keys())
        unexpected = actual_fields - expected_fields

        if unexpected:
            errors.append(f"Unexpected fields: {list(unexpected)}")

        return len(errors) == 0, errors

    def validate_text_field(self, text: str, field_name: str = "text") -> Tuple[bool, List[str]]:
        """Validate text field content."""
        errors = []

        if not isinstance(text, str):
            errors.append(f"{field_name} must be a string, got {type(text)}")
            return False, errors

        # Check for empty or whitespace-only text
        if not text.strip():
            errors.append(f"{field_name} is empty or contains only whitespace")

        # Check for control characters (excluding common ones like \n, \t)
        control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text)
        if control_chars:
            errors.append(f"{field_name} contains control characters: {set(control_chars)}")

        # Check for encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            errors.append(f"{field_name} has encoding issues: {e}")

        # Check for extremely long lines that might indicate formatting issues
        lines = text.split('\n')
        long_lines = [i for i, line in enumerate(lines) if len(line) > 10000]
        if long_lines:
            errors.append(f"{field_name} has extremely long lines at positions: {long_lines[:5]}")

        return len(errors) == 0, errors

    def validate_metadata_field(self, metadata: Any, field_name: str) -> Tuple[bool, List[str]]:
        """Validate metadata field."""
        errors = []

        if metadata is None:
            return True, []  # Metadata can be None

        # Check if metadata is JSON serializable
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as e:
            errors.append(f"{field_name} is not JSON serializable: {e}")

        # Check for reasonable metadata size
        metadata_str = str(metadata)
        if len(metadata_str) > 100000:  # 100KB limit
            errors.append(f"{field_name} is too large ({len(metadata_str)} chars)")

        return len(errors) == 0, errors

    def validate_document(self, document: Dict[str, Any],
                         schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Validate a single document."""
        errors = []

        # Basic structure validation
        if not isinstance(document, dict):
            return False, ["Document must be a dictionary"]

        # Schema validation if provided
        if schema:
            try:
                validate(instance=document, schema=schema)
            except ValidationError as e:
                errors.append(f"Schema validation failed: {e.message}")

        # Validate text fields
        text_fields = ['text', 'content', 'instruction', 'input', 'output', 'response']
        for field in text_fields:
            if field in document:
                valid, field_errors = self.validate_text_field(document[field], field)
                if not valid:
                    errors.extend(field_errors)

        # Validate metadata fields
        metadata_fields = ['metadata', 'source', 'timestamp', 'labels']
        for field in metadata_fields:
            if field in document:
                valid, field_errors = self.validate_metadata_field(document[field], field)
                if not valid:
                    errors.extend(field_errors)

        return len(errors) == 0, errors

    def validate_dataset(self, documents: List[Dict[str, Any]],
                        schema: Optional[Dict[str, Any]] = None,
                        sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Validate entire dataset."""
        if sample_size and len(documents) > sample_size:
            import random
            sample_docs = random.sample(documents, sample_size)
            self.logger.info(f"Validating random sample of {sample_size} documents")
        else:
            sample_docs = documents

        validation_results = {
            'total_documents': len(documents),
            'validated_documents': len(sample_docs),
            'valid_documents': 0,
            'invalid_documents': 0,
            'validation_errors': [],
            'error_summary': {},
            'field_statistics': {}
        }

        valid_count = 0
        error_types = {}

        for i, doc in enumerate(sample_docs):
            is_valid, errors = self.validate_document(doc, schema)

            if is_valid:
                valid_count += 1
            else:
                validation_results['validation_errors'].append({
                    'document_index': i,
                    'errors': errors
                })

                # Count error types
                for error in errors:
                    error_type = error.split(':')[0] if ':' in error else error
                    error_types[error_type] = error_types.get(error_type, 0) + 1

        validation_results['valid_documents'] = valid_count
        validation_results['invalid_documents'] = len(sample_docs) - valid_count
        validation_results['error_summary'] = error_types

        # Calculate field statistics
        validation_results['field_statistics'] = self._calculate_field_statistics(sample_docs)

        return validation_results

    def _calculate_field_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about fields in the dataset."""
        field_counts = {}
        field_types = {}
        field_lengths = {}

        for doc in documents:
            for field, value in doc.items():
                # Count field presence
                field_counts[field] = field_counts.get(field, 0) + 1

                # Track field types
                value_type = type(value).__name__
                if field not in field_types:
                    field_types[field] = {}
                field_types[field][value_type] = field_types[field].get(value_type, 0) + 1

                # Track field lengths for strings
                if isinstance(value, str):
                    if field not in field_lengths:
                        field_lengths[field] = []
                    field_lengths[field].append(len(value))

        # Calculate length statistics
        length_stats = {}
        for field, lengths in field_lengths.items():
            if lengths:
                length_stats[field] = {
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'avg_length': sum(lengths) / len(lengths),
                    'median_length': sorted(lengths)[len(lengths) // 2]
                }

        return {
            'field_presence': field_counts,
            'field_types': field_types,
            'length_statistics': length_stats,
            'total_documents': len(documents)
        }


class SchemaValidator:
    """JSON Schema validator for dataset schemas."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_schema(self, sample_documents: List[Dict[str, Any]],
                     required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Auto-generate JSON schema from sample documents."""
        if not sample_documents:
            return {}

        required_fields = required_fields or []

        # Analyze field types and patterns
        field_analysis = {}
        for doc in sample_documents:
            for field, value in doc.items():
                if field not in field_analysis:
                    field_analysis[field] = {
                        'types': set(),
                        'examples': [],
                        'null_count': 0
                    }

                if value is None:
                    field_analysis[field]['null_count'] += 1
                else:
                    field_analysis[field]['types'].add(type(value).__name__)
                    if len(field_analysis[field]['examples']) < 5:
                        field_analysis[field]['examples'].append(value)

        # Build schema
        schema = {
            "type": "object",
            "properties": {},
            "required": required_fields
        }

        for field, analysis in field_analysis.items():
            field_schema = self._infer_field_schema(analysis)
            schema["properties"][field] = field_schema

        return schema

    def _infer_field_schema(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Infer schema for a single field."""
        types = analysis['types']
        has_null = analysis['null_count'] > 0

        if len(types) == 1:
            field_type = list(types)[0]
            schema = {"type": self._python_to_json_type(field_type)}
        elif len(types) > 1:
            json_types = [self._python_to_json_type(t) for t in types]
            schema = {"type": json_types}
        else:
            schema = {"type": "null"}

        if has_null and "null" not in schema.get("type", []):
            if isinstance(schema["type"], list):
                schema["type"].append("null")
            else:
                schema["type"] = [schema["type"], "null"]

        # Add additional constraints for strings
        if "string" in schema.get("type", []) or schema.get("type") == "string":
            examples = analysis.get('examples', [])
            string_examples = [ex for ex in examples if isinstance(ex, str)]
            if string_examples:
                lengths = [len(s) for s in string_examples]
                schema["minLength"] = 1  # Assume non-empty strings
                if max(lengths) < 1000:  # Reasonable max length
                    schema["maxLength"] = max(lengths) * 2

        return schema

    def _python_to_json_type(self, python_type: str) -> str:
        """Convert Python type to JSON schema type."""
        type_mapping = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object',
            'NoneType': 'null'
        }
        return type_mapping.get(python_type, 'string')

    def validate_against_schema(self, documents: List[Dict[str, Any]],
                               schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate documents against a schema."""
        results = {
            'total_documents': len(documents),
            'valid_documents': 0,
            'invalid_documents': 0,
            'validation_errors': []
        }

        for i, doc in enumerate(documents):
            try:
                validate(instance=doc, schema=schema)
                results['valid_documents'] += 1
            except ValidationError as e:
                results['invalid_documents'] += 1
                results['validation_errors'].append({
                    'document_index': i,
                    'error': e.message,
                    'path': list(e.path) if e.path else [],
                    'failed_value': e.instance
                })

        return results


class DataQualityValidator:
    """High-level validator combining multiple validation approaches."""

    def __init__(self):
        self.data_validator = DataValidator()
        self.schema_validator = SchemaValidator()
        self.logger = logging.getLogger(__name__)

    def comprehensive_validation(self, documents: List[Dict[str, Any]],
                                schema: Optional[Dict[str, Any]] = None,
                                auto_generate_schema: bool = True,
                                sample_size: Optional[int] = 1000) -> Dict[str, Any]:
        """Perform comprehensive validation of the dataset."""
        self.logger.info(f"Starting comprehensive validation of {len(documents)} documents")

        validation_report = {
            'dataset_info': {
                'total_documents': len(documents),
                'validation_timestamp': pd.Timestamp.now().isoformat()
            },
            'schema_validation': {},
            'data_validation': {},
            'recommendations': []
        }

        # Auto-generate schema if needed
        if not schema and auto_generate_schema:
            sample_docs = documents[:min(100, len(documents))]
            schema = self.schema_validator.create_schema(sample_docs)
            validation_report['generated_schema'] = schema

        # Schema validation
        if schema:
            schema_results = self.schema_validator.validate_against_schema(documents, schema)
            validation_report['schema_validation'] = schema_results

        # Data validation
        data_results = self.data_validator.validate_dataset(documents, schema, sample_size)
        validation_report['data_validation'] = data_results

        # Generate recommendations
        recommendations = self._generate_recommendations(validation_report)
        validation_report['recommendations'] = recommendations

        # Overall quality score
        quality_score = self._calculate_quality_score(validation_report)
        validation_report['overall_quality_score'] = quality_score

        return validation_report

    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check data validation results
        data_validation = validation_report.get('data_validation', {})
        error_summary = data_validation.get('error_summary', {})

        if 'Missing required field' in error_summary:
            recommendations.append("Some documents are missing required fields. Consider data cleaning or relaxing requirements.")

        if 'control characters' in str(error_summary):
            recommendations.append("Text contains control characters. Apply text cleaning preprocessing.")

        if 'encoding issues' in str(error_summary):
            recommendations.append("Encoding issues detected. Check file encoding and apply text normalization.")

        # Check schema validation results
        schema_validation = validation_report.get('schema_validation', {})
        if schema_validation.get('invalid_documents', 0) > 0:
            recommendations.append("Schema validation failures detected. Review data structure consistency.")

        # Check field statistics
        field_stats = data_validation.get('field_statistics', {})
        length_stats = field_stats.get('length_statistics', {})

        for field, stats in length_stats.items():
            if stats.get('max_length', 0) > 100000:
                recommendations.append(f"Field '{field}' has very long texts. Consider chunking or length limits.")

        if not recommendations:
            recommendations.append("Dataset validation passed! Data quality appears good.")

        return recommendations

    def _calculate_quality_score(self, validation_report: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)."""
        score = 1.0

        # Penalize for validation errors
        data_validation = validation_report.get('data_validation', {})
        total_docs = data_validation.get('validated_documents', 1)
        invalid_docs = data_validation.get('invalid_documents', 0)

        if total_docs > 0:
            error_rate = invalid_docs / total_docs
            score -= error_rate * 0.5  # Up to 50% penalty for data errors

        # Penalize for schema validation errors
        schema_validation = validation_report.get('schema_validation', {})
        schema_total = schema_validation.get('total_documents', 1)
        schema_invalid = schema_validation.get('invalid_documents', 0)

        if schema_total > 0:
            schema_error_rate = schema_invalid / schema_total
            score -= schema_error_rate * 0.3  # Up to 30% penalty for schema errors

        return max(0.0, min(1.0, score))

    def export_validation_report(self, validation_report: Dict[str, Any],
                                output_path: str, format: str = 'json'):
        """Export validation report to file."""
        output_path = Path(output_path)

        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
        elif format.lower() == 'html':
            self._export_html_report(validation_report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Validation report exported to {output_path}")

    def _export_html_report(self, validation_report: Dict[str, Any], output_path: Path):
        """Export validation report as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 10px; }}
                .section {{ margin: 20px 0; }}
                .error {{ color: #d32f2f; }}
                .success {{ color: #388e3c; }}
                .warning {{ color: #f57c00; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1 class="header">Dataset Validation Report</h1>

            <div class="section">
                <h2>Dataset Information</h2>
                <p>Total Documents: {validation_report['dataset_info']['total_documents']}</p>
                <p>Validation Date: {validation_report['dataset_info']['validation_timestamp']}</p>
                <p>Overall Quality Score: {validation_report.get('overall_quality_score', 0):.2f}</p>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                {''.join(f'<li>{rec}</li>' for rec in validation_report.get('recommendations', []))}
                </ul>
            </div>

            <div class="section">
                <h2>Validation Details</h2>
                <pre>{json.dumps(validation_report, indent=2, default=str)}</pre>
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)