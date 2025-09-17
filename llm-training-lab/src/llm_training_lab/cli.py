#!/usr/bin/env python3
"""
Command-line interface for LLM Training Lab.
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional, List

from .pipeline import DataPipeline
from .config import Config
from ..utils.logger import setup_logging
from ..versioning.dataset_tracker import DatasetTracker
from ..quality.validators import DataQualityValidator


@click.group()
@click.version_option(version="1.0.0")
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """LLM Training Lab - A comprehensive toolkit for LLM dataset preparation."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose

    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging({'log_level': log_level})


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True, help='Configuration file')
@click.option('--input', '-i', multiple=True, help='Input file paths or patterns')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--resume/--no-resume', default=True, help='Resume from checkpoints')
@click.option('--validate/--no-validate', default=True, help='Validate output')
@click.pass_context
def process(ctx, config, input, output, resume, validate):
    """Process dataset with the full pipeline."""
    try:
        click.echo("🚀 Starting LLM Training Lab pipeline...")

        # Initialize pipeline
        pipeline = DataPipeline(config)

        # Run pipeline
        with pipeline:
            result = pipeline.run(
                input_paths=list(input) if input else None,
                output_path=output,
                validate=validate,
                resume=resume
            )

        # Display results
        click.echo("✅ Pipeline completed successfully!")
        click.echo(f"📊 Processed {result.get('format_stats', {}).get('total_documents', 0)} documents")
        click.echo(f"📁 Output paths: {result.get('output_paths', {})}")

        if result.get('dataset_version'):
            click.echo(f"🏷️  Dataset version: {result['dataset_version']}")

    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--remove-html/--keep-html', default=True, help='Remove HTML tags')
@click.option('--min-length', type=int, default=10, help='Minimum text length')
@click.option('--max-length', type=int, help='Maximum text length')
@click.option('--languages', multiple=True, default=['en'], help='Target languages')
def clean(input_file, output_file, remove_html, min_length, max_length, languages):
    """Clean text data with basic preprocessing."""
    try:
        from ..preprocessing.text_cleaner import TextCleaner, LanguageDetector
        from ..ingestion.file_reader import FileReader
        from ..formatting.jsonl_formatter import JSONLFormatter

        click.echo(f"🧹 Cleaning {input_file}...")

        # Setup components
        file_reader = FileReader()
        text_cleaner = TextCleaner(
            remove_html=remove_html,
            min_length=min_length,
            max_length=max_length
        )
        language_detector = LanguageDetector(list(languages))

        # Read documents
        documents = list(file_reader.read_file(input_file))
        click.echo(f"📖 Read {len(documents)} documents")

        # Clean documents
        cleaned_documents = []
        for doc in documents:
            if 'text' in doc:
                cleaned_text = text_cleaner.clean_text(doc['text'])
                if cleaned_text and language_detector.filter_by_language(cleaned_text):
                    doc = doc.copy()
                    doc['text'] = cleaned_text
                    cleaned_documents.append(doc)

        # Write cleaned documents
        formatter = JSONLFormatter(output_file)
        formatter.write_documents(cleaned_documents)

        click.echo(f"✅ Cleaned {len(documents)} -> {len(cleaned_documents)} documents")
        click.echo(f"💾 Saved to {output_file}")

    except Exception as e:
        click.echo(f"❌ Cleaning failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='quality_report.html', help='Output report path')
@click.option('--sample-size', type=int, default=1000, help='Sample size for validation')
def assess(input_path, output, sample_size):
    """Assess data quality and generate report."""
    try:
        from ..ingestion.file_reader import FileReader
        from ..quality.validators import DataQualityValidator

        click.echo(f"📊 Assessing quality of {input_path}...")

        # Read documents
        file_reader = FileReader()
        if Path(input_path).is_file():
            documents = list(file_reader.read_file(input_path))
        else:
            documents = list(file_reader.read_directory(input_path))

        click.echo(f"📖 Analyzing {len(documents)} documents")

        # Run quality assessment
        validator = DataQualityValidator()
        validation_report = validator.comprehensive_validation(
            documents,
            sample_size=sample_size
        )

        # Export report
        validator.export_validation_report(validation_report, output, 'html')

        click.echo(f"✅ Quality assessment complete")
        click.echo(f"📈 Overall quality score: {validation_report.get('overall_quality_score', 0):.2f}")
        click.echo(f"📄 Report saved to {output}")

    except Exception as e:
        click.echo(f"❌ Assessment failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--technique', type=click.Choice(['paraphrase', 'back_translation', 'synthetic']),
              default='paraphrase', help='Augmentation technique')
@click.option('--ratio', type=float, default=0.5, help='Augmentation ratio')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def augment(input_file, technique, ratio, output):
    """Generate augmented data using specified technique."""
    try:
        from ..ingestion.file_reader import FileReader
        from ..formatting.jsonl_formatter import JSONLFormatter

        click.echo(f"🔄 Augmenting {input_file} with {technique}...")

        # Read documents
        file_reader = FileReader()
        documents = list(file_reader.read_file(input_file))
        click.echo(f"📖 Read {len(documents)} documents")

        # Apply augmentation
        if technique == 'paraphrase':
            from ..augmentation.paraphraser import CompositeParaphraser
            augmenter = CompositeParaphraser()
            augmented_docs = augmenter.generate_paraphrase_dataset(
                documents, paraphrases_per_doc=int(ratio * 2)
            )

        elif technique == 'back_translation':
            from ..augmentation.back_translation import BackTranslationAugmenter
            augmenter = BackTranslationAugmenter()
            augmented_docs = augmenter.augment_dataset(
                documents, augmentation_ratio=ratio
            )

        elif technique == 'synthetic':
            from ..augmentation.synthetic_generator import SyntheticDataGenerator
            generator = SyntheticDataGenerator()
            synthetic_count = int(len(documents) * ratio)
            synthetic_docs = generator.generate_mixed_dataset(synthetic_count)
            augmented_docs = documents + synthetic_docs

        # Determine output path
        if not output:
            input_path = Path(input_file)
            output = input_path.parent / f"{input_path.stem}_{technique}_augmented.jsonl"

        # Write augmented documents
        formatter = JSONLFormatter(output)
        formatter.write_documents(augmented_docs)

        click.echo(f"✅ Augmented {len(documents)} -> {len(augmented_docs)} documents")
        click.echo(f"💾 Saved to {output}")

    except Exception as e:
        click.echo(f"❌ Augmentation failed: {e}", err=True)
        sys.exit(1)


@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command()
@click.argument('output_path', type=click.Path())
@click.option('--format', type=click.Choice(['yaml', 'json']), default='yaml', help='Config format')
def create(output_path, format):
    """Create a default configuration file."""
    try:
        from ..utils.config import create_default_config

        create_default_config(output_path, format)
        click.echo(f"✅ Created default configuration: {output_path}")

    except Exception as e:
        click.echo(f"❌ Failed to create config: {e}", err=True)
        sys.exit(1)


@config.command()
@click.argument('config_path', type=click.Path(exists=True))
def validate(config_path):
    """Validate a configuration file."""
    try:
        config_manager = Config(config_path)
        issues = config_manager.validate_config()

        if issues:
            click.echo("❌ Configuration validation failed:")
            for issue in issues:
                click.echo(f"  • {issue}")
            sys.exit(1)
        else:
            click.echo("✅ Configuration is valid")

    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command()
@click.argument('project_name')
@click.option('--tracking-dir', default='./tracking', help='Tracking directory')
def list_versions(project_name, tracking_dir):
    """List dataset versions for a project."""
    try:
        tracker = DatasetTracker(project_name, tracking_dir)
        versions = tracker.list_versions()

        if not versions:
            click.echo(f"No versions found for project: {project_name}")
            return

        click.echo(f"📦 Dataset versions for {project_name}:")
        for version in versions:
            click.echo(f"  • {version.version_id} - {version.created_at}")
            click.echo(f"    Documents: {version.metadata.get('document_count', 0)}")
            click.echo(f"    Description: {version.metadata.get('description', '')}")

    except Exception as e:
        click.echo(f"❌ Failed to list versions: {e}", err=True)
        sys.exit(1)


@dataset.command()
@click.argument('project_name')
@click.argument('version_id')
@click.argument('export_path', type=click.Path())
@click.option('--format', type=click.Choice(['json', 'jsonl']), default='jsonl', help='Export format')
@click.option('--tracking-dir', default='./tracking', help='Tracking directory')
def export_version(project_name, version_id, export_path, format, tracking_dir):
    """Export a specific dataset version."""
    try:
        tracker = DatasetTracker(project_name, tracking_dir)
        tracker.export_version(version_id, export_path, format)
        click.echo(f"✅ Exported version {version_id} to {export_path}")

    except Exception as e:
        click.echo(f"❌ Export failed: {e}", err=True)
        sys.exit(1)


@dataset.command()
@click.argument('project_name')
@click.argument('version_id1')
@click.argument('version_id2')
@click.option('--tracking-dir', default='./tracking', help='Tracking directory')
def compare(project_name, version_id1, version_id2, tracking_dir):
    """Compare two dataset versions."""
    try:
        tracker = DatasetTracker(project_name, tracking_dir)
        comparison = tracker.compare_versions(version_id1, version_id2)

        click.echo(f"📊 Comparing versions {version_id1} vs {version_id2}:")
        click.echo(f"  Version 1: {comparison['version1']['document_count']} documents")
        click.echo(f"  Version 2: {comparison['version2']['document_count']} documents")
        click.echo(f"  Identical: {comparison['identical']}")

        if 'statistics_diff' in comparison:
            diff = comparison['statistics_diff']
            click.echo(f"  Document count difference: {diff['document_count_diff']}")
            if diff['new_fields']:
                click.echo(f"  New fields: {diff['new_fields']}")
            if diff['removed_fields']:
                click.echo(f"  Removed fields: {diff['removed_fields']}")

    except Exception as e:
        click.echo(f"❌ Comparison failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='localhost', help='Host to run server on')
@click.option('--port', default=8050, help='Port to run server on')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def dashboard(host, port, debug):
    """Launch monitoring dashboard (requires Dash)."""
    try:
        import dash
        from dash import html, dcc, Input, Output
        import plotly.graph_objs as go
        import pandas as pd

        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H1("LLM Training Lab Dashboard"),
            html.Div(id="dashboard-content"),
            dcc.Interval(id="interval", interval=5000, n_intervals=0)
        ])

        @app.callback(Output("dashboard-content", "children"), Input("interval", "n_intervals"))
        def update_dashboard(n):
            return html.Div([
                html.H2("System Status"),
                html.P("Dashboard functionality requires implementation of specific monitoring endpoints."),
                html.P("This is a placeholder for the monitoring dashboard.")
            ])

        click.echo(f"🚀 Starting dashboard at http://{host}:{port}")
        app.run_server(host=host, port=port, debug=debug)

    except ImportError:
        click.echo("❌ Dashboard requires 'dash' and 'plotly' packages", err=True)
        click.echo("Install with: pip install dash plotly")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Dashboard failed: {e}", err=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()