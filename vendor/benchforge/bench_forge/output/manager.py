"""Output management for benchmark results."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

from bench_forge.engine.inference import InferenceResult


logger = logging.getLogger(__name__)


class OutputManager:
    """Manage output paths and file formats for benchmark results."""

    def __init__(
        self,
        base_dir: Union[str, Path] = "results",
        create_dirs: bool = True,
        timestamp_format: str = "%Y%m%d_%H%M%S",
        organize_by_date: bool = True,
    ):
        """Initialize output manager.

        Args:
            base_dir: Base directory for outputs
            create_dirs: Whether to create directories if they don't exist
            timestamp_format: Format for timestamps in filenames
            organize_by_date: Whether to organize outputs by date
        """
        self.base_dir = Path(base_dir)
        self.create_dirs = create_dirs
        self.timestamp_format = timestamp_format
        self.organize_by_date = organize_by_date

        # Statistics
        self.stats = {
            "files_written": 0,
            "total_bytes": 0,
            "formats_used": {},
            "last_write": None,
        }

        # Ensure base directory exists
        if create_dirs:
            self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"OutputManager initialized with base_dir: {self.base_dir}")

    def generate_path(
        self,
        task: str,
        model: str,
        format: str = "json",
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        timestamp: bool = True,
        custom_dir: Optional[str] = None,
    ) -> Path:
        """Generate output path for results.

        Args:
            task: Task name
            model: Model name
            format: File format (json, csv, parquet)
            prefix: Optional prefix for filename
            suffix: Optional suffix for filename
            timestamp: Whether to include timestamp
            custom_dir: Custom subdirectory

        Returns:
            Generated path
        """
        # Build directory structure
        if custom_dir:
            dir_path = self.base_dir / custom_dir
        elif self.organize_by_date:
            date_str = datetime.now().strftime("%Y-%m-%d")
            dir_path = self.base_dir / task / date_str
        else:
            dir_path = self.base_dir / task

        # Create directory if needed
        if self.create_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Build filename
        parts = []

        if prefix:
            parts.append(prefix)

        parts.append(task)

        # Clean model name for filename
        model_clean = model.replace("/", "_").replace(":", "_")
        parts.append(model_clean)

        if suffix:
            parts.append(suffix)

        if timestamp:
            ts = datetime.now().strftime(self.timestamp_format)
            parts.append(ts)

        filename = "_".join(parts) + f".{format}"

        return dir_path / filename

    def save_results(
        self,
        results: Union[InferenceResult, List[Dict[str, Any]], pd.DataFrame],
        path: Optional[Path] = None,
        format: str = "json",
        **kwargs,
    ) -> Path:
        """Save results to file.

        Args:
            results: Results to save
            path: Output path (generated if not provided)
            format: Output format
            **kwargs: Additional parameters for path generation

        Returns:
            Path where results were saved
        """
        # Convert InferenceResult to dictionary
        if isinstance(results, InferenceResult):
            data = results.to_dict()
        elif isinstance(results, list):
            data = results
        elif isinstance(results, pd.DataFrame):
            data = results
        else:
            data = results

        # Generate path if not provided
        if path is None:
            if "task" not in kwargs or "model" not in kwargs:
                raise ValueError("Must provide 'task' and 'model' for path generation")
            path = self.generate_path(format=format, **kwargs)
        else:
            path = Path(path)
            # Ensure parent directory exists
            if self.create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        if format == "json":
            self._save_json(data, path)
        elif format == "csv":
            self._save_csv(data, path)
        elif format == "parquet":
            self._save_parquet(data, path)
        elif format == "jsonl":
            self._save_jsonl(data, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Update statistics
        self.stats["files_written"] += 1
        self.stats["total_bytes"] += path.stat().st_size
        self.stats["formats_used"][format] = (
            self.stats["formats_used"].get(format, 0) + 1
        )
        self.stats["last_write"] = datetime.now()

        logger.info(f"Saved results to {path} ({path.stat().st_size:,} bytes)")

        return path

    def _save_json(self, data: Any, path: Path):
        """Save data as JSON.

        Args:
            data: Data to save
            path: Output path
        """
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, pd.DataFrame):
                data.to_json(f, orient="records", indent=2)
            else:
                json.dump(data, f, indent=2, default=str)

    def _save_jsonl(self, data: Any, path: Path):
        """Save data as JSON Lines.

        Args:
            data: Data to save
            path: Output path
        """
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, pd.DataFrame):
                for _, row in data.iterrows():
                    f.write(json.dumps(row.to_dict(), default=str) + "\n")
            elif isinstance(data, list):
                for item in data:
                    f.write(json.dumps(item, default=str) + "\n")
            else:
                f.write(json.dumps(data, default=str) + "\n")

    def _save_csv(self, data: Any, path: Path):
        """Save data as CSV.

        Args:
            data: Data to save
            path: Output path
        """
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Handle nested dict from InferenceResult
            if "results" in data and isinstance(data["results"], list):
                df = pd.DataFrame(data["results"])
            else:
                df = pd.DataFrame([data])
        else:
            raise TypeError(f"Cannot save {type(data)} as CSV")

        df.to_csv(path, index=False)

    def _save_parquet(self, data: Any, path: Path):
        """Save data as Parquet.

        Args:
            data: Data to save
            path: Output path
        """
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            raise ImportError("pyarrow required for parquet format")

        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "results" in data and isinstance(data["results"], list):
                df = pd.DataFrame(data["results"])
            else:
                df = pd.DataFrame([data])
        else:
            raise TypeError(f"Cannot save {type(data)} as Parquet")

        df.to_parquet(path, index=False)

    def load_results(
        self, path: Union[str, Path], format: Optional[str] = None
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """Load results from file.

        Args:
            path: File path
            format: File format (inferred from extension if not provided)

        Returns:
            Loaded data
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Infer format from extension
        if format is None:
            format = path.suffix[1:]  # Remove the dot

        if format == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif format == "jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        elif format == "csv":
            return pd.read_csv(path)
        elif format == "parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def list_results(
        self,
        task: Optional[str] = None,
        model: Optional[str] = None,
        date: Optional[str] = None,
        format: Optional[str] = None,
    ) -> List[Path]:
        """List result files matching criteria.

        Args:
            task: Filter by task name
            model: Filter by model name
            date: Filter by date
            format: Filter by file format

        Returns:
            List of matching file paths
        """
        results = []

        # Build search pattern
        pattern = "*"
        if format:
            pattern = f"*.{format}"

        # Search in appropriate directories
        if task and self.organize_by_date and date:
            search_dir = self.base_dir / task / date
            if search_dir.exists():
                results.extend(search_dir.glob(pattern))
        elif task:
            task_dir = self.base_dir / task
            if task_dir.exists():
                results.extend(task_dir.rglob(pattern))
        else:
            results.extend(self.base_dir.rglob(pattern))

        # Filter by model if specified
        if model:
            model_clean = model.replace("/", "_").replace(":", "_")
            results = [p for p in results if model_clean in p.name]

        return sorted(results)

    def get_latest(
        self, task: str, model: Optional[str] = None, format: str = "json"
    ) -> Optional[Path]:
        """Get most recent result file.

        Args:
            task: Task name
            model: Model name (optional)
            format: File format

        Returns:
            Path to latest file or None
        """
        files = self.list_results(task=task, model=model, format=format)

        if not files:
            return None

        # Sort by modification time
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return files[0]

    def create_summary(
        self,
        task: str,
        output_path: Optional[Path] = None,
        include_metrics: bool = True,
    ) -> pd.DataFrame:
        """Create summary of all results for a task.

        Args:
            task: Task name
            output_path: Optional path to save summary
            include_metrics: Whether to include metrics in summary

        Returns:
            Summary DataFrame
        """
        files = self.list_results(task=task)

        if not files:
            logger.warning(f"No results found for task: {task}")
            return pd.DataFrame()

        summaries = []

        for file_path in files:
            try:
                # Load file
                data = self.load_results(file_path)

                # Extract summary info
                summary = {
                    "file": file_path.name,
                    "date": datetime.fromtimestamp(file_path.stat().st_mtime),
                    "size_bytes": file_path.stat().st_size,
                }

                # Add metrics if available
                if isinstance(data, dict):
                    if "config" in data:
                        summary["model"] = data["config"].get("model", "unknown")
                    if "statistics" in data:
                        summary.update(data["statistics"])
                    if include_metrics and "metrics" in data:
                        summary.update(data["metrics"])

                summaries.append(summary)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

        df = pd.DataFrame(summaries)

        # Save summary if requested
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved summary to {output_path}")

        return df

    def cleanup_old_results(
        self, days: int = 30, keep_latest_n: int = 5, dry_run: bool = True
    ) -> List[Path]:
        """Clean up old result files.

        Args:
            days: Remove files older than this many days
            keep_latest_n: Always keep this many latest files per task
            dry_run: If True, only show what would be deleted

        Returns:
            List of deleted (or would-be deleted) files
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        to_delete = []

        # Process each task
        for task_dir in self.base_dir.iterdir():
            if not task_dir.is_dir():
                continue

            # Get all files for this task
            files = list(task_dir.rglob("*.*"))

            if len(files) <= keep_latest_n:
                continue

            # Sort by modification time
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Keep the latest N files
            candidates = files[keep_latest_n:]

            # Check age of candidates
            for file_path in candidates:
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mod_time < cutoff_date:
                    to_delete.append(file_path)

        # Delete or report
        if dry_run:
            logger.info(f"Would delete {len(to_delete)} files (dry run)")
            for path in to_delete:
                logger.debug(f"  Would delete: {path}")
        else:
            for path in to_delete:
                try:
                    path.unlink()
                    logger.info(f"Deleted: {path}")
                except Exception as e:
                    logger.error(f"Failed to delete {path}: {e}")

        return to_delete

    def generate_report(
        self, task: str, model: Optional[str] = None, output_format: str = "html"
    ) -> str:
        """Generate a report for task results.

        Args:
            task: Task name
            model: Model name (optional)
            output_format: Report format (html, markdown)

        Returns:
            Report content
        """
        files = self.list_results(task=task, model=model)

        if not files:
            return f"No results found for task: {task}"

        if output_format == "html":
            return self._generate_html_report(task, files)
        elif output_format == "markdown":
            return self._generate_markdown_report(task, files)
        else:
            raise ValueError(f"Unsupported report format: {output_format}")

    def _generate_html_report(self, task: str, files: List[Path]) -> str:
        """Generate HTML report.

        Args:
            task: Task name
            files: List of result files

        Returns:
            HTML content
        """
        html = f"""
        <html>
        <head>
            <title>{task} Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>{task} Results Report</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Total files: {len(files)}</p>
            
            <table>
                <tr>
                    <th>File</th>
                    <th>Date</th>
                    <th>Size</th>
                    <th>Path</th>
                </tr>
        """

        for file_path in sorted(files, reverse=True):
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            size = file_path.stat().st_size

            html += f"""
                <tr>
                    <td>{file_path.name}</td>
                    <td>{mod_time.strftime("%Y-%m-%d %H:%M")}</td>
                    <td>{size:,} bytes</td>
                    <td>{file_path}</td>
                </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def _generate_markdown_report(self, task: str, files: List[Path]) -> str:
        """Generate Markdown report.

        Args:
            task: Task name
            files: List of result files

        Returns:
            Markdown content
        """
        lines = [
            f"# {task} Results Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total files:** {len(files)}",
            "",
            "| File | Date | Size | Path |",
            "|------|------|------|------|",
        ]

        for file_path in sorted(files, reverse=True):
            mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            size = file_path.stat().st_size

            lines.append(
                f"| {file_path.name} | {mod_time.strftime('%Y-%m-%d %H:%M')} | "
                f"{size:,} bytes | `{file_path}` |"
            )

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get output manager statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        # Add directory stats
        if self.base_dir.exists():
            all_files = list(self.base_dir.rglob("*.*"))
            stats["total_files"] = len(all_files)
            stats["total_size_mb"] = sum(f.stat().st_size for f in all_files) / (
                1024 * 1024
            )

            # Count by format
            format_counts = {}
            for f in all_files:
                ext = f.suffix[1:] if f.suffix else "unknown"
                format_counts[ext] = format_counts.get(ext, 0) + 1
            stats["files_by_format"] = format_counts

        return stats
