"""Visualization utilities for benchmark results."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn not available, visualization features limited")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("plotly not available, interactive plots disabled")


class ResultVisualizer:
    """Visualize benchmark results with various plot types."""

    def __init__(
        self,
        style: str = "whitegrid",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        save_dir: Optional[Path] = None,
    ):
        """Initialize visualizer.

        Args:
            style: Seaborn style for plots
            figsize: Default figure size
            dpi: DPI for saved figures
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = Path(save_dir) if save_dir else None

        if HAS_MATPLOTLIB:
            sns.set_style(style)
            plt.rcParams["figure.dpi"] = dpi

        # Statistics
        self.stats = {"plots_created": 0, "plots_saved": 0, "plot_types": {}}

        logger.info(
            f"ResultVisualizer initialized (matplotlib: {HAS_MATPLOTLIB}, plotly: {HAS_PLOTLY})"
        )

    def plot_metrics_comparison(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        group_by: str = "model",
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        interactive: bool = False,
    ) -> Optional[Any]:
        """Plot comparison of metrics across models or tasks.

        Args:
            data: DataFrame with results
            metrics: List of metric columns to plot
            group_by: Column to group by
            title: Plot title
            save_path: Path to save plot
            interactive: Use plotly for interactive plot

        Returns:
            Figure object or None
        """
        if interactive and HAS_PLOTLY:
            return self._plot_metrics_comparison_plotly(
                data, metrics, group_by, title, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._plot_metrics_comparison_matplotlib(
                data, metrics, group_by, title, save_path
            )
        else:
            logger.error("No plotting library available")
            return None

    def _plot_metrics_comparison_matplotlib(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        group_by: str,
        title: Optional[str],
        save_path: Optional[Path],
    ) -> Any:
        """Create matplotlib comparison plot.

        Args:
            data: DataFrame with results
            metrics: Metrics to plot
            group_by: Grouping column
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Prepare data
        plot_data = data.groupby(group_by)[metrics].mean()

        # Create bar plot
        plot_data.plot(kind="bar", ax=ax)

        # Customize
        ax.set_xlabel(group_by.capitalize())
        ax.set_ylabel("Score")
        ax.set_title(title or f"Metrics Comparison by {group_by}")
        ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Rotate x labels if many groups
        if len(plot_data) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            self.stats["plots_saved"] += 1
            logger.info(f"Saved plot to {save_path}")

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["metrics_comparison"] = (
            self.stats["plot_types"].get("metrics_comparison", 0) + 1
        )

        return fig

    def _plot_metrics_comparison_plotly(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        group_by: str,
        title: Optional[str],
        save_path: Optional[Path],
    ) -> Any:
        """Create plotly interactive comparison plot.

        Args:
            data: DataFrame with results
            metrics: Metrics to plot
            group_by: Grouping column
            title: Plot title
            save_path: Save path

        Returns:
            Plotly figure object
        """
        # Prepare data
        plot_data = data.groupby(group_by)[metrics].mean().reset_index()

        # Create figure
        fig = go.Figure()

        # Add bars for each metric
        for metric in metrics:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=plot_data[group_by],
                    y=plot_data[metric],
                    text=plot_data[metric].round(3),
                    textposition="auto",
                )
            )

        # Update layout
        fig.update_layout(
            title=title or f"Metrics Comparison by {group_by}",
            xaxis_title=group_by.capitalize(),
            yaxis_title="Score",
            barmode="group",
            hovermode="x unified",
            showlegend=True,
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100,
        )

        # Save if requested
        if save_path:
            fig.write_html(str(save_path))
            self.stats["plots_saved"] += 1
            logger.info(f"Saved interactive plot to {save_path}")

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["metrics_comparison"] = (
            self.stats["plot_types"].get("metrics_comparison", 0) + 1
        )

        return fig

    def plot_performance_over_time(
        self,
        data: pd.DataFrame,
        metric: str,
        time_col: str = "timestamp",
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        interactive: bool = False,
    ) -> Optional[Any]:
        """Plot performance metrics over time.

        Args:
            data: DataFrame with results
            metric: Metric column to plot
            time_col: Time column
            group_by: Optional grouping column
            title: Plot title
            save_path: Path to save plot
            interactive: Use plotly for interactive plot

        Returns:
            Figure object or None
        """
        if interactive and HAS_PLOTLY:
            return self._plot_time_series_plotly(
                data, metric, time_col, group_by, title, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._plot_time_series_matplotlib(
                data, metric, time_col, group_by, title, save_path
            )
        else:
            logger.error("No plotting library available")
            return None

    def _plot_time_series_matplotlib(
        self,
        data: pd.DataFrame,
        metric: str,
        time_col: str,
        group_by: Optional[str],
        title: Optional[str],
        save_path: Optional[Path],
    ) -> Any:
        """Create matplotlib time series plot.

        Args:
            data: DataFrame with results
            metric: Metric to plot
            time_col: Time column
            group_by: Grouping column
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])

        if group_by and group_by in data.columns:
            # Plot each group
            for group in data[group_by].unique():
                group_data = data[data[group_by] == group].sort_values(time_col)
                ax.plot(
                    group_data[time_col], group_data[metric], marker="o", label=group
                )
        else:
            # Single line
            plot_data = data.sort_values(time_col)
            ax.plot(plot_data[time_col], plot_data[metric], marker="o")

        # Customize
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} Over Time")

        if group_by:
            ax.legend(title=group_by.capitalize())

        # Format x-axis
        fig.autofmt_xdate()

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            self.stats["plots_saved"] += 1

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["time_series"] = (
            self.stats["plot_types"].get("time_series", 0) + 1
        )

        return fig

    def _plot_time_series_plotly(
        self,
        data: pd.DataFrame,
        metric: str,
        time_col: str,
        group_by: Optional[str],
        title: Optional[str],
        save_path: Optional[Path],
    ) -> Any:
        """Create plotly time series plot.

        Args:
            data: DataFrame with results
            metric: Metric to plot
            time_col: Time column
            group_by: Grouping column
            title: Plot title
            save_path: Save path

        Returns:
            Plotly figure object
        """
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])

        if group_by and group_by in data.columns:
            fig = px.line(
                data,
                x=time_col,
                y=metric,
                color=group_by,
                markers=True,
                title=title or f"{metric} Over Time",
            )
        else:
            fig = px.line(
                data,
                x=time_col,
                y=metric,
                markers=True,
                title=title or f"{metric} Over Time",
            )

        # Update layout
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=metric,
            hovermode="x unified",
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100,
        )

        # Save if requested
        if save_path:
            fig.write_html(str(save_path))
            self.stats["plots_saved"] += 1

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["time_series"] = (
            self.stats["plot_types"].get("time_series", 0) + 1
        )

        return fig

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None,
        normalize: bool = False,
        interactive: bool = False,
    ) -> Optional[Any]:
        """Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            labels: Class labels
            title: Plot title
            save_path: Path to save plot
            normalize: Whether to normalize values
            interactive: Use plotly for interactive plot

        Returns:
            Figure object or None
        """
        if interactive and HAS_PLOTLY:
            return self._plot_confusion_matrix_plotly(
                confusion_matrix, labels, title, save_path, normalize
            )
        elif HAS_MATPLOTLIB:
            return self._plot_confusion_matrix_matplotlib(
                confusion_matrix, labels, title, save_path, normalize
            )
        else:
            logger.error("No plotting library available")
            return None

    def _plot_confusion_matrix_matplotlib(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]],
        title: str,
        save_path: Optional[Path],
        normalize: bool,
    ) -> Any:
        """Create matplotlib confusion matrix plot.

        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            save_path: Save path
            normalize: Whether to normalize

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            cbar_kws={"label": "Count"},
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            self.stats["plots_saved"] += 1

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["confusion_matrix"] = (
            self.stats["plot_types"].get("confusion_matrix", 0) + 1
        )

        return fig

    def _plot_confusion_matrix_plotly(
        self,
        cm: np.ndarray,
        labels: Optional[List[str]],
        title: str,
        save_path: Optional[Path],
        normalize: bool,
    ) -> Any:
        """Create plotly confusion matrix plot.

        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            save_path: Save path
            normalize: Whether to normalize

        Returns:
            Plotly figure object
        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            colorbar_title = "Proportion"
        else:
            colorbar_title = "Count"

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm.round(2) if normalize else cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar=dict(title=colorbar_title),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100,
        )

        # Save if requested
        if save_path:
            fig.write_html(str(save_path))
            self.stats["plots_saved"] += 1

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["confusion_matrix"] = (
            self.stats["plot_types"].get("confusion_matrix", 0) + 1
        )

        return fig

    def plot_distribution(
        self,
        data: pd.DataFrame,
        column: str,
        group_by: Optional[str] = None,
        plot_type: str = "histogram",
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        interactive: bool = False,
    ) -> Optional[Any]:
        """Plot distribution of values.

        Args:
            data: DataFrame with data
            column: Column to plot
            group_by: Optional grouping column
            plot_type: Type of plot (histogram, box, violin)
            title: Plot title
            save_path: Path to save plot
            interactive: Use plotly for interactive plot

        Returns:
            Figure object or None
        """
        if interactive and HAS_PLOTLY:
            return self._plot_distribution_plotly(
                data, column, group_by, plot_type, title, save_path
            )
        elif HAS_MATPLOTLIB:
            return self._plot_distribution_matplotlib(
                data, column, group_by, plot_type, title, save_path
            )
        else:
            logger.error("No plotting library available")
            return None

    def _plot_distribution_matplotlib(
        self,
        data: pd.DataFrame,
        column: str,
        group_by: Optional[str],
        plot_type: str,
        title: Optional[str],
        save_path: Optional[Path],
    ) -> Any:
        """Create matplotlib distribution plot.

        Args:
            data: DataFrame with data
            column: Column to plot
            group_by: Grouping column
            plot_type: Plot type
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if plot_type == "histogram":
            if group_by:
                for group in data[group_by].unique():
                    group_data = data[data[group_by] == group][column]
                    ax.hist(group_data, alpha=0.5, label=group, bins=20)
                ax.legend()
            else:
                ax.hist(data[column], bins=20, edgecolor="black")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")

        elif plot_type == "box":
            if group_by:
                data.boxplot(column=column, by=group_by, ax=ax)
                ax.set_xlabel(group_by)
            else:
                ax.boxplot(data[column])
                ax.set_xticklabels([column])
            ax.set_ylabel(column)

        elif plot_type == "violin":
            if group_by:
                sns.violinplot(data=data, x=group_by, y=column, ax=ax)
            else:
                sns.violinplot(data=data, y=column, ax=ax)

        ax.set_title(title or f"Distribution of {column}")
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            self.stats["plots_saved"] += 1

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["distribution"] = (
            self.stats["plot_types"].get("distribution", 0) + 1
        )

        return fig

    def _plot_distribution_plotly(
        self,
        data: pd.DataFrame,
        column: str,
        group_by: Optional[str],
        plot_type: str,
        title: Optional[str],
        save_path: Optional[Path],
    ) -> Any:
        """Create plotly distribution plot.

        Args:
            data: DataFrame with data
            column: Column to plot
            group_by: Grouping column
            plot_type: Plot type
            title: Plot title
            save_path: Save path

        Returns:
            Plotly figure object
        """
        if plot_type == "histogram":
            if group_by:
                fig = px.histogram(
                    data,
                    x=column,
                    color=group_by,
                    title=title or f"Distribution of {column}",
                    marginal="box",
                )
            else:
                fig = px.histogram(
                    data,
                    x=column,
                    title=title or f"Distribution of {column}",
                    marginal="box",
                )

        elif plot_type == "box":
            if group_by:
                fig = px.box(
                    data,
                    x=group_by,
                    y=column,
                    title=title or f"Distribution of {column}",
                )
            else:
                fig = px.box(data, y=column, title=title or f"Distribution of {column}")

        elif plot_type == "violin":
            if group_by:
                fig = px.violin(
                    data,
                    x=group_by,
                    y=column,
                    title=title or f"Distribution of {column}",
                )
            else:
                fig = px.violin(
                    data, y=column, title=title or f"Distribution of {column}"
                )

        fig.update_layout(width=self.figsize[0] * 100, height=self.figsize[1] * 100)

        # Save if requested
        if save_path:
            fig.write_html(str(save_path))
            self.stats["plots_saved"] += 1

        self.stats["plots_created"] += 1
        self.stats["plot_types"]["distribution"] = (
            self.stats["plot_types"].get("distribution", 0) + 1
        )

        return fig

    def create_dashboard(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        output_path: Path,
        title: str = "Benchmark Results Dashboard",
    ) -> bool:
        """Create an interactive dashboard with multiple plots.

        Args:
            data: DataFrame with results
            metrics: List of metrics to include
            output_path: Path to save HTML dashboard
            title: Dashboard title

        Returns:
            True if successful
        """
        if not HAS_PLOTLY:
            logger.error("Plotly required for dashboard creation")
            return False

        try:
            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Metrics Comparison",
                    "Performance Over Time",
                    "Score Distribution",
                    "Model Rankings",
                ),
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "box"}, {"type": "bar"}],
                ],
            )

            # 1. Metrics comparison
            if "model" in data.columns:
                model_metrics = data.groupby("model")[metrics].mean()
                for i, metric in enumerate(metrics[:3]):  # Limit to 3 metrics
                    fig.add_trace(
                        go.Bar(
                            name=metric, x=model_metrics.index, y=model_metrics[metric]
                        ),
                        row=1,
                        col=1,
                    )

            # 2. Performance over time (if timestamp available)
            if "timestamp" in data.columns and metrics:
                data_sorted = data.sort_values("timestamp")
                fig.add_trace(
                    go.Scatter(
                        x=data_sorted["timestamp"],
                        y=data_sorted[metrics[0]],
                        mode="lines+markers",
                        name=metrics[0],
                    ),
                    row=1,
                    col=2,
                )

            # 3. Score distribution
            if metrics:
                for metric in metrics[:2]:  # Limit to 2 metrics
                    fig.add_trace(go.Box(y=data[metric], name=metric), row=2, col=1)

            # 4. Model rankings
            if "model" in data.columns and metrics:
                rankings = data.groupby("model")[metrics[0]].mean().sort_values()
                fig.add_trace(
                    go.Bar(
                        x=rankings.values,
                        y=rankings.index,
                        orientation="h",
                        name="Score",
                    ),
                    row=2,
                    col=2,
                )

            # Update layout
            fig.update_layout(title_text=title, showlegend=True, height=800, width=1400)

            # Save dashboard
            fig.write_html(str(output_path))
            logger.info(f"Dashboard saved to {output_path}")

            self.stats["plots_created"] += 1
            self.stats["plots_saved"] += 1
            self.stats["plot_types"]["dashboard"] = (
                self.stats["plot_types"].get("dashboard", 0) + 1
            )

            return True

        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get visualizer statistics.

        Returns:
            Statistics dictionary
        """
        return self.stats.copy()
