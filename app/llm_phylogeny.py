"""Interactive phylogenetic graph of large language models.

This module builds a Bokeh figure that visualises the relationship between
transformer-based language models and the key architectural innovations that link
them together.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
from bokeh.io import output_file, save, show
from bokeh.models import (
    BoxZoomTool,
    HoverTool,
    Label,
    PanTool,
    ResetTool,
    TapTool,
    WheelZoomTool,
)
from bokeh.palettes import Category20
from bokeh.plotting import figure, from_networkx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "llm_models.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "interactive_llm_phylogeny.html"
DEFAULT_SVG_OUTPUT_PATH = REPO_ROOT / "docs" / "interactive_llm_phylogeny.svg"


@dataclass(frozen=True)
class TimelineLayout:
    """Store node coordinates for both interactive and static outputs."""

    families: Tuple[str, ...]
    node_positions_ms: Dict[str, tuple[float, float]]
    node_positions_dt: Dict[str, tuple[dt.datetime, float]]
    x_range_ms: tuple[float, float]
    x_range_dt: tuple[dt.datetime, dt.datetime]


def _parse_date(raw: str) -> dt.datetime:
    """Convert the input "day-month-year" string into a datetime."""
    return dt.datetime.strptime(raw, "%d-%m-%Y")


def _load_models_from_csv(data_path: Path | None = None) -> List[Dict[str, object]]:
    """Load model metadata from the CSV dataset."""

    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    models: List[Dict[str, object]] = []
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "family", "release_month", "innovation"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV file is missing required columns: {missing}")

        for row in reader:
            name = row.get("name", "").strip()
            if not name:
                continue

            raw_influences = (row.get("influences") or "").strip()
            influences = [
                part.strip()
                for part in raw_influences.split(";")
                if part.strip()
            ]

            model = {
                "name": name,
                "family": (row.get("family") or "Unknown").strip() or "Unknown",
                "release_month": (row.get("release_month") or "").strip(),
                "influences": influences,
                "innovation": (row.get("innovation") or "").strip(),
            }
            models.append(model)

    if not models:
        raise ValueError(f"No model records were loaded from {data_path}")

    return models


def _prepare_models(raw_models: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return a normalised and chronologically sorted list of models."""

    combined: List[Dict[str, object]] = []
    for model in raw_models:
        normalised = dict(model)
        release_month = str(normalised.get("release_month", "")).strip()
        if not release_month:
            raise ValueError(f"Model '{normalised.get('name')}' is missing a release month")
        release_date = _parse_date(release_month)
        normalised["release_date"] = release_date
        normalised["release_label"] = release_date.strftime("%b %Y")
        combined.append(normalised)

    combined.sort(key=lambda item: item["release_date"])
    return combined


def _build_graph(models: Iterable[Dict[str, object]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for model in models:
        graph.add_node(model["name"], **model)
    for model in models:
        for influence in model["influences"]:
            if influence in graph:
                graph.add_edge(influence, model["name"], innovation=graph.nodes[model["name"]]["innovation"])
    return graph


def _build_timeline_layout(models: Iterable[Dict[str, object]]) -> TimelineLayout:
    """Return node coordinates for both Bokeh (ms) and Matplotlib (datetime)."""

    families_sorted = tuple(sorted({model["family"] for model in models}))
    family_y = {family: index for index, family in enumerate(families_sorted)}

    family_release_offsets: Dict[str, Dict[dt.date, int]] = {}
    positions_ms: Dict[str, tuple[float, float]] = {}
    positions_dt: Dict[str, tuple[dt.datetime, float]] = {}

    models_list = list(models)
    if not models_list:
        raise ValueError("No models were provided to build the timeline layout")

    for model in models_list:
        name = str(model["name"])
        release_date: dt.datetime = model["release_date"]
        family: str = model["family"]
        family_release_offsets.setdefault(family, {})
        offsets_for_family = family_release_offsets[family]
        base_key = release_date.date()
        offsets_for_family[base_key] = offsets_for_family.get(base_key, -1) + 1
        jitter = (offsets_for_family[base_key] % 3) * 0.15
        x_ms = release_date.timestamp() * 1000.0
        y = family_y[family] + jitter
        positions_ms[name] = (x_ms, y)
        positions_dt[name] = (release_date, y)

    start_dt = models_list[0]["release_date"] - dt.timedelta(days=60)
    end_dt = models_list[-1]["release_date"] + dt.timedelta(days=60)
    start_ts = start_dt.timestamp() * 1000.0
    end_ts = end_dt.timestamp() * 1000.0

    return TimelineLayout(
        families=families_sorted,
        node_positions_ms=positions_ms,
        node_positions_dt=positions_dt,
        x_range_ms=(start_ts, end_ts),
        x_range_dt=(start_dt, end_dt),
    )


def _prepare_visualisation(
    *, data_path: Path | None = None
) -> tuple[List[Dict[str, object]], nx.DiGraph, TimelineLayout, Dict[str, str]]:
    """Load the dataset and compute shared artefacts for all outputs."""

    raw_models = _load_models_from_csv(data_path=data_path)
    models = _prepare_models(raw_models)
    graph = _build_graph(models)
    layout = _build_timeline_layout(models)

    palette = Category20[20]
    color_map = {
        family: palette[index % len(palette)]
        for index, family in enumerate(layout.families)
    }

    return models, graph, layout, color_map


def _construct_bokeh_figure(
    graph: nx.DiGraph, layout: TimelineLayout, color_map: Dict[str, str]
):
    """Create the configured Bokeh figure from prepared components."""

    plot = figure(
        width=1200,
        height=800,
        x_axis_type="datetime",
        x_range=layout.x_range_ms,
        y_range=(-1, len(layout.families) + 1),
        title="Phylogeny of Transformer Language Models",
        toolbar_location="above",
    )
    plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), TapTool())

    graph_renderer = from_networkx(graph, layout.node_positions_ms)

    node_source = graph_renderer.node_renderer.data_source
    node_source.data["family"] = [graph.nodes[name]["family"] for name in graph.nodes]
    node_source.data["release"] = [graph.nodes[name]["release_label"] for name in graph.nodes]
    node_source.data["innovation"] = [graph.nodes[name]["innovation"] for name in graph.nodes]
    node_source.data["color"] = [color_map[graph.nodes[name]["family"]] for name in graph.nodes]

    graph_renderer.node_renderer.glyph.size = 18
    graph_renderer.node_renderer.glyph.fill_color = "color"
    graph_renderer.node_renderer.glyph.line_color = "#222222"

    edge_source = graph_renderer.edge_renderer.data_source
    edge_source.data["innovation"] = [
        graph.nodes[end]["innovation"] for end in edge_source.data["end"]
    ]
    edge_source.data["release"] = [
        graph.nodes[end]["release_label"] for end in edge_source.data["end"]
    ]

    graph_renderer.edge_renderer.glyph.line_alpha = 0.4
    graph_renderer.edge_renderer.glyph.line_width = 2

    plot.renderers.append(graph_renderer)

    node_hover = HoverTool(
        tooltips=[
            ("Model", "@index"),
            ("Family", "@family"),
            ("Released", "@release"),
            ("Key innovation", "@innovation"),
        ],
        renderers=[graph_renderer.node_renderer],
    )
    edge_hover = HoverTool(
        tooltips=[
            ("Influence", "@start → @end"),
            ("Child release", "@release"),
            ("Innovation carried forward", "@innovation"),
        ],
        renderers=[graph_renderer.edge_renderer],
    )
    plot.add_tools(node_hover, edge_hover)

    # Configure y-axis to display family names.
    families_sorted = list(layout.families)
    plot.yaxis.ticker = list(range(len(families_sorted)))
    plot.yaxis.major_label_overrides = {
        index: family for index, family in enumerate(families_sorted)
    }
    plot.xaxis.axis_label = "Release timeline"
    plot.yaxis.axis_label = "Model family"

    # Add a subtitle style label to guide interaction.
    subtitle = Label(
        x=0,
        y=len(families_sorted) + 0.8,
        x_units="screen",
        y_units="data",
        text="Hover nodes or edges to see innovations. Use scroll to zoom.",
        text_font_size="10pt",
    )
    plot.add_layout(subtitle)

    # Add invisible circles for legend entries.
    for family in families_sorted:
        plot.scatter([], [], size=12, color=color_map[family], legend_label=family)
    plot.legend.location = "top_left"
    plot.legend.click_policy = "mute"

    return plot


def build_plot(*, data_path: Path | None = None):
    """Construct the interactive Bokeh plot for the phylogenetic graph."""

    _, graph, layout, color_map = _prepare_visualisation(data_path=data_path)
    return _construct_bokeh_figure(graph, layout, color_map)


def export_static_svg(
    graph: nx.DiGraph,
    layout: TimelineLayout,
    color_map: Dict[str, str],
    destination: Path,
) -> Path:
    """Render a static SVG version of the phylogeny using Matplotlib."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for start, end in graph.edges():
        x0, y0 = layout.node_positions_dt[start]
        x1, y1 = layout.node_positions_dt[end]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#888888",
            alpha=0.45,
            linewidth=1.5,
            zorder=1,
        )

    node_x = []
    node_y = []
    node_colors = []
    for name in graph.nodes:
        x, y = layout.node_positions_dt[name]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(color_map[graph.nodes[name]["family"]])

    ax.scatter(
        node_x,
        node_y,
        s=110,
        c=node_colors,
        edgecolors="#222222",
        linewidths=0.6,
        zorder=2,
    )

    ax.set_xlabel("Release timeline")
    ax.set_ylabel("Model family")
    ax.set_ylim(-1, len(layout.families) + 1)
    ax.set_xlim(layout.x_range_dt)

    ax.set_yticks(range(len(layout.families)))
    ax.set_yticklabels(layout.families)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.grid(True, axis="x", color="#dddddd", linewidth=0.8, alpha=0.6)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)

    subtitle = "Hover in the HTML version to explore innovations"
    ax.set_title(
        "Phylogeny of Transformer Language Models\n" + subtitle,
        fontsize=14,
        pad=18,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[family],
            markeredgecolor="#222222",
            markersize=9,
            linewidth=0,
        )
        for family in layout.families
    ]
    ax.legend(
        legend_handles,
        layout.families,
        title="Model families",
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(destination, format="svg", dpi=150)
    plt.close(fig)
    return destination


def main(
    output_path: Path | None = None,
    data_path: Path | None = None,
    *,
    svg_output_path: Path | None = DEFAULT_SVG_OUTPUT_PATH,
    open_browser: bool = False,
) -> Path:
    """Generate the phylogeny plot and write it to an HTML file."""
    _, graph, layout, color_map = _prepare_visualisation(data_path=data_path)
    plot = _construct_bokeh_figure(graph, layout, color_map)
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    output_file(str(output_path), title="LLM Phylogeny")
    save(plot)
    if svg_output_path is not None:
        export_static_svg(graph, layout, color_map, svg_output_path)
    if open_browser:
        show(plot)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the HTML file that will store the plot.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV file containing model metadata.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated plot in a web browser after saving.",
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=DEFAULT_SVG_OUTPUT_PATH,
        help="Path to write the static SVG snapshot of the phylogeny.",
    )
    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip exporting the SVG figure (requires matplotlib when enabled).",
    )
    args = parser.parse_args()

    destination = main(
        output_path=args.output,
        data_path=args.data,
        svg_output_path=None if args.no_svg else args.svg_output,
        open_browser=args.show,
    )
    print(f"Saved interactive phylogeny to {destination}")
