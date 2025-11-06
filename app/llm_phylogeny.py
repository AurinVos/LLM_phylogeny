"""Interactive phylogenetic graph of large language models.

This module builds a Bokeh figure that visualises the relationship between
transformer-based language models and the key architectural innovations that link
them together.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
from bokeh.io import output_file, save, show
from bokeh.models import (
    BoxZoomTool,
    HoverTool,
    Label,
    Legend,
    LegendItem,
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
DEFAULT_TITLE = "Phylogeny of Transformer Language Models"

TOOLTIP_BOX_STYLE = (
    "width: max-content; "
    "max-width: min(360px, calc(100vw - 32px)); "
    "white-space: normal; "
    "overflow-wrap: anywhere; "
    "word-break: break-word; "
    "line-height: 1.25; "
    "box-sizing: border-box;"
)

NODE_TOOLTIP_TEMPLATE = f"""
<div style=\"{TOOLTIP_BOX_STYLE}\">
  <div style=\"font-weight: 600; margin-bottom: 4px;\">@index</div>
  <div><span style=\"font-weight: 600;\">Brand:</span> @brand_label</div>
  <div><span style=\"font-weight: 600;\">Family:</span> @family</div>
  <div><span style=\"font-weight: 600;\">Released:</span> @release</div>
  <div><span style=\"font-weight: 600;\">Innovation:</span> @innovation_category</div>
  <div><span style=\"font-weight: 600;\">Summary:</span> @innovation_summary</div>
</div>
"""

EDGE_TOOLTIP_TEMPLATE = f"""
<div style=\"{TOOLTIP_BOX_STYLE}\">
  <div style=\"font-weight: 600; margin-bottom: 4px;\">@start → @end</div>
  <div><span style=\"font-weight: 600;\">Child release:</span> @release</div>
  <div><span style=\"font-weight: 600;\">Innovation:</span> @innovation_category</div>
  <div><span style=\"font-weight: 600;\">Summary:</span> @innovation_summary</div>
</div>
"""


@dataclass(frozen=True)
class InnovationTimelineLayout:
    """Store node coordinates with innovations on x-axis and time on y-axis."""

    innovations: Tuple[str, ...]
    node_positions_ms: Dict[str, tuple[float, float]]
    node_positions_dt: Dict[str, tuple[float, dt.datetime]]
    x_range: tuple[float, float]
    y_range_ms: tuple[float, float]
    y_range_dt: tuple[dt.datetime, dt.datetime]


def _parse_date(raw: str) -> dt.datetime:
    """Parse ISO-format dates from the curated dataset."""

    try:
        return dt.datetime.strptime(raw, "%Y-%m-%d")
    except ValueError as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Could not parse release_date '{raw}'") from exc


def _format_brand_label(name: str, brand: str) -> str:
    """Return a brand label formatted as "Model (Institution)"."""

    clean_name = name.strip()
    clean_brand = brand.strip()
    if not clean_name:
        return clean_brand or "Unknown"

    if not clean_brand or clean_brand.lower() == "unknown":
        return f"{clean_name} (Unknown)"

    suffix = f" ({clean_name})"
    if clean_brand.endswith(suffix):
        clean_brand = clean_brand[: -len(suffix)].rstrip()

    prefix = f"{clean_name} ("
    if clean_brand.startswith(prefix) and clean_brand.endswith(")"):
        clean_brand = clean_brand[len(prefix) : -1].strip()

    return f"{clean_name} ({clean_brand})"


def _load_models_from_csv(data_path: Path | None = None) -> List[Dict[str, object]]:
    """Load model metadata from the CSV dataset."""

    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    models: List[Dict[str, object]] = []
    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "name",
            "brand",
            "family",
            "release_date",
            "innovation_category",
            "innovation_summary",
        }
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
                "brand": (row.get("brand") or "Unknown").strip() or "Unknown",
                "family": (row.get("family") or "Unknown").strip() or "Unknown",
                "release_date_raw": (row.get("release_date") or "").strip(),
                "influences": influences,
                "innovation_category": (row.get("innovation_category") or "").strip(),
                "innovation_summary": (row.get("innovation_summary") or "").strip(),
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
        release_raw = str(normalised.get("release_date_raw", "")).strip()
        if not release_raw:
            raise ValueError(
                f"Model '{normalised.get('name')}' is missing a release_date"
            )
        release_date = _parse_date(release_raw)
        normalised["release_date"] = release_date
        normalised["release_label"] = release_date.strftime("%b %Y")
        normalised["brand_label"] = _format_brand_label(
            str(normalised.get("name", "")),
            str(normalised.get("brand", "")),
        )
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
                graph.add_edge(
                    influence,
                    model["name"],
                    innovation_category=model["innovation_category"],
                    innovation_summary=model["innovation_summary"],
                )
    return graph


def _build_innovation_timeline_layout(
    models: Iterable[Dict[str, object]]
) -> InnovationTimelineLayout:
    """Return node coordinates with innovation categories on x and time on y."""

    models_list = list(models)
    if not models_list:
        raise ValueError("No models were provided to build the timeline layout")

    innovation_order: List[str] = []
    seen_innovations = set()
    for model in models_list:
        category = model["innovation_category"] or "Unspecified"
        if category not in seen_innovations:
            seen_innovations.add(category)
            innovation_order.append(category)

    innovation_positions = {name: idx for idx, name in enumerate(innovation_order)}

    positions_ms: Dict[str, tuple[float, float]] = {}
    positions_dt: Dict[str, tuple[float, dt.datetime]] = {}
    innovation_offsets: Dict[str, Dict[dt.date, int]] = {}

    for model in models_list:
        name = str(model["name"])
        release_date: dt.datetime = model["release_date"]
        category = model["innovation_category"] or "Unspecified"

        innovation_offsets.setdefault(category, {})
        offsets_for_category = innovation_offsets[category]
        base_key = release_date.date()
        offsets_for_category[base_key] = offsets_for_category.get(base_key, -1) + 1
        jitter = (offsets_for_category[base_key] % 3) * 0.18

        x = innovation_positions[category] + jitter
        y_ms = release_date.timestamp() * 1000.0

        positions_ms[name] = (x, y_ms)
        positions_dt[name] = (x, release_date)

    start_dt = models_list[0]["release_date"] - dt.timedelta(days=60)
    end_dt = models_list[-1]["release_date"] + dt.timedelta(days=60)
    start_ms = start_dt.timestamp() * 1000.0
    end_ms = end_dt.timestamp() * 1000.0

    padding = 0.6 if innovation_positions else 0.5
    x_min = -padding
    x_max = (len(innovation_order) - 1) + padding if innovation_order else padding

    return InnovationTimelineLayout(
        innovations=tuple(innovation_order),
        node_positions_ms=positions_ms,
        node_positions_dt=positions_dt,
        x_range=(x_min, x_max),
        y_range_ms=(start_ms, end_ms),
        y_range_dt=(start_dt, end_dt),
    )


def _prepare_visualisation(
    *, data_path: Path | None = None
) -> tuple[List[Dict[str, object]], nx.DiGraph, InnovationTimelineLayout, Dict[str, str]]:
    """Load the dataset and compute shared artefacts for all outputs."""

    raw_models = _load_models_from_csv(data_path=data_path)
    models = _prepare_models(raw_models)
    graph = _build_graph(models)
    layout = _build_innovation_timeline_layout(models)

    palette = Category20[20]
    brand_labels = tuple(
        dict.fromkeys(model["brand_label"] for model in models)
    )
    color_map = {
        label: palette[index % len(palette)]
        for index, label in enumerate(brand_labels)
    }

    return models, graph, layout, color_map


def _construct_bokeh_figure(
    graph: nx.DiGraph,
    layout: InnovationTimelineLayout,
    color_map: Dict[str, str],
    *,
    title: str = DEFAULT_TITLE,
):
    """Create the configured Bokeh figure from prepared components."""

    plot = figure(
        width=960,
        height=640,
        sizing_mode="stretch_width",
        y_axis_type="datetime",
        x_range=layout.x_range,
        y_range=layout.y_range_ms,
        title=title,
        toolbar_location="above",
    )
    plot.min_border_bottom = 120
    plot.min_border_left = 80
    plot.min_border_right = 40
    plot.min_border_top = 40
    plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), TapTool())

    graph_renderer = from_networkx(graph, layout.node_positions_ms)

    node_source = graph_renderer.node_renderer.data_source
    node_source.data["family"] = [graph.nodes[name]["family"] for name in graph.nodes]
    node_source.data["brand"] = [graph.nodes[name]["brand"] for name in graph.nodes]
    node_source.data["brand_label"] = [
        graph.nodes[name]["brand_label"] for name in graph.nodes
    ]
    node_source.data["release"] = [graph.nodes[name]["release_label"] for name in graph.nodes]
    node_source.data["innovation_category"] = [
        graph.nodes[name]["innovation_category"] for name in graph.nodes
    ]
    node_source.data["innovation_summary"] = [
        graph.nodes[name]["innovation_summary"] for name in graph.nodes
    ]
    node_source.data["color"] = [
        color_map[graph.nodes[name]["brand_label"]] for name in graph.nodes
    ]

    graph_renderer.node_renderer.glyph.size = 18
    graph_renderer.node_renderer.glyph.fill_color = "color"
    graph_renderer.node_renderer.glyph.line_color = "#222222"

    edge_source = graph_renderer.edge_renderer.data_source
    starts = edge_source.data.get("start", [])
    ends = edge_source.data.get("end", [])
    innovation_categories: List[str] = []
    innovation_summaries: List[str] = []
    releases: List[str] = []
    for start, end in zip(starts, ends):
        attributes = graph.edges[start, end]
        innovation_categories.append(attributes.get("innovation_category", ""))
        innovation_summaries.append(attributes.get("innovation_summary", ""))
        releases.append(graph.nodes[end]["release_label"])

    edge_source.data["innovation_category"] = innovation_categories
    edge_source.data["innovation_summary"] = innovation_summaries
    edge_source.data["release"] = releases

    graph_renderer.edge_renderer.glyph.line_alpha = 0.4
    graph_renderer.edge_renderer.glyph.line_width = 2

    plot.renderers.append(graph_renderer)

    node_hover = HoverTool(
        tooltips=NODE_TOOLTIP_TEMPLATE,
        renderers=[graph_renderer.node_renderer],
    )
    node_hover.attachment = "horizontal"
    node_hover.show_arrow = False
    node_hover.point_policy = "follow_mouse"
    edge_hover = HoverTool(
        tooltips=EDGE_TOOLTIP_TEMPLATE,
        renderers=[graph_renderer.edge_renderer],
    )
    edge_hover.attachment = "horizontal"
    edge_hover.show_arrow = False
    edge_hover.point_policy = "follow_mouse"
    edge_hover.line_policy = "nearest"
    plot.add_tools(node_hover, edge_hover)

    # Configure axes labels and tick overrides.
    innovation_labels = {index: label for index, label in enumerate(layout.innovations)}
    plot.xaxis.ticker = list(innovation_labels.keys())
    plot.xaxis.major_label_overrides = innovation_labels
    plot.xaxis.major_label_orientation = math.radians(-45)
    plot.xaxis.major_label_text_align = "right"
    plot.xaxis.major_label_text_baseline = "middle"
    plot.xaxis.major_label_standoff = 12
    plot.xaxis.axis_label = "Technical innovation"
    plot.yaxis.axis_label = "Release timeline"

    # Add a subtitle style label to guide interaction.
    y_min, y_max = layout.y_range_ms
    subtitle_y = y_max - 0.05 * (y_max - y_min)

    subtitle = Label(
        x=0,
        y=subtitle_y,
        x_units="screen",
        y_units="data",
        text="Hover nodes or edges to see innovations. Use scroll to zoom.",
        text_font_size="10pt",
    )
    plot.add_layout(subtitle)

    # Add invisible circles for legend entries positioned outside the plot area.
    legend_items: List[LegendItem] = []
    legend_x = layout.x_range[1] + 0.6
    legend_y_start = layout.y_range_ms[1] + 86_400_000.0  # one day in ms offset
    for index, (label, color) in enumerate(color_map.items()):
        legend_renderer = plot.scatter(
            x=[legend_x],
            y=[legend_y_start + index * 86_400_000.0],
            size=12,
            marker="circle",
            fill_color=color,
            line_color="#222222",
            muted_color=color,
            muted_alpha=0.15,
        )
        legend_items.append(LegendItem(label=label, renderers=[legend_renderer]))

    legend = Legend(items=legend_items, title="Model brands")
    legend.border_line_color = None
    legend.padding = 8
    legend.spacing = 4
    legend.label_text_font_size = "10pt"
    legend.title_text_font_size = "11pt"
    legend.click_policy = "mute"
    plot.add_layout(legend, "right")

    return plot


def build_plot(*, data_path: Path | None = None, title: str = DEFAULT_TITLE):
    """Construct the interactive Bokeh plot for the phylogenetic graph."""

    _, graph, layout, color_map = _prepare_visualisation(data_path=data_path)
    return _construct_bokeh_figure(graph, layout, color_map, title=title)


def export_static_svg(
    graph: nx.DiGraph,
    layout: InnovationTimelineLayout,
    color_map: Dict[str, str],
    destination: Path,
    *,
    title: str = DEFAULT_TITLE,
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
        node_colors.append(color_map[graph.nodes[name]["brand_label"]])

    ax.scatter(
        node_x,
        node_y,
        s=110,
        c=node_colors,
        edgecolors="#222222",
        linewidths=0.6,
        zorder=2,
    )

    ax.set_xlabel("Technical innovation")
    ax.set_ylabel("Release timeline")
    ax.set_xlim(layout.x_range)
    ax.set_ylim(layout.y_range_dt)

    ax.set_xticks(range(len(layout.innovations)))
    ax.set_xticklabels(layout.innovations, rotation=30, ha="right")

    ax.yaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax.grid(True, axis="y", color="#dddddd", linewidth=0.8, alpha=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    subtitle = "Hover in the HTML version to explore innovations"
    ax.set_title(
        f"{title}\n" + subtitle,
        fontsize=14,
        pad=18,
    )

    legend_handles = []
    legend_labels = []
    for label, color in color_map.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="#222222",
                markersize=9,
                linewidth=0,
            )
        )
        legend_labels.append(label)
    ax.legend(
        legend_handles,
        legend_labels,
        title="Model brands",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(destination, format="svg", dpi=150)
    plt.close(fig)
    return destination


def main(
    output_path: Path | None = None,
    data_path: Path | None = None,
    *,
    svg_output_path: Path | None = DEFAULT_SVG_OUTPUT_PATH,
    open_browser: bool = False,
    title: str = DEFAULT_TITLE,
) -> Path:
    """Generate the phylogeny plot and write it to an HTML file."""
    _, graph, layout, color_map = _prepare_visualisation(data_path=data_path)
    plot = _construct_bokeh_figure(graph, layout, color_map, title=title)
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    output_file(str(output_path), title=title)
    save(plot)
    if svg_output_path is not None:
        export_static_svg(graph, layout, color_map, svg_output_path, title=title)
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
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override the title displayed on the generated figure.",
    )
    args = parser.parse_args()

    destination = main(
        output_path=args.output,
        data_path=args.data,
        svg_output_path=None if args.no_svg else args.svg_output,
        open_browser=args.show,
        title=args.title or DEFAULT_TITLE,
    )
    print(f"Saved interactive phylogeny to {destination}")
