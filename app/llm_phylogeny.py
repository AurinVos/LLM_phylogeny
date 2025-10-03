"""Interactive phylogenetic graph of large language models.

This module builds a Bokeh figure that visualises the relationship between
transformer-based language models and the key architectural innovations that link
them together.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, List

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


def _timeline_layout(graph: nx.DiGraph) -> Dict[str, tuple[float, float]]:
    models = dict(graph.nodes(data=True))
    families = sorted({data["family"] for data in models.values()})
    family_y = {family: index for index, family in enumerate(families)}

    # Track repeated release months per family to avoid exact overlap.
    family_release_offsets: Dict[str, Dict[dt.date, int]] = {}
    layout: Dict[str, tuple[float, float]] = {}
    for name, data in models.items():
        release_date: dt.datetime = data["release_date"]
        family: str = data["family"]
        family_release_offsets.setdefault(family, {})
        offsets_for_family = family_release_offsets[family]
        base_key = release_date.date()
        offsets_for_family[base_key] = offsets_for_family.get(base_key, -1) + 1
        jitter = (offsets_for_family[base_key] % 3) * 0.15
        x = release_date.timestamp() * 1000.0
        y = family_y[family] + jitter
        layout[name] = (x, y)
    return layout


def build_plot(*, data_path: Path | None = None):
    """Construct the interactive Bokeh plot for the phylogenetic graph."""
    raw_models = _load_models_from_csv(data_path=data_path)
    models = _prepare_models(raw_models)
    graph = _build_graph(models)

    palette = Category20[20]
    families = sorted({data["family"] for _, data in graph.nodes(data=True)})
    color_map = {family: palette[index % len(palette)] for index, family in enumerate(families)}

    start_ts = (models[0]["release_date"] - dt.timedelta(days=60)).timestamp() * 1000.0
    end_ts = (models[-1]["release_date"] + dt.timedelta(days=60)).timestamp() * 1000.0

    plot = figure(
        width=1200,
        height=800,
        x_axis_type="datetime",
        x_range=(start_ts, end_ts),
        y_range=(-1, len(families) + 1),
        title="Phylogeny of Transformer Language Models",
        toolbar_location="above",
    )
    plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), TapTool())

    graph_renderer = from_networkx(graph, _timeline_layout)

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
    families_sorted = sorted(families)
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


def main(
    output_path: Path | None = None,
    data_path: Path | None = None,
    *,
    open_browser: bool = False,
) -> Path:
    """Generate the phylogeny plot and write it to an HTML file."""
    plot = build_plot(data_path=data_path)
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    output_file(str(output_path), title="LLM Phylogeny")
    save(plot)
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
    args = parser.parse_args()

    destination = main(
        output_path=args.output,
        data_path=args.data,
        open_browser=args.show,
    )
    print(f"Saved interactive phylogeny to {destination}")
