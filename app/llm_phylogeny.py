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


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_models.csv"


def _parse_date(raw: str) -> dt.datetime:
    """Convert the input "day-month-year" string into a datetime."""
    return dt.datetime.strptime(raw, "%d-%m-%Y")


def _load_raw_models(data_path: Path | None = None) -> List[Dict[str, object]]:
    """Load the LLM model metadata from the CSV dataset."""

    target = data_path or DEFAULT_DATA_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"Could not locate model dataset at {target}. Please provide --data to the script."
        )

    with target.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "name": row["name"].strip(),
                    "family": row["family"].strip(),
                    "release_month": row["release_month"].strip(),
                    "influences": row.get("influences", "").strip(),
                    "innovation": row["innovation"].strip(),
                }
            )
    return rows


def _prepare_models(data_path: Path | None = None) -> List[Dict[str, object]]:
    """Return a normalised and chronologically sorted list of models."""
    combined = _load_raw_models(data_path)
    for model in combined:
        raw_influences = model.get("influences", []) or []
        if isinstance(raw_influences, str):
            influences = [part.strip() for part in raw_influences.split(";") if part.strip()]
        else:
            influences = list(raw_influences)
        model["influences"] = influences
        release_date = _parse_date(str(model["release_month"]))
        model["release_date"] = release_date
        model["release_label"] = release_date.strftime("%b %Y")
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


def build_plot(data_path: Path | None = None):
    """Construct the interactive Bokeh plot for the phylogenetic graph."""
    models = _prepare_models(data_path)
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
    *,
    open_browser: bool = False,
    data_path: Path | None = None,
) -> Path:
    """Generate the phylogeny plot and write it to an HTML file."""
    plot = build_plot(data_path)
    if output_path is None:
        output_path = Path("llm_phylogeny.html")
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
        default=Path("llm_phylogeny.html"),
        help="Path to the HTML file that will store the plot.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV file containing the LLM phylogeny dataset.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the generated plot in a web browser after saving.",
    )
    args = parser.parse_args()

    destination = main(
        output_path=args.output,
        open_browser=args.show,
        data_path=args.data,
    )
    print(f"Saved interactive phylogeny to {destination}")
