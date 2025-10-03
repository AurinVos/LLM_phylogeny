"""Interactive phylogenetic graph of large language models.

This module builds a Bokeh figure that visualises the relationship between
transformer-based language models and the key architectural innovations that link
them together.
"""
from __future__ import annotations

import argparse
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


def _parse_date(raw: str) -> dt.datetime:
    """Convert the input "day-month-year" string into a datetime."""
    return dt.datetime.strptime(raw, "%d-%m-%Y")


MODELS: List[Dict[str, object]] = [
    {
        "name": "Attention Is All You Need",
        "family": "Root",
        "release_month": "1-6-2017",
        "influences": [],
        "innovation": "Introduced the Transformer architecture with multi-head self-attention and positional encoding",
    },
    {
        "name": "GPT-1",
        "family": "GPT",
        "release_month": "1-6-2018",
        "influences": ["Attention Is All You Need"],
        "innovation": "Applied decoder-only Transformer with generative pretraining on BooksCorpus",
    },
    {
        "name": "BERT",
        "family": "Encoder-only",
        "release_month": "1-10-2018",
        "influences": ["Attention Is All You Need"],
        "innovation": "Bidirectional encoder pretraining with masked language modelling and next sentence prediction",
    },
    {
        "name": "GPT-2",
        "family": "GPT",
        "release_month": "1-2-2019",
        "influences": ["GPT-1"],
        "innovation": "Scaled decoder-only models with zero-shot transfer via WebText",
    },
    {
        "name": "Transformer-XL",
        "family": "Long-context",
        "release_month": "1-1-2019",
        "influences": ["Attention Is All You Need"],
        "innovation": "Segment-level recurrence and relative positional encoding for long-context modelling",
    },
    {
        "name": "XLNet",
        "family": "Encoder-only",
        "release_month": "1-6-2019",
        "influences": ["Transformer-XL", "BERT"],
        "innovation": "Permuted language modelling objective blending autoregressive and autoencoding pretraining",
    },
    {
        "name": "Megatron-LM",
        "family": "Scaling",
        "release_month": "1-8-2019",
        "influences": ["GPT-2"],
        "innovation": "Model parallelism for trillion-parameter scale using tensor and pipeline parallelism",
    },
    {
        "name": "RoBERTa",
        "family": "Encoder-only",
        "release_month": "1-7-2019",
        "influences": ["BERT"],
        "innovation": "Optimised masked language modelling with longer training, dynamic masking, and larger batches",
    },
    {
        "name": "ALBERT",
        "family": "Encoder-only",
        "release_month": "1-9-2019",
        "influences": ["BERT"],
        "innovation": "Parameter sharing and factorised embeddings for lightweight bidirectional transformers",
    },
    {
        "name": "BART",
        "family": "Seq2Seq",
        "release_month": "1-10-2019",
        "influences": ["Attention Is All You Need"],
        "innovation": "Denoising autoencoder that bridges encoder-decoder pretraining for text generation",
    },
    {
        "name": "T5",
        "family": "Seq2Seq",
        "release_month": "1-10-2019",
        "influences": ["Attention Is All You Need"],
        "innovation": "Text-to-Text framework with unified transfer learning and span-corruption objective",
    },
    {
        "name": "ELECTRA",
        "family": "Encoder-only",
        "release_month": "1-3-2020",
        "influences": ["BERT"],
        "innovation": "Replaced masked language modelling with discriminator that detects replaced tokens",
    },
    {
        "name": "GPT-3",
        "family": "GPT",
        "release_month": "1-5-2020",
        "influences": ["GPT-2"],
        "innovation": "175B parameter scaling with in-context learning across diverse tasks",
    },
    {
        "name": "Switch Transformer",
        "family": "Mixture-of-Experts",
        "release_month": "1-1-2021",
        "influences": ["T5"],
        "innovation": "Sparse mixture-of-experts routing enabling trillion-parameter efficiency",
    },
    {
        "name": "LaMDA",
        "family": "Google",
        "release_month": "1-5-2021",
        "influences": ["Attention Is All You Need"],
        "innovation": "Dialogue-optimised training with safety fine-tuning and grounded responses",
    },
    {
        "name": "Gopher",
        "family": "DeepMind/Scaling",
        "release_month": "1-12-2021",
        "influences": ["Attention Is All You Need"],
        "innovation": "Scaling laws with retrieval-style evaluation and precision study for large transformers",
    },
    {
        "name": "GLaM",
        "family": "Mixture-of-Experts",
        "release_month": "1-12-2021",
        "influences": ["Switch Transformer"],
        "innovation": "Hierarchical mixture-of-experts with token-level routing across 1.2T parameters",
    },
    {
        "name": "Chinchilla",
        "family": "DeepMind/Scaling",
        "release_month": "1-3-2022",
        "influences": ["Gopher"],
        "innovation": "Data/parameter scaling law balancing showing benefits of more tokens over parameters",
    },
    {
        "name": "PaLM",
        "family": "Google",
        "release_month": "1-4-2022",
        "influences": ["LaMDA"],
        "innovation": "Pathways system with parallelism and chain-of-thought prompting across 540B parameters",
    },
    {
        "name": "UL2",
        "family": "Seq2Seq",
        "release_month": "1-5-2022",
        "influences": ["T5"],
        "innovation": "Mixture-of-denoisers objective supporting multiple corruption schemes for unified learning",
    },
    {
        "name": "OPT",
        "family": "Open-Source GPT",
        "release_month": "1-5-2022",
        "influences": ["GPT-3"],
        "innovation": "Reproducible GPT-3 class model with fully documented training pipeline",
    },
    {
        "name": "BLOOM",
        "family": "Open-Source GPT",
        "release_month": "1-7-2022",
        "influences": ["GPT-3", "Megatron-LM"],
        "innovation": "Multilingual open-access 176B parameter model trained collaboratively via Megatron-DeepSpeed",
    },
    {
        "name": "LLaMA",
        "family": "LLaMA",
        "release_month": "1-2-2023",
        "influences": ["Chinchilla"],
        "innovation": "Efficient scaling via smaller datasets, grouped-query attention, and open research weights",
    },
    {
        "name": "GPT-4",
        "family": "GPT",
        "release_month": "1-3-2023",
        "influences": ["Chinchilla", "GPT-3"],
        "innovation": "Large multimodal alignment with reinforced fine-tuning and tool integration",
    },
    {
        "name": "InstructGPT",
        "family": "Alignment",
        "release_month": "1-1-2022",
        "influences": ["GPT-3"],
        "innovation": "Reinforcement learning from human feedback tailored to instruction following",
    },
    {
        "name": "Constitutional AI",
        "family": "Alignment",
        "release_month": "1-12-2022",
        "influences": ["InstructGPT"],
        "innovation": "Self-critiquing alignment loop guided by explicit normative principles",
    },
    {
        "name": "Claude 1",
        "family": "Claude",
        "release_month": "1-3-2023",
        "influences": ["Constitutional AI"],
        "innovation": "Applied constitutional AI feedback to align helpful and harmless behaviour",
    },
    {
        "name": "Baichuan-7B",
        "family": "Baichuan",
        "release_month": "1-6-2023",
        "influences": ["LLaMA"],
        "innovation": "Chinese-English bilingual adaptation of LLaMA with extended vocabulary",
    },
    {
        "name": "Claude 2",
        "family": "Claude",
        "release_month": "1-7-2023",
        "influences": ["Claude 1"],
        "innovation": "Expanded context and constitutional alignment refinements with tool use",
    },
    {
        "name": "Llama 2",
        "family": "LLaMA",
        "release_month": "1-7-2023",
        "influences": ["LLaMA"],
        "innovation": "Open-weight release with supervised fine-tuning and RLHF safety tuning",
    },
    {
        "name": "InternLM-7B",
        "family": "InternLM",
        "release_month": "1-7-2023",
        "influences": ["LLaMA"],
        "innovation": "Toolkit-oriented Chinese open model with multi-stage pretraining and alignment",
    },
    {
        "name": "Claude Instant 1.2",
        "family": "Claude",
        "release_month": "1-8-2023",
        "influences": ["Claude 1"],
        "innovation": "Latency-optimised Claude variant retaining constitutional safety guarantees",
    },
    {
        "name": "Qwen 7B",
        "family": "Qwen",
        "release_month": "1-8-2023",
        "influences": ["Chinchilla", "LLaMA"],
        "innovation": "Chinese-English foundation with rotary embeddings and fine-grained tokeniser",
    },
    {
        "name": "Mistral 7B",
        "family": "Mistral",
        "release_month": "1-9-2023",
        "influences": ["Chinchilla", "LLaMA"],
        "innovation": "Sliding window attention and grouped-query attention for efficient small models",
    },
    {
        "name": "Baichuan2",
        "family": "Baichuan",
        "release_month": "1-9-2023",
        "influences": ["Baichuan-7B"],
        "innovation": "Improved bilingual data curation with extended context and tool APIs",
    },
    {
        "name": "Claude 2.1",
        "family": "Claude",
        "release_month": "1-11-2023",
        "influences": ["Claude 2"],
        "innovation": "Higher factual reliability and longer context for enterprise tasks",
    },
    {
        "name": "Yi-34B",
        "family": "Yi",
        "release_month": "1-11-2023",
        "influences": ["Llama 2"],
        "innovation": "Balanced bilingual dataset with progressive context extension",
    },
    {
        "name": "Mixtral 8x7B",
        "family": "Mistral",
        "release_month": "1-12-2023",
        "influences": ["Mistral 7B"],
        "innovation": "Sparse mixture-of-experts combining eight Mistral experts with router training",
    },
    {
        "name": "InternLM2-7B",
        "family": "InternLM",
        "release_month": "1-1-2024",
        "influences": ["InternLM-7B"],
        "innovation": "Iterative distillation with modular tool-using skills and expanded context",
    },
    {
        "name": "Baichuan3",
        "family": "Baichuan",
        "release_month": "1-1-2024",
        "influences": ["Baichuan2"],
        "innovation": "Domain-adaptive pretraining with knowledge-augmented decoding",
    },
    {
        "name": "Gemini 1.5",
        "family": "Google",
        "release_month": "1-2-2024",
        "influences": ["PaLM"],
        "innovation": "Mixture-of-experts multimodal model with million-token context",
    },
    {
        "name": "Claude 3 Haiku",
        "family": "Claude",
        "release_month": "1-3-2024",
        "influences": ["Claude 2.1"],
        "innovation": "Fast multimodal assistant with revised constitutional tuning",
    },
    {
        "name": "Claude 3 Sonnet",
        "family": "Claude",
        "release_month": "1-3-2024",
        "influences": ["Claude 2.1"],
        "innovation": "Mid-tier multimodal reasoning with tool orchestration",
    },
    {
        "name": "Claude 3 Opus",
        "family": "Claude",
        "release_month": "1-3-2024",
        "influences": ["Claude 2.1"],
        "innovation": "Flagship Claude with state-of-the-art reasoning and coding alignment",
    },
    {
        "name": "Llama 3",
        "family": "LLaMA",
        "release_month": "1-4-2024",
        "influences": ["Llama 2"],
        "innovation": "Token-efficient vocabulary and speculatively decoded training mix",
    },
    {
        "name": "GPT-4o",
        "family": "GPT",
        "release_month": "1-5-2024",
        "influences": ["GPT-4"],
        "innovation": "Unified multimodal end-to-end model with real-time streaming latency",
    },
    {
        "name": "Claude 3.5 Sonnet",
        "family": "Claude",
        "release_month": "1-6-2024",
        "influences": ["Claude 3 Sonnet"],
        "innovation": "Improved tool-use reliability and creative reasoning",
    },
    {
        "name": "Qwen2",
        "family": "Qwen",
        "release_month": "1-6-2024",
        "influences": ["Qwen 7B"],
        "innovation": "Data mixture refresh with extended context and reasoning tuning",
    },
    {
        "name": "Llama 3.1",
        "family": "LLaMA",
        "release_month": "1-7-2024",
        "influences": ["Llama 3"],
        "innovation": "Multi-token prediction and improved tool-calling APIs",
    },
    {
        "name": "DeepSeek-V2.5",
        "family": "DeepSeek",
        "release_month": "1-9-2024",
        "influences": ["LLaMA"],
        "innovation": "Sparse mixture-of-experts with hybrid reinforcement learning",
    },
    {
        "name": "Claude 3.5 Haiku",
        "family": "Claude",
        "release_month": "1-10-2024",
        "influences": ["Claude 3 Haiku"],
        "innovation": "Faster multimodal responses with improved grounding",
    },
    {
        "name": "DeepSeek-V3",
        "family": "DeepSeek",
        "release_month": "1-12-2024",
        "influences": ["DeepSeek-V2.5"],
        "innovation": "Unified MoE and dense experts with reinforcement fine-tuning",
    },
    {
        "name": "Qwen2.5 (Max)",
        "family": "Qwen",
        "release_month": "1-1-2025",
        "influences": ["Qwen2"],
        "innovation": "Expanded multilingual coverage with large context generalisation",
    },
    {
        "name": "DeepSeek-R1",
        "family": "DeepSeek",
        "release_month": "1-1-2025",
        "influences": ["DeepSeek-V3"],
        "innovation": "Reinforced reasoning with reward models targeting mathematical proofs",
    },
    {
        "name": "Claude 3.7 Sonnet",
        "family": "Claude",
        "release_month": "1-2-2025",
        "influences": ["Claude 3.5 Sonnet"],
        "innovation": "Long-context orchestration and improved agentic behaviours",
    },
    {
        "name": "Qwen3",
        "family": "Qwen",
        "release_month": "1-4-2025",
        "influences": ["Qwen2.5 (Max)"],
        "innovation": "Mixture-of-experts scaling with automated reasoning curriculum",
    },
    {
        "name": "Claude 4 Sonnet",
        "family": "Claude",
        "release_month": "1-5-2025",
        "influences": ["Claude 3.7 Sonnet"],
        "innovation": "Structured tool-use planning with autonomous memory",
    },
    {
        "name": "Claude 4 Opus",
        "family": "Claude",
        "release_month": "1-5-2025",
        "influences": ["Claude 3 Opus"],
        "innovation": "Frontier reasoning with multi-agent constitutional guidance",
    },
    {
        "name": "GPT-5",
        "family": "GPT",
        "release_month": "1-8-2025",
        "influences": ["GPT-4o"],
        "innovation": "Next-generation multimodal orchestration with autonomous tool chains",
    },
    {
        "name": "Claude Opus 4.1",
        "family": "Claude",
        "release_month": "1-8-2025",
        "influences": ["Claude 4 Opus"],
        "innovation": "Iterative self-improvement through agentic evaluation loops",
    },
    {
        "name": "Claude 4.5 Sonnet",
        "family": "Claude",
        "release_month": "1-9-2025",
        "influences": ["Claude 4 Sonnet"],
        "innovation": "Hybrid symbolic-neural planning with compressed memory",
    },
]


def _prepare_models() -> List[Dict[str, object]]:
    """Return a normalised and chronologically sorted list of models."""
    combined = MODELS.copy()
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


def build_plot():
    """Construct the interactive Bokeh plot for the phylogenetic graph."""
    models = _prepare_models()
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


def main(output_path: Path | None = None, *, open_browser: bool = False) -> Path:
    """Generate the phylogeny plot and write it to an HTML file."""
    plot = build_plot()
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
        "--show",
        action="store_true",
        help="Open the generated plot in a web browser after saving.",
    )
    args = parser.parse_args()

    destination = main(output_path=args.output, open_browser=args.show)
    print(f"Saved interactive phylogeny to {destination}")
