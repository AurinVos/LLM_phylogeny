# LLM Phylogeny

This project maps how landmark large language models build upon one another's
architectural innovations. The interactive network links each release to the
ideas it inherits, making it easier to trace the technical lineage that shaped
the current generation of transformer systems.

## Explore the interactive network

> 📈 **Interact in your browser**  
> Launch the [LLM phylogeny explorer](https://htmlpreview.github.io/?https://raw.githubusercontent.com/OWNER/LLM_phylogeny/main/docs/interactive_llm_phylogeny.html) to pan, zoom, and inspect the model graph. Replace `OWNER` with the GitHub user or organisation that hosts this repository when sharing a fork.
>
> The same HTML file lives at [`docs/interactive_llm_phylogeny.html`](docs/interactive_llm_phylogeny.html) if you prefer to open it locally.

Each node in the visualization represents a model release, coloured by family
(GPT, LLaMA, Claude, etc.). Hovering over nodes reveals the month of release and
the key innovation introduced, while the directional edges show how ideas flow
from one generation to the next.

## Data

All model metadata is stored in [`data/llm_models.csv`](data/llm_models.csv).
The table captures the information needed to reconstruct the graph:

| Column | Description |
| --- | --- |
| `name` | Model or paper name. |
| `family` | High-level family or ecosystem. Used to colour the visualization. |
| `release_month` | Release date encoded as `day-month-year` for chronological sorting. |
| `influences` | Semi-colon separated list of upstream models that directly inspired the release. |
| `innovation` | Short summary of the primary technical contribution. |

The dataset currently contains 61 milestone releases spanning 2017–2025. If a
model is missing, simply append a new row to the CSV—the visualization will pick
it up automatically the next time it is rendered.

## Regenerating the visualization

1. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate the HTML file (outputs to `docs/interactive_llm_phylogeny.html` by default):
   ```bash
   python app/llm_phylogeny.py
   ```
3. Open the generated HTML in your browser to explore the network. Pass
   `--show` to automatically launch a browser window, `--output` to customise
   the destination file, or `--data` to point at an alternate CSV.

## How it works

`app/llm_phylogeny.py` loads the CSV, constructs a directed graph with NetworkX,
and renders it with Bokeh. The x-axis encodes the release timeline while the
stacked y-axis separates families, helping you see both temporal progression and
conceptual clustering at a glance.

## Contributing

Contributions that expand the dataset or improve the interactive explorer are
welcome. Please keep entries concise, cite publicly available milestones, and
ensure new data rows follow the existing CSV schema so the tooling continues to
work without modification.
