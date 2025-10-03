# LLM_phylogeny

A repository to show the phylogeny of LLMs based on technical innovations starting from "Attention Is All You Need".

## Generating the phylogeny

The interactive graph is rendered with [Bokeh](https://bokeh.org/) and written to a self-contained HTML file. To generate it:

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Inspect or extend the underlying dataset located at
   [`data/llm_models.csv`](data/llm_models.csv). The CSV stores the model name,
   family, release month, key influences, and the innovation that model
   introduced or popularised. Influences are separated by semicolons.

3. Run the script to build the visualisation:

   ```bash
   python app/llm_phylogeny.py --output llm_phylogeny.html
   ```

   Add `--show` to automatically open the graph in your default browser after it is saved.

   To render the plot with a modified dataset, pass the `--data` flag pointing to
   an alternative CSV file that follows the same column structure:

   ```bash
   python app/llm_phylogeny.py --data path/to/custom_models.csv
   ```

The resulting HTML file can be opened in any modern browser and shared without additional servers.

## Interacting with the graph

Once opened, the graph can be explored using standard Bokeh controls:

- **Pan:** Click and drag anywhere on the canvas to reposition the graph.
- **Zoom:** Use the mouse wheel or trackpad pinch gesture to zoom in and out smoothly.
- **Hover details:** Pause the cursor over nodes or edges to reveal release dates and the technical innovations connecting the models.
- **Reset view:** Press the reset control in the toolbar (or refresh the page) to return to the default zoom and position.

The y-axis groups models by family, while the x-axis arranges them chronologically to emphasise how innovations propagate through time.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
