"""Interactive phylogenetic graph of large language models.

This module builds a Bokeh figure that visualises the relationship between
transformer-based language models and the key architectural innovations that link
them together.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
from bokeh.io import output_file, save, show
from bokeh.models import Div
from bokeh.palettes import Category20

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
    models: List[Dict[str, object]],
    graph: nx.DiGraph,
    layout: TimelineLayout,
    color_map: Dict[str, str],
) -> tuple[Div, Dict[str, object], str]:
    """Create the interactive 3D figure rendered inside a Bokeh ``Div`` widget."""

    if not models:
        raise ValueError("No models were provided to construct the 3D figure")

    time_zero = models[0]["release_date"]
    if not isinstance(time_zero, dt.datetime):
        raise TypeError("Model release dates must be datetime objects")

    model_indices = {model["name"]: index for index, model in enumerate(models)}
    innovations = sorted({(model.get("innovation") or "Unknown") for model in models})
    innovation_indices = {label: index for index, label in enumerate(innovations)}

    node_data: Dict[str, List[object]] = {
        "x": [],
        "y": [],
        "z": [],
        "color": [],
        "name": [],
        "family": [],
        "release": [],
        "innovation": [],
        "influences": [],
    }
    positions: Dict[str, Dict[str, float]] = {}

    seconds_in_day = 60 * 60 * 24

    for model in models:
        name = str(model["name"])
        release_date: dt.datetime = model["release_date"]  # type: ignore[assignment]
        x_value = (release_date - time_zero).total_seconds() / seconds_in_day
        y_value = float(model_indices[name])
        innovation_label = str(model.get("innovation") or "Unknown")
        z_value = float(innovation_indices[innovation_label])

        node_data["x"].append(x_value)
        node_data["y"].append(y_value)
        node_data["z"].append(z_value)
        family = str(model["family"])
        node_data["color"].append(color_map.get(family, "#9fa8da"))
        node_data["name"].append(name)
        node_data["family"].append(family)
        node_data["release"].append(model["release_label"])
        node_data["innovation"].append(innovation_label)
        influences = model.get("influences") or []
        node_data["influences"].append(", ".join(influences) if influences else "None")

        positions[name] = {"x": x_value, "y": y_value, "z": z_value}

    x_values = [float(value) for value in node_data["x"]]
    y_values = [float(value) for value in node_data["y"]]
    z_values = [float(value) for value in node_data["z"]]

    def _with_padding(values: List[float], padding: float = 0.5) -> List[float]:
        if not values:
            return [0.0, 1.0]
        span = max(values) - min(values)
        if span == 0:
            span = 1.0
        pad = max(padding, span * 0.05)
        return [min(values) - pad, max(values) + pad]

    axis_limits = {
        "x": _with_padding(x_values, padding=10.0),
        "y": _with_padding(y_values, padding=1.5),
        "z": _with_padding(z_values, padding=1.5),
    }

    edge_pairs: List[Dict[str, object]] = []
    for start, end in graph.edges():
        if start not in positions or end not in positions:
            continue
        start_pos = positions[start]
        end_pos = positions[end]
        edge_pairs.append(
            {
                "x0": start_pos["x"],
                "y0": start_pos["y"],
                "z0": start_pos["z"],
                "x1": end_pos["x"],
                "y1": end_pos["y"],
                "z1": end_pos["z"],
            }
        )

    category_data = {
        "models": [
            {"index": index, "label": str(model["name"])}
            for index, model in enumerate(models)
        ],
        "innovations": [
            {"index": innovation_indices[label], "label": str(label)}
            for label in innovations
        ],
    }

    axis_labels = {
        "x": f"Time (days since {time_zero.strftime('%b %Y')})",
        "y": "Model (chronological index)",
        "z": "Technical innovation",
    }

    legend_items = [
        {"label": family, "color": color_map[family]}
        for family in layout.families
    ]

    instructions = (
        "Drag to rotate • Scroll to zoom • Hover a node to inspect the model and its links. "
        "Panels list the indices used on the model and innovation axes."
    )

    config = {
        "data": node_data,
        "edges": edge_pairs,
        "axis_labels": axis_labels,
        "axis_limits": axis_limits,
        "categories": category_data,
        "legend_items": legend_items,
        "point_size": 16.0,
        "background_color": "#05070d",
        "instructions": instructions,
    }

    div, container_id = _build_threejs_div(config, width=1200, height=800)
    return div, config, container_id



def _build_threejs_div(config: Dict[str, object], *, width: int, height: int) -> tuple[Div, str]:
    """Create a placeholder ``Div`` and identifier for post-processed Three.js wiring."""

    container_id = f"three-phylogeny-{uuid.uuid4().hex}"
    placeholder = f"__THREE_PHYLOGENY::{container_id}__"
    div = Div(text=placeholder, width=width, height=height, render_as_text=False)
    return div, container_id



def _inject_threejs_html(
    destination: Path,
    *,
    container_id: str,
    config: Dict[str, object],
    width: int,
    height: int,
) -> None:
    """Post-process the saved HTML to embed the Three.js bootstrap script."""

    html = destination.read_text(encoding="utf-8")
    placeholder = f"__THREE_PHYLOGENY::{container_id}__"
    container_markup = (
        f'<div id="{container_id}" '
        "class=\"three-phylogeny-container\" style=\"width:100%;height:100%;\"></div>"
    )
    if placeholder in html:
        escaped_markup = json.dumps(container_markup)[1:-1]
        html = html.replace(placeholder, escaped_markup)
    elif container_markup not in html:
        raise RuntimeError(
            "Unable to locate Three.js container markup in saved HTML."
        )

    css_rules = """
.three-phylogeny-container { position: relative; width: 100%; height: 100%; font-family: 'Inter','Helvetica Neue',Arial,sans-serif; }
.three-phylogeny-overlay { position: absolute; inset: 0; pointer-events: none; color: #f8fafc; }
.three-phylogeny-tooltip { position: absolute; min-width: 220px; background: rgba(15,23,42,0.92); border: 1px solid rgba(148,163,184,0.35); border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.4; display: none; pointer-events: none; box-shadow: 0 10px 30px rgba(15,23,42,0.45); backdrop-filter: blur(6px); }
.three-phylogeny-tooltip-title { font-weight: 600; font-size: 14px; margin-bottom: 6px; color: #e2e8f0; }
.three-phylogeny-tooltip-row { display: flex; justify-content: space-between; margin-bottom: 4px; gap: 12px; }
.three-phylogeny-tooltip-row span:first-child { opacity: 0.75; }
.three-phylogeny-axis-panel { position: absolute; bottom: 20px; left: 20px; padding: 12px 16px; border-radius: 10px; background: rgba(15,23,42,0.72); border: 1px solid rgba(148,163,184,0.35); backdrop-filter: blur(6px); box-shadow: 0 8px 25px rgba(15,23,42,0.35); pointer-events: auto; max-width: 360px; font-size: 13px; line-height: 1.5; }
.three-phylogeny-axis-row { display: flex; justify-content: space-between; margin-bottom: 4px; gap: 12px; }
.three-phylogeny-axis-row span:last-child { font-weight: 600; }
.three-phylogeny-legend { position: absolute; top: 20px; left: 20px; display: grid; grid-template-columns: repeat(2, minmax(140px, 1fr)); gap: 8px 14px; padding: 14px 16px; background: rgba(15,23,42,0.72); border-radius: 12px; border: 1px solid rgba(148,163,184,0.35); box-shadow: 0 8px 25px rgba(15,23,42,0.35); backdrop-filter: blur(6px); pointer-events: auto; }
.three-phylogeny-legend-item { display: flex; align-items: center; gap: 10px; font-size: 13px; color: #e2e8f0; }
.three-phylogeny-swatch { width: 14px; height: 14px; border-radius: 50%; box-shadow: 0 2px 8px rgba(15,23,42,0.45); border: 1px solid rgba(15,23,42,0.4); display: inline-block; }
.three-phylogeny-categories { position: absolute; right: 20px; top: 20px; display: grid; gap: 12px; width: min(320px, 28%); pointer-events: auto; }
.three-phylogeny-category { background: rgba(15,23,42,0.72); border-radius: 12px; border: 1px solid rgba(148,163,184,0.35); padding: 14px 16px; backdrop-filter: blur(6px); box-shadow: 0 8px 25px rgba(15,23,42,0.35); max-height: 240px; overflow-y: auto; font-size: 12.5px; line-height: 1.5; }
.three-phylogeny-category h3 { margin: 0 0 8px; font-size: 13px; letter-spacing: 0.01em; text-transform: uppercase; opacity: 0.8; }
.three-phylogeny-category ul { margin: 0; padding-left: 18px; }
.three-phylogeny-category li { margin-bottom: 4px; }
.three-phylogeny-instructions { position: absolute; right: 20px; bottom: 20px; padding: 12px 16px; border-radius: 12px; background: rgba(15,23,42,0.72); border: 1px solid rgba(148,163,184,0.35); backdrop-filter: blur(6px); font-size: 13px; max-width: min(360px, 40%); box-shadow: 0 8px 25px rgba(15,23,42,0.35); pointer-events: auto; }
.three-phylogeny-error { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #f8fafc; font-size: 16px; background: rgba(15,23,42,0.9); border-radius: 12px; }
""".strip()

    script_template = """
(function() {
  const CONFIG = __CONFIG__;
  const CSS_RULES = __CSS__;
  const TARGET_ID = '__CONTAINER_ID__';

  function waitForHost() {
    return new Promise((resolve) => {
      let attempts = 0;
      function check() {
        const host = document.querySelector('[data-root-id] .bk-Div');
        if (host) {
          resolve(host);
          return;
        }
        attempts += 1;
        if (attempts > 200) {
          console.error('Three.js host element not found');
          resolve(null);
          return;
        }
        requestAnimationFrame(check);
      }
      check();
    });
  }

  waitForHost().then((host) => {
    if (!host) {
      return;
    }

    let container = document.getElementById(TARGET_ID);
    if (!container) {
      container = document.createElement('div');
      container.id = TARGET_ID;
      container.className = 'three-phylogeny-container';
      container.style.width = '100%';
      container.style.height = '100%';
      host.innerHTML = '';
      host.appendChild(container);
    }

    if (!document.getElementById('three-phylogeny-style')) {
      const style = document.createElement('style');
      style.id = 'three-phylogeny-style';
      style.textContent = CSS_RULES;
      document.head.appendChild(style);
    }

    function loadScript(url) {
    return new Promise((resolve, reject) => {
      const existing = Array.from(document.getElementsByTagName('script')).find((el) => el.src === url);
      if (existing) {
        if (existing.dataset.loaded === 'true') {
          resolve();
        } else {
          const handle = () => { existing.dataset.loaded = 'true'; resolve(); };
          existing.addEventListener('load', handle, {once: true});
          existing.addEventListener('error', () => reject(new Error('Failed to load ' + url)), {once: true});
        }
        return;
      }
      const script = document.createElement('script');
      script.src = url;
      script.dataset.loaded = 'false';
      script.addEventListener('load', () => { script.dataset.loaded = 'true'; resolve(); }, {once: true});
      script.addEventListener('error', () => reject(new Error('Failed to load ' + url)), {once: true});
      document.head.appendChild(script);
    });
  }

    const ensureThree = loadScript('assets/three.min.js');
    ensureThree
      .then(() => loadScript('assets/OrbitControls.js'))
      .then(init)
      .catch((err) => {
        container.innerHTML = '<div class="three-phylogeny-error">Unable to load 3D resources. See console for details.</div>';
        console.error(err);
      });

  function init() {
    if (!(window.THREE && window.THREE.OrbitControls)) {
      console.error('Three.js resources missing');
      container.innerHTML = '<div class="three-phylogeny-error">Three.js resources missing.</div>';
      return;
    }

    container.innerHTML = '';
    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.background_color || '#05070d');

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const ambient = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.65);
    directional.position.set(1.2, 1.6, 2.4);
    scene.add(directional);

    const overlay = document.createElement('div');
    overlay.className = 'three-phylogeny-overlay';
    container.appendChild(overlay);

    const tooltip = document.createElement('div');
    tooltip.className = 'three-phylogeny-tooltip';
    overlay.appendChild(tooltip);

    const axisPanel = document.createElement('div');
    axisPanel.className = 'three-phylogeny-axis-panel';
    overlay.appendChild(axisPanel);

    ['x','y','z'].forEach((key) => {
      const row = document.createElement('div');
      row.className = 'three-phylogeny-axis-row';
      const label = document.createElement('span');
      label.textContent = CONFIG.axis_labels?.[key] || key.toUpperCase();
      const span = document.createElement('span');
      const limits = CONFIG.axis_limits?.[key] || [];
      span.textContent = limits.map((value) => Number(value).toFixed(1)).join(' → ');
      row.appendChild(label);
      row.appendChild(span);
      axisPanel.appendChild(row);
    });

    const legend = document.createElement('div');
    legend.className = 'three-phylogeny-legend';
    (CONFIG.legend_items || []).forEach((item) => {
      const entry = document.createElement('div');
      entry.className = 'three-phylogeny-legend-item';
      entry.innerHTML = `<span class="three-phylogeny-swatch" style="background:${item.color || '#ffffff'}"></span><span>${item.label || ''}</span>`;
      legend.appendChild(entry);
    });
    overlay.appendChild(legend);

    const categoryWrap = document.createElement('div');
    categoryWrap.className = 'three-phylogeny-categories';
    overlay.appendChild(categoryWrap);

    function buildCategory(key, title) {
      const data = CONFIG.categories?.[key] || [];
      if (!Array.isArray(data) || data.length === 0)
        return;
      const panel = document.createElement('div');
      panel.className = 'three-phylogeny-category';
      const heading = document.createElement('h3');
      heading.textContent = title;
      panel.appendChild(heading);
      const list = document.createElement('ul');
      data.forEach((entry) => {
        const item = document.createElement('li');
        item.textContent = `${entry.index}: ${entry.label}`;
        list.appendChild(item);
      });
      panel.appendChild(list);
      categoryWrap.appendChild(panel);
    }

    buildCategory('models', 'Model index');
    buildCategory('innovations', 'Innovation index');

    if (CONFIG.instructions) {
      const instructions = document.createElement('div');
      instructions.className = 'three-phylogeny-instructions';
      instructions.textContent = CONFIG.instructions;
      overlay.appendChild(instructions);
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const nodeData = CONFIG.data || {};
    const count = (nodeData.x || []).length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      positions[i * 3] = Number(nodeData.x?.[i] ?? 0);
      positions[i * 3 + 1] = Number(nodeData.y?.[i] ?? 0);
      positions[i * 3 + 2] = Number(nodeData.z?.[i] ?? 0);
      const color = new THREE.Color(nodeData.color?.[i] || '#ffffff');
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    pointGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const pointMaterial = new THREE.PointsMaterial({size: CONFIG.point_size || 16, vertexColors: true, sizeAttenuation: true});
    const points = new THREE.Points(pointGeometry, pointMaterial);
    scene.add(points);

    const edges = CONFIG.edges || [];
    if (edges.length > 0) {
      const edgePositions = new Float32Array(edges.length * 6);
      edges.forEach((edge, index) => {
        edgePositions[index * 6] = Number(edge.x0 ?? 0);
        edgePositions[index * 6 + 1] = Number(edge.y0 ?? 0);
        edgePositions[index * 6 + 2] = Number(edge.z0 ?? 0);
        edgePositions[index * 6 + 3] = Number(edge.x1 ?? 0);
        edgePositions[index * 6 + 4] = Number(edge.y1 ?? 0);
        edgePositions[index * 6 + 5] = Number(edge.z1 ?? 0);
      });
      const edgeGeometry = new THREE.BufferGeometry();
      edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
      const edgeMaterial = new THREE.LineBasicMaterial({color: 0x8891a7, transparent: true, opacity: 0.35});
      const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
      scene.add(edgeLines);
    }

    const limits = CONFIG.axis_limits || {};
    const xLimits = limits.x || [0, 1];
    const yLimits = limits.y || [0, 1];
    const zLimits = limits.z || [0, 1];
    const origin = new THREE.Vector3(xLimits[0], yLimits[0], zLimits[0]);
    const axisGroup = new THREE.Group();
    const axisDefs = [
      {dir: new THREE.Vector3(xLimits[1] - xLimits[0], 0, 0), color: 0xff6b6b},
      {dir: new THREE.Vector3(0, yLimits[1] - yLimits[0], 0), color: 0x4ecdc4},
      {dir: new THREE.Vector3(0, 0, zLimits[1] - zLimits[0]), color: 0x1a8cff},
    ];
    axisDefs.forEach((axis) => {
      const geom = new THREE.BufferGeometry().setFromPoints([
        origin,
        origin.clone().add(axis.dir),
      ]);
      axisGroup.add(new THREE.Line(geom, new THREE.LineBasicMaterial({color: axis.color})));
    });
    scene.add(axisGroup);

    function showTooltip(index, event) {
      const name = nodeData.name?.[index] ?? '';
      const family = nodeData.family?.[index] ?? '';
      const release = nodeData.release?.[index] ?? '';
      const innovation = nodeData.innovation?.[index] ?? '';
      const influences = nodeData.influences?.[index] ?? 'None';
      tooltip.innerHTML = `
        <div class="three-phylogeny-tooltip-title">${name}</div>
        <div class="three-phylogeny-tooltip-row"><span>Family</span><span>${family}</span></div>
        <div class="three-phylogeny-tooltip-row"><span>Released</span><span>${release}</span></div>
        <div class="three-phylogeny-tooltip-row"><span>Innovation</span><span>${innovation}</span></div>
        <div class="three-phylogeny-tooltip-row"><span>Influences</span><span>${influences}</span></div>
      `;
      const rect = renderer.domElement.getBoundingClientRect();
      const left = event.clientX - rect.left + 14;
      const top = event.clientY - rect.top + 14;
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
      tooltip.style.display = 'block';
    }

    function hideTooltip() {
      tooltip.style.display = 'none';
    }

    function onPointerMove(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
      raycaster.setFromCamera(mouse, camera);
      const intersections = raycaster.intersectObject(points);
      if (intersections.length > 0 && intersections[0].index != null) {
        showTooltip(intersections[0].index, event);
      } else {
        hideTooltip();
      }
    }

    renderer.domElement.addEventListener('mousemove', onPointerMove);
    renderer.domElement.addEventListener('mouseleave', hideTooltip);

    function resize() {
      const viewWidth = container.clientWidth || __WIDTH__;
      const viewHeight = container.clientHeight || __HEIGHT__;
      renderer.setSize(viewWidth, viewHeight, false);
      camera.aspect = viewWidth / viewHeight;
      camera.updateProjectionMatrix();
    }

    window.addEventListener('resize', resize);
    resize();

    const spanX = xLimits[1] - xLimits[0];
    const spanY = yLimits[1] - yLimits[0];
    const spanZ = zLimits[1] - zLimits[0];
    const maxSpan = Math.max(spanX, spanY, spanZ, 1);
    camera.position.set(origin.x + spanX * 0.6, origin.y + spanY * 0.5, origin.z + maxSpan * 2.2);
    controls.target.copy(origin.clone().add(new THREE.Vector3(spanX / 2, spanY / 2, spanZ / 2)));

    function renderLoop() {
      requestAnimationFrame(renderLoop);
      controls.update();
      renderer.render(scene, camera);
    }

    renderLoop();
  }
  });
})();
"""

    replacements = {
        "__CONFIG__": json.dumps(config),
        "__CONTAINER_ID__": container_id,
        "__CSS__": json.dumps(css_rules),
        "__WIDTH__": str(width),
        "__HEIGHT__": str(height),
    }

    script_body = script_template
    for key, value in replacements.items():
        script_body = script_body.replace(key, value)

    script_tag = f"<script type=\"text/javascript\">\n{script_body}\n</script>"

    insertion_point = html.rfind("</body>")
    if insertion_point == -1:
        insertion_point = len(html)

    updated_html = html[:insertion_point] + script_tag + "\n" + html[insertion_point:]
    destination.write_text(updated_html, encoding="utf-8")



def build_plot(*, data_path: Path | None = None):
    """Construct the interactive Bokeh plot for the phylogenetic graph."""

    models, graph, layout, color_map = _prepare_visualisation(data_path=data_path)
    plot, _, _ = _construct_bokeh_figure(models, graph, layout, color_map)
    return plot


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
    models, graph, layout, color_map = _prepare_visualisation(data_path=data_path)
    plot, config, container_id = _construct_bokeh_figure(models, graph, layout, color_map)
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    output_file(str(output_path), title="LLM Phylogeny")
    save(plot)
    _inject_threejs_html(output_path, container_id=container_id, config=config, width=plot.width or 1200, height=plot.height or 800)
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
