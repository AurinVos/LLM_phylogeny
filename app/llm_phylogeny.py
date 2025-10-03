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
from bokeh.core.properties import Any as BkAny
from bokeh.core.properties import Dict as BkDict
from bokeh.core.properties import Float as BkFloat
from bokeh.core.properties import List as BkList
from bokeh.core.properties import String as BkString
from bokeh.io import output_file, save, show
from bokeh.models import LayoutDOM
from bokeh.palettes import Category20
from bokeh.util.compiler import JavaScript

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


class ThreeDScatter(LayoutDOM):
    """Custom Bokeh model that renders an interactive 3D scatter plot."""

    __javascript__ = [
        "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js",
        "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.min.js",
    ]

    __implementation__ = JavaScript(
        """
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"
import * as p from "core/properties"

declare const THREE: any

type AxisVectors = {
    origin: any
    x: any
    y: any
    z: any
}

export class ThreeDScatterView extends LayoutDOMView {
    declare model: ThreeDScatter

    private _container: HTMLElement | null = null
    private _renderer: any = null
    private _scene: any = null
    private _camera: any = null
    private _controls: any = null
    private _points: any = null
    private _edges: any = null
    private _axisGroup: any = null
    private _axisVectors: AxisVectors | null = null
    private _raycaster: any
    private _mouse: any
    private _overlay: HTMLElement | null = null
    private _tooltip: HTMLElement | null = null
    private _axisLabels: Record<string, HTMLElement> = {}
    private _legendEl: HTMLElement | null = null
    private _categoryEls: Record<string, HTMLElement> = {}
    private _instructionsEl: HTMLElement | null = null
    private _animationHandle: number | null = null

    private readonly _handleResize = () => this._resize()
    private readonly _handlePointerMove = (event: MouseEvent) => this._onPointerMove(event)
    private readonly _handlePointerLeave = () => this._hideTooltip()

    constructor(options: any) {
        super(options)
        this._raycaster = new THREE.Raycaster()
        this._mouse = new THREE.Vector2()
    }

    override connect_signals(): void {
        super.connect_signals()
        const {
            data,
            edges,
            axis_labels,
            axis_limits,
            categories,
            legend_items,
            point_size,
            background_color,
            instructions,
        } = this.model.properties
        this.on_change(data, () => this._updatePoints())
        this.on_change(edges, () => this._updateEdges())
        this.on_change(axis_labels, () => this._updateAxisLabels())
        this.on_change(axis_limits, () => this._rebuildAxes())
        this.on_change(categories, () => this._buildCategories())
        this.on_change(legend_items, () => this._buildLegend())
        this.on_change(point_size, () => this._updatePointSize())
        this.on_change(background_color, () => this._updateBackground())
        this.on_change(instructions, () => this._updateInstructions())
    }

    override remove(): void {
        super.remove()
        if (this._animationHandle != null) {
            cancelAnimationFrame(this._animationHandle)
            this._animationHandle = null
        }
        if (this._controls != null) {
            this._controls.dispose()
            this._controls = null
        }
        if (this._renderer != null) {
            this._renderer.dispose()
            this._renderer = null
        }
        window.removeEventListener("resize", this._handleResize)
    }

    override render(): void {
        super.render()
        if (this._container == null) {
            this._container = document.createElement("div")
            this._container.style.position = "relative"
            this._container.style.width = "100%"
            this._container.style.height = "100%"
            this.shadow_el.appendChild(this._container)

            this._initThree()
            this._buildOverlay()
            this._buildLegend()
            this._buildCategories()
            this._updateInstructions()

            const canvas = this._renderer.domElement
            canvas.style.width = "100%"
            canvas.style.height = "100%"
            canvas.addEventListener("mousemove", this._handlePointerMove)
            canvas.addEventListener("mouseleave", this._handlePointerLeave)
            window.addEventListener("resize", this._handleResize)
        }

        this._resize()
        this._updateBackground()
        this._rebuildAxes()
        this._updatePoints()
        this._updateEdges()
        this._startAnimationLoop()
    }

    private _initThree(): void {
        if (this._container == null)
            return
        this._renderer = new THREE.WebGLRenderer({antialias: true})
        this._renderer.setPixelRatio(window.devicePixelRatio || 1)
        this._container.appendChild(this._renderer.domElement)

        this._scene = new THREE.Scene()

        this._camera = new THREE.PerspectiveCamera(45, 1, 0.1, 10000)
        this._camera.position.set(0, 0, 100)

        this._controls = new THREE.OrbitControls(this._camera, this._renderer.domElement)
        this._controls.enableDamping = true
        this._controls.dampingFactor = 0.08

        const ambient = new THREE.AmbientLight(0xffffff, 0.65)
        this._scene.add(ambient)
        const directional = new THREE.DirectionalLight(0xffffff, 0.65)
        directional.position.set(1.2, 1.6, 2.4)
        this._scene.add(directional)
    }

    private _buildOverlay(): void {
        if (this._container == null)
            return

        const style = document.createElement("style")
        style.textContent = `
            :host {
                font-family: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
            }
            .three-overlay {
                position: absolute;
                inset: 0;
                pointer-events: none;
                color: #f5f5f5;
                font-size: 12px;
            }
            .three-tooltip {
                position: absolute;
                pointer-events: none;
                background: rgba(12, 12, 18, 0.92);
                border-radius: 8px;
                padding: 10px 12px;
                border: 1px solid rgba(255, 255, 255, 0.12);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
                min-width: 220px;
                display: none;
                line-height: 1.45;
            }
            .three-tooltip .tooltip-title {
                font-weight: 600;
                margin-bottom: 6px;
            }
            .three-tooltip .tooltip-row {
                display: flex;
                justify-content: space-between;
                gap: 8px;
            }
            .three-tooltip .tooltip-row span:first-child {
                color: #9fa8da;
            }
            .three-axis-label {
                position: absolute;
                transform: translate(-50%, -50%);
                background: rgba(14, 15, 24, 0.78);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 4px;
                padding: 4px 8px;
                pointer-events: none;
                font-size: 11px;
                letter-spacing: 0.2px;
            }
            .three-legend,
            .three-category-panel,
            .three-instructions {
                background: rgba(10, 12, 20, 0.78);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(6px);
                padding: 12px;
                pointer-events: auto;
            }
            .three-legend {
                position: absolute;
                top: 18px;
                right: 18px;
                min-width: 200px;
            }
            .three-legend .legend-title {
                font-weight: 600;
                margin-bottom: 8px;
            }
            .three-legend .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 4px;
            }
            .three-legend .legend-swatch {
                width: 14px;
                height: 14px;
                border-radius: 50%;
                border: 1px solid rgba(255, 255, 255, 0.65);
            }
            .three-category-panel {
                position: absolute;
                bottom: 18px;
                max-height: 220px;
                overflow-y: auto;
                width: 260px;
                line-height: 1.35;
            }
            .three-category-panel .panel-title {
                font-weight: 600;
                margin-bottom: 6px;
            }
            .three-category-panel .panel-row {
                font-size: 11px;
                margin-bottom: 4px;
            }
            .three-category-panel.models {
                left: 18px;
            }
            .three-category-panel.innovations {
                right: 18px;
            }
            .three-instructions {
                position: absolute;
                top: 18px;
                left: 18px;
                max-width: 260px;
                line-height: 1.5;
                font-size: 12px;
            }
        `
        this.shadow_el.appendChild(style)

        this._overlay = document.createElement("div")
        this._overlay.className = "three-overlay"
        this._container.appendChild(this._overlay)

        this._tooltip = document.createElement("div")
        this._tooltip.className = "three-tooltip"
        this._overlay.appendChild(this._tooltip)

        this._axisLabels = {
            x: document.createElement("div"),
            y: document.createElement("div"),
            z: document.createElement("div"),
        }
        for (const key of Object.keys(this._axisLabels)) {
            const element = this._axisLabels[key]
            element.className = "three-axis-label"
            this._overlay.appendChild(element)
        }

        this._instructionsEl = document.createElement("div")
        this._instructionsEl.className = "three-instructions"
        this._overlay.appendChild(this._instructionsEl)
    }

    private _buildLegend(): void {
        if (this._overlay == null)
            return
        if (this._legendEl != null) {
            this._legendEl.remove()
            this._legendEl = null
        }
        const items = this.model.legend_items
        if (items.length === 0)
            return

        const legend = document.createElement("div")
        legend.className = "three-legend"

        const title = document.createElement("div")
        title.className = "legend-title"
        title.textContent = "Model families"
        legend.appendChild(title)

        for (const item of items) {
            const row = document.createElement("div")
            row.className = "legend-item"

            const swatch = document.createElement("span")
            swatch.className = "legend-swatch"
            swatch.style.background = item.color ?? "#cccccc"
            row.appendChild(swatch)

            const label = document.createElement("span")
            label.textContent = item.label ?? ""
            row.appendChild(label)

            legend.appendChild(row)
        }

        this._overlay.appendChild(legend)
        this._legendEl = legend
    }

    private _buildCategories(): void {
        if (this._overlay == null)
            return
        for (const key of Object.keys(this._categoryEls)) {
            const panel = this._categoryEls[key]
            panel.remove()
        }
        this._categoryEls = {}

        const categories = this.model.categories
        const modelEntries = categories["models"] ?? []
        if (modelEntries.length > 0) {
            const panel = document.createElement("div")
            panel.className = "three-category-panel models"
            panel.innerHTML = '<div class="panel-title">Model indices</div>'
            for (const entry of modelEntries) {
                const row = document.createElement("div")
                row.className = "panel-row"
                row.textContent = `${entry.index}: ${entry.label}`
                panel.appendChild(row)
            }
            this._overlay.appendChild(panel)
            this._categoryEls["models"] = panel
        }

        const innovationEntries = categories["innovations"] ?? []
        if (innovationEntries.length > 0) {
            const panel = document.createElement("div")
            panel.className = "three-category-panel innovations"
            panel.innerHTML = '<div class="panel-title">Technical innovations</div>'
            for (const entry of innovationEntries) {
                const row = document.createElement("div")
                row.className = "panel-row"
                row.textContent = `${entry.index}: ${entry.label}`
                panel.appendChild(row)
            }
            this._overlay.appendChild(panel)
            this._categoryEls["innovations"] = panel
        }
    }

    private _updateInstructions(): void {
        if (this._instructionsEl != null) {
            this._instructionsEl.textContent = this.model.instructions
        }
    }

    private _rebuildAxes(): void {
        if (this._scene == null)
            return
        if (this._axisGroup != null) {
            this._scene.remove(this._axisGroup)
            this._axisGroup = null
        }

        const limits = this.model.axis_limits
        const xLimits = limits["x"] ?? null
        const yLimits = limits["y"] ?? null
        const zLimits = limits["z"] ?? null
        if (xLimits == null || yLimits == null || zLimits == null)
            return

        const origin = new THREE.Vector3(xLimits[0], yLimits[0], zLimits[0])
        const xEnd = new THREE.Vector3(xLimits[1], yLimits[0], zLimits[0])
        const yEnd = new THREE.Vector3(xLimits[0], yLimits[1], zLimits[0])
        const zEnd = new THREE.Vector3(xLimits[0], yLimits[0], zLimits[1])

        const makeLine = (start: any, end: any, color: number) => {
            const geometry = new THREE.BufferGeometry().setFromPoints([start, end])
            const material = new THREE.LineBasicMaterial({color, linewidth: 1.5})
            return new THREE.Line(geometry, material)
        }

        const axisGroup = new THREE.Group()
        axisGroup.add(makeLine(origin, xEnd, 0x5dade2))
        axisGroup.add(makeLine(origin, yEnd, 0x58d68d))
        axisGroup.add(makeLine(origin, zEnd, 0xf4d03f))

        this._scene.add(axisGroup)
        this._axisGroup = axisGroup
        this._axisVectors = {origin, x: xEnd, y: yEnd, z: zEnd}

        this._positionCamera(xLimits, yLimits, zLimits)
        this._updateAxisLabels()
        this._updateOverlayPositions()
    }

    private _positionCamera(xLimits: number[], yLimits: number[], zLimits: number[]): void {
        if (this._camera == null || this._controls == null)
            return
        const center = new THREE.Vector3(
            (xLimits[0] + xLimits[1]) / 2,
            (yLimits[0] + yLimits[1]) / 2,
            (zLimits[0] + zLimits[1]) / 2,
        )
        const spanX = Math.max(1, Math.abs(xLimits[1] - xLimits[0]))
        const spanY = Math.max(1, Math.abs(yLimits[1] - yLimits[0]))
        const spanZ = Math.max(1, Math.abs(zLimits[1] - zLimits[0]))
        const maxSpan = Math.max(spanX, spanY, spanZ)

        this._camera.position.set(
            center.x + maxSpan * 1.6,
            center.y + maxSpan * 1.15,
            center.z + maxSpan * 1.8,
        )
        this._controls.target.copy(center)
        this._controls.update()
    }

    private _updateAxisLabels(): void {
        const labels = this.model.axis_labels
        if (this._axisLabels.x != null) {
            this._axisLabels.x.textContent = labels.x ?? "Time"
        }
        if (this._axisLabels.y != null) {
            this._axisLabels.y.textContent = labels.y ?? "Model"
        }
        if (this._axisLabels.z != null) {
            this._axisLabels.z.textContent = labels.z ?? "Innovation"
        }
        this._updateOverlayPositions()
    }

    private _updateOverlayPositions(): void {
        if (
            this._renderer == null ||
            this._camera == null ||
            this._axisVectors == null
        ) {
            return
        }
        const width = this._renderer.domElement.clientWidth
        const height = this._renderer.domElement.clientHeight
        const project = (vector: any) => {
            const projected = vector.clone().project(this._camera)
            return {
                x: (projected.x + 1) / 2 * width,
                y: (-projected.y + 1) / 2 * height,
            }
        }

        const xPos = project(this._axisVectors.x)
        if (this._axisLabels.x != null) {
            this._axisLabels.x.style.left = `${xPos.x}px`
            this._axisLabels.x.style.top = `${xPos.y}px`
        }
        const yPos = project(this._axisVectors.y)
        if (this._axisLabels.y != null) {
            this._axisLabels.y.style.left = `${yPos.x}px`
            this._axisLabels.y.style.top = `${yPos.y}px`
        }
        const zPos = project(this._axisVectors.z)
        if (this._axisLabels.z != null) {
            this._axisLabels.z.style.left = `${zPos.x}px`
            this._axisLabels.z.style.top = `${zPos.y}px`
        }
    }

    private _updateBackground(): void {
        if (this._scene != null) {
            this._scene.background = new THREE.Color(this.model.background_color)
        }
    }

    private _updatePoints(): void {
        if (this._scene == null)
            return
        if (this._points != null) {
            this._scene.remove(this._points)
            this._points.geometry.dispose()
            this._points.material.dispose()
            this._points = null
        }

        const data = this.model.data as any
        const xs: number[] = data.x ?? []
        const ys: number[] = data.y ?? []
        const zs: number[] = data.z ?? []
        if (xs.length === 0)
            return

        const colors: string[] = data.color ?? []

        const positions = new Float32Array(xs.length * 3)
        const colorValues = new Float32Array(xs.length * 3)
        for (let i = 0; i < xs.length; i++) {
            positions[i * 3] = xs[i]
            positions[i * 3 + 1] = ys[i]
            positions[i * 3 + 2] = zs[i]

            const color = new THREE.Color(colors[i] ?? "#9fa8da")
            colorValues[i * 3] = color.r
            colorValues[i * 3 + 1] = color.g
            colorValues[i * 3 + 2] = color.b
        }

        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))
        geometry.setAttribute("color", new THREE.Float32BufferAttribute(colorValues, 3))

        const material = new THREE.PointsMaterial({
            size: this.model.point_size,
            vertexColors: true,
            sizeAttenuation: true,
        })

        this._points = new THREE.Points(geometry, material)
        this._scene.add(this._points)
    }

    private _updatePointSize(): void {
        if (this._points != null) {
            this._points.material.size = this.model.point_size
        }
    }

    private _updateEdges(): void {
        if (this._scene == null)
            return
        if (this._edges != null) {
            this._scene.remove(this._edges)
            this._edges.geometry.dispose()
            this._edges.material.dispose()
            this._edges = null
        }

        const edges = this.model.edges
        if (edges.length === 0)
            return

        const positions = new Float32Array(edges.length * 6)
        edges.forEach((edge: any, index: number) => {
            positions[index * 6] = edge.x0
            positions[index * 6 + 1] = edge.y0
            positions[index * 6 + 2] = edge.z0
            positions[index * 6 + 3] = edge.x1
            positions[index * 6 + 4] = edge.y1
            positions[index * 6 + 5] = edge.z1
        })

        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3))

        const material = new THREE.LineBasicMaterial({color: 0x888888, transparent: true, opacity: 0.35})
        this._edges = new THREE.LineSegments(geometry, material)
        this._scene.add(this._edges)
    }

    private _onPointerMove(event: MouseEvent): void {
        if (this._renderer == null || this._camera == null || this._points == null)
            return
        const rect = this._renderer.domElement.getBoundingClientRect()
        const x = (event.clientX - rect.left) / rect.width
        const y = (event.clientY - rect.top) / rect.height
        this._mouse.x = x * 2 - 1
        this._mouse.y = -(y * 2 - 1)
        this._raycaster.setFromCamera(this._mouse, this._camera)
        const intersections = this._raycaster.intersectObject(this._points)
        if (intersections.length > 0) {
            const index = intersections[0].index ?? 0
            this._showTooltip(index, event)
        } else {
            this._hideTooltip()
        }
    }

    private _showTooltip(index: number, event: MouseEvent): void {
        if (this._tooltip == null || this._renderer == null)
            return
        const data = this.model.data as any
        const name = data.name?.[index] ?? ""
        const family = data.family?.[index] ?? ""
        const release = data.release?.[index] ?? ""
        const innovation = data.innovation?.[index] ?? ""
        const influences = data.influences?.[index] ?? "None"

        this._tooltip.innerHTML = `
            <div class="tooltip-title">${name}</div>
            <div class="tooltip-row"><span>Family</span><span>${family}</span></div>
            <div class="tooltip-row"><span>Released</span><span>${release}</span></div>
            <div class="tooltip-row"><span>Innovation</span><span>${innovation}</span></div>
            <div class="tooltip-row"><span>Influences</span><span>${influences}</span></div>
        `
        const rect = this._renderer.domElement.getBoundingClientRect()
        const left = event.clientX - rect.left + 14
        const top = event.clientY - rect.top + 14
        this._tooltip.style.left = `${left}px`
        this._tooltip.style.top = `${top}px`
        this._tooltip.style.display = "block"
    }

    private _hideTooltip(): void {
        if (this._tooltip != null) {
            this._tooltip.style.display = "none"
        }
    }

    private _resize(): void {
        if (this._renderer == null || this._camera == null || this._container == null)
            return
        const width = this._container.clientWidth || 800
        const height = this._container.clientHeight || 600
        this._renderer.setSize(width, height, false)
        this._camera.aspect = width / height
        this._camera.updateProjectionMatrix()
        this._updateOverlayPositions()
    }

    private _startAnimationLoop(): void {
        if (this._animationHandle != null)
            return
        const renderFrame = () => {
            this._animationHandle = requestAnimationFrame(renderFrame)
            if (this._controls != null)
                this._controls.update()
            if (this._renderer != null && this._scene != null && this._camera != null)
                this._renderer.render(this._scene, this._camera)
            this._updateOverlayPositions()
        }
        renderFrame()
    }
}

export namespace ThreeDScatter {
    export type Attrs = p.AttrsOf<Props>
    export type Props = LayoutDOM.Props & {
        data: p.Property<Record<string, unknown[]>>
        edges: p.Property<Record<string, unknown>[]>
        axis_labels: p.Property<Record<string, string>>
        axis_limits: p.Property<Record<string, number[]>>
        categories: p.Property<Record<string, unknown[]>>
        legend_items: p.Property<Record<string, unknown>[]>
        point_size: p.Property<number>
        background_color: p.Property<string>
        instructions: p.Property<string>
    }
}

export interface ThreeDScatter extends ThreeDScatter.Attrs {}

export class ThreeDScatter extends LayoutDOM {
    declare properties: ThreeDScatter.Props
    declare __view_type__: ThreeDScatterView

    static override __module__ = "app.llm_phylogeny"

    static {
        this.prototype.default_view = ThreeDScatterView
        this.define<ThreeDScatter.Props>(({Dict, List, Float, String, Any}) => ({
            data: [Dict(String, List(Any)), {}],
            edges: [List(Dict(String, Any)), []],
            axis_labels: [Dict(String, String), {}],
            axis_limits: [Dict(String, List(Float)), {}],
            categories: [Dict(String, List(Any)), {}],
            legend_items: [List(Dict(String, Any)), []],
            point_size: [Float, 18],
            background_color: [String, "#080b12"],
            instructions: [String, ""],
        }))
    }
}
        """
    )

    data = BkDict(BkString, BkList(BkAny), default=dict)
    edges = BkList(BkDict(BkString, BkAny), default=list)
    axis_labels = BkDict(BkString, BkString, default=dict)
    axis_limits = BkDict(BkString, BkList(BkFloat), default=dict)
    categories = BkDict(BkString, BkList(BkAny), default=dict)
    legend_items = BkList(BkDict(BkString, BkAny), default=list)
    point_size = BkFloat(default=18.0)
    background_color = BkString(default="#080b12")
    instructions = BkString(default="")


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
) -> ThreeDScatter:
    """Create the interactive 3D Bokeh figure from prepared components."""

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

    scatter = ThreeDScatter(
        width=1200,
        height=800,
        data=node_data,
        edges=edge_pairs,
        axis_labels=axis_labels,
        axis_limits=axis_limits,
        categories=category_data,
        legend_items=legend_items,
        instructions=instructions,
        point_size=16.0,
        background_color="#05070d",
    )

    return scatter


def build_plot(*, data_path: Path | None = None):
    """Construct the interactive Bokeh plot for the phylogenetic graph."""

    models, graph, layout, color_map = _prepare_visualisation(data_path=data_path)
    return _construct_bokeh_figure(models, graph, layout, color_map)


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
    plot = _construct_bokeh_figure(models, graph, layout, color_map)
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
