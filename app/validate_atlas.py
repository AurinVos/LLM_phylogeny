"""Validate the self-contained atlas and its embedded, source-linked data.

No third-party dependencies. Run from any working directory:
    python app/validate_atlas.py [path/to/index.html]
"""
from __future__ import annotations

import argparse
import datetime as dt
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlparse

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "docs" / "index.html"


class AtlasParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.collecting = False
        self.blocks: list[str] = []
        self.external_resources: list[str] = []
        self.count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and values.get("id") == "atlas-data":
            if values.get("type") != "application/json":
                raise ValueError("atlas-data must be application/json")
            self.collecting = True
            self.count += 1
        if tag in {"script", "img", "iframe"} and values.get("src"):
            self.external_resources.append(str(values["src"]))
        if tag == "link" and values.get("rel") == "stylesheet":
            self.external_resources.append(str(values.get("href")))

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.collecting = False

    def handle_data(self, data: str) -> None:
        if self.collecting:
            self.blocks.append(data)


def load_atlas(path: Path = DEFAULT_PATH) -> dict:
    parser = AtlasParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if parser.count != 1:
        raise ValueError("Expected exactly one embedded atlas-data block")
    if parser.external_resources:
        raise ValueError(f"Runtime resources must be embedded: {parser.external_resources}")
    data = json.loads("".join(parser.blocks))
    if not isinstance(data, dict):
        raise ValueError("Atlas data must be an object")
    return data


def validate(data: dict) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(data.get("schema_version") == 1, "Unsupported schema version")
    reviewed = dt.date.fromisoformat(data["reviewed"])
    coverage = dt.date.fromisoformat(data["coverage_end"])
    require(coverage <= reviewed, "Coverage exceeds review date")
    require(bool(data.get("scope")), "Scope statement is required")
    ids: dict[str, set[str]] = {}
    for key in ("sources", "milestones", "capabilities", "activities", "patterns"):
        rows = data[key]
        require(isinstance(rows, list) and bool(rows), f"Empty or invalid {key}")
        values = [r["id"] for r in rows]
        require(len(values) == len(set(values)), f"Duplicate IDs in {key}")
        require(all(re.fullmatch(r"[a-z0-9_-]+", v) for v in values), f"Unsafe ID in {key}")
        ids[key] = set(values)

    for s in data["sources"]:
        u = urlparse(s["url"])
        require(u.scheme == "https" and bool(u.hostname) and not u.username, "Unsafe source URL")
        require(dt.date.fromisoformat(s["date"]) <= reviewed, "Source date exceeds review date")
        require(bool(s["title"]) and bool(s["author"]), "Missing source attribution")

    def refs(row: dict, field: str, collection: str) -> None:
        values = row[field]
        require(bool(values), f"Missing {field} for {row['id']}")
        require(set(values) <= ids[collection], f"Unknown {field} for {row['id']}")
        require(len(values) == len(set(values)), f"Repeated {field} for {row['id']}")

    kinds = {"model", "method", "product", "protocol", "system"}
    lanes = {"language", "evidence", "reasoning", "tools", "coordination"}
    date_types = {"Paper submission", "Product announcement", "Model announcement", "Protocol announcement", "Engineering publication"}
    for m in data["milestones"]:
        require(m["kind"] in kinds, f"Unknown milestone type: {m['id']}")
        require(m["lane"] in lanes, f"Unknown milestone lane: {m['id']}")
        require(m["date_type"] in date_types, f"Missing/unknown date meaning: {m['id']}")
        date = dt.date.fromisoformat(m["date"])
        require(dt.date(2017, 1, 1) <= date <= coverage, f"Milestone outside declared coverage: {m['id']}")
        require(bool(m["change"]) and bool(m["limit"]), f"Unqualified claim: {m['id']}")
        refs(m, "sources", "sources")
        refs(m, "activities", "activities")
    require(max(m["date"] for m in data["milestones"]) == data["coverage_end"], "Coverage end is stale")
    for c in data["capabilities"]:
        require(c["kind"] in {"model", "system", "check"}, f"Unknown capability kind: {c['id']}")
        require(bool(c["description"]) and bool(c["limit"]), f"Unqualified capability: {c['id']}")
        refs(c, "sources", "sources")
    for a in data["activities"]:
        refs(a, "capabilities", "capabilities")
        require(len(a["capabilities"]) == 2, "Current map layout requires two example ingredients per activity")
        require(bool(re.fullmatch(r"#[0-9a-fA-F]{6}", a["color"])), "Invalid activity color")
        for key in ("title", "description", "scientific", "general", "prompt_scientific", "prompt_general", "output", "human_check"):
            require(bool(a[key]), f"Missing {key} for {a['id']}")
    for p in data["patterns"]:
        refs(p, "sources", "sources")
        require(bool(p["human_check"]) and bool(p["controller"]), "Working patterns need control and review descriptions")
    require(len(data["activities"]) == 6 and len(data["patterns"]) == 5, "Layout cardinality changed; update diagrams and tests together")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    try:
        data = load_atlas(args.path)
        validate(data)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Atlas validation failed: {exc}\n")
    print(f"Atlas valid: {len(data['milestones'])} milestones, {len(data['sources'])} sources, {len(data['activities'])} activities.")


if __name__ == "__main__":
    main()
