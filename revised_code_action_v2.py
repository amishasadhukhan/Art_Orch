# revised_code_action.py
#
# Lightweight entry point: raw query → subgoals → Krita Scripter code.
#
# Unlike process_generator_refined.py (10-stage pipeline) this does the
# query → steps breakdown in a single qwen2.5:7b call, thinking like a
# painter working background-to-foreground. Self-contained — does not
# import from step_code_generator.py.
#
# No vision model, no canvas screenshots, no analysis/correction pass —
# pure text: query -> subgoals -> code.
#
# Usage:
#   python revised_code_action.py "Create a detailed underdrawing with a
#   light pencil, outlining the cafe terrace..." [--canvas=900x600] [--cloud]

import json
import math
import os
import re
import sys

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# ─── Config ───────────────────────────────────────────────────────────────────

LOCAL_MODEL = "qwen2.5:7b"      # text-only, no vision
CLOUD_MODEL = "gpt-oss:120b-cloud"   # Ollama Cloud requires the "-cloud" suffix on hosted model tags
CLOUD_HOST  = "https://ollama.com"

EMBED_MODEL = "nomic-embed-text"   # Ollama embedding model
RAG_TOP_K   = 12                   # how many method docs to retrieve per step

PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "krita_presets.json")


def _load_brush_presets(path: str = PRESETS_PATH) -> str:
    """Load krita_presets.json (name + description per preset) and format as a
    compact reference list for the prompt — real preset names/descriptions,
    not a handful of hardcoded guesses."""
    if not os.path.isfile(path):
        print("  krita_presets.json not found — brush guidance disabled.")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        presets = json.load(f)
    return "\n".join(f"{p['name']} — {p['description']}" for p in presets)


def _validate_brush_presets(code: str, brush_list: str) -> str:
    """Check every resources("preset").get("...") call in generated code against the
    real preset names. The LLM sometimes invents a shortened name (e.g. "Basic-1"
    instead of "b) Basic-1"), which makes setCurrentBrushPreset() silently no-op and
    leave a leftover brush from an earlier script active. Auto-correct when there's
    an unambiguous match; otherwise print a visible warning so it isn't missed."""
    known_names = [line.split(" — ", 1)[0].strip() for line in brush_list.splitlines() if " — " in line]

    def _fix(match):
        used = match.group(1)
        if used in known_names:
            return match.group(0)
        if " — " in used:
            prefix = used.split(" — ", 1)[0].strip()
            if prefix in known_names:
                print(f"  brush preset name included its description — trimmed to '{prefix}'")
                return match.group(0).replace(used, prefix)
        candidates = [n for n in known_names if n.endswith(used) or n.split(") ", 1)[-1] == used]
        if len(candidates) == 1:
            print(f"  brush preset '{used}' not found — auto-corrected to '{candidates[0]}'")
            return match.group(0).replace(used, candidates[0])
        print(f"  WARNING: brush preset '{used}' not found in krita_presets.json "
              f"and no unambiguous match — setCurrentBrushPreset() will silently no-op for this line.")
        return match.group(0)

    return re.sub(r'resources\("preset"\)\.get\("([^"]+)"\)', _fix, code)


def _fix_krita_class_calls(code: str) -> str:
    """Auto-fix a recurring LLM mistake: calling Krita.resources(...) directly
    on the class instead of on the Krita instance. This raises a TypeError at
    runtime ("first argument of unbound method must have type 'Krita'") —
    the prompt rule alone doesn't reliably stop the model from doing this, so
    fix it deterministically instead of just warning about it."""
    fixed = re.sub(r'\bKrita\.resources\(', 'Krita.instance().resources(', code)
    if fixed != code:
        print("  Auto-fixed: Krita.resources(...) -> Krita.instance().resources(...)")
    return fixed


BRUSH_RAG_TOP_K = 5   # candidate presets to surface per shape — a tight shortlist, not another long list


def _make_brush_retriever(path: str = PRESETS_PATH):
    """Embed each preset's name+description and build a FAISS retriever, so a
    shape's own description/color/placement can surface the closest-matching
    presets (e.g. "z) Stamp Grass" for a grass patch) instead of the model
    having to scan all 143 names unassisted every single time."""
    from langchain_community.vectorstores import FAISS

    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        presets = json.load(f)
    if not presets:
        return None

    docs = [
        Document(page_content=f"{p['name']} — {p['description']}", metadata={"name": p["name"]})
        for p in presets
    ]
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vs = FAISS.from_documents(docs, embeddings)
        return vs.as_retriever(search_kwargs={"k": BRUSH_RAG_TOP_K})
    except Exception as e:
        print(f"  Brush embedding index build failed ({e}) — falling back to full preset list.")
        return None


def retrieve_brushes(retriever, step: dict, full_list: str) -> str:
    """Retrieve the closest-matching presets per shape (using that shape's own
    description/color/placement), instead of dumping all 143 presets into every
    prompt. Falls back to the full list if the retriever is unavailable."""
    if retriever is None:
        return full_list

    seen_names = set()
    all_lines = []
    try:
        for shape in step.get("shapes", []):
            query = f"{shape.get('description', '')} {shape.get('color', '')} {shape.get('placement', '')}"
            docs = retriever.invoke(query)
            for d in docs:
                name = d.metadata.get("name", d.page_content)
                if name not in seen_names:
                    seen_names.add(name)
                    all_lines.append(d.page_content)
        return "\n".join(all_lines) if all_lines else full_list
    except Exception as e:
        print(f"  Brush retrieval failed ({e}) — falling back to full preset list.")
        return full_list


def _make_llm(cloud: bool = False):
    if cloud:
        api_key = os.environ.get("OLLAMA_API_KEY", "")
        if not api_key:
            raise ValueError("Set OLLAMA_API_KEY for cloud mode.")
        return ChatOllama(
            model=CLOUD_MODEL, temperature=0.3,
            base_url=CLOUD_HOST,
            client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
        )
    return ChatOllama(model=LOCAL_MODEL, temperature=0.3)


# ─── JSON helper ──────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract valid JSON:\n{raw[:800]}")


# ─── Krita API RAG (own copy — indexes libkis_headers/) ──────────────────────

_KRITA_QUIRKS = """\
# CRITICAL KRITA QUIRKS (always apply):

# 1. ManagedColor RGBA/U8 setComponents() order is [B, G, R, A] — NOT RGB order.
#    Always use this helper:
#    def set_color(hex_str):
#        qc = QColor(hex_str)
#        mc = ManagedColor("RGBA", "U8", "")
#        mc.setComponents([qc.blueF(), qc.greenF(), qc.redF(), 1.0])
#        view.setForeGroundColor(mc)

# 2. Always call doc.refreshProjection() at the very end of every script.

# 3. Layer creation pattern:
#    node = doc.createNode("name", "paintlayer")
#    doc.rootNode().addChildNode(node, None)   # None = add on top
#    doc.setActiveNode(node)

# 4. paintRectangle/paintEllipse/paintPolygon/paintPath all take TWO style
#    arguments, always in this order: (strokeStyle, fillStyle)
#      strokeStyle = how the BORDER/EDGE is drawn
#      fillStyle   = how the INTERIOR is drawn
#    Each can independently be: "ForegroundColor" | "BackgroundColor" | "None"
#    (fillStyle also accepts "Pattern").
#    Border AND interior filled -> paintRectangle(rect, "ForegroundColor", "ForegroundColor")
#    Border only, hollow inside -> paintRectangle(rect, "ForegroundColor", "None")
#    Interior filled, no border -> paintRectangle(rect, "None", "ForegroundColor")
#    (paintEllipse/paintPolygon/paintPath take the same two arguments in the same order.)
#
#    paintLine is DIFFERENT — it only takes strokeStyle, no fillStyle at all
#    (a line has no interior to fill). Quirk: even strokeStyle="None" still
#    draws the line in ForegroundColor — Krita has no invisible-line option.

# 5. Pick brush presets by name: ki.resources("preset").get("b) Basic-5 Size")
"""


def _parse_headers_to_docs(headers_dir: str) -> list:
    """Extract one Document per method block (docstring + signature) from all headers."""
    files = sorted(f for f in os.listdir(headers_dir) if f.endswith(".h"))
    docs = []
    for fname in files:
        path = os.path.join(headers_dir, fname)
        raw   = open(path, "r", encoding="utf-8").read()
        klass = fname.replace(".h", "")

        blocks = re.findall(
            r'(/\*\*.*?\*/)\s*\n\s*([A-Za-z][^\n;{]*[;{])',
            raw, re.DOTALL
        )
        for doc_raw, sig in blocks:
            doc_txt = re.sub(r'\n\s*\*\s?', ' ', doc_raw)
            doc_txt = re.sub(r'/\*\*\s*', '', doc_txt)
            doc_txt = re.sub(r'\s*\*/', '', doc_txt).strip()
            sig_clean = sig.strip().rstrip('{').strip()

            content = f"[{klass}] {sig_clean}\n# {doc_txt[:300]}"
            docs.append(Document(page_content=content,
                                 metadata={"class": klass, "sig": sig_clean}))
    return docs


def _build_api_index(headers_dir: str, embed_base_url: str | None = None,
                     embed_api_key: str | None = None):
    """Parse libkis headers, embed with nomic-embed-text, build FAISS retriever.
    Returns None if headers folder not found — falls back to built-in quirks only."""
    from langchain_community.vectorstores import FAISS

    if not os.path.isdir(headers_dir):
        print("  libkis_headers/ not found — RAG disabled, using built-in quirks.")
        return None

    print("  Parsing headers and building FAISS index...")
    docs = _parse_headers_to_docs(headers_dir)
    if not docs:
        print("  No method blocks parsed — RAG disabled.")
        return None
    print(f"  {len(docs)} method blocks indexed.")

    embed_kwargs = {"model": EMBED_MODEL}
    if embed_base_url:
        embed_kwargs["base_url"] = embed_base_url
    if embed_api_key:
        embed_kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {embed_api_key}"}}

    try:
        embeddings = OllamaEmbeddings(**embed_kwargs)
        vs = FAISS.from_documents(docs, embeddings)
        return vs.as_retriever(search_kwargs={"k": RAG_TOP_K})
    except Exception as e:
        # Cloud endpoints often don't serve embedding models (only large chat
        # models) — degrade to quirks-only instead of crashing the whole run.
        print(f"  Embedding index build failed ({e}) — RAG disabled, using built-in quirks.")
        return None


def _make_retriever():
    """Embeddings always run against a LOCAL Ollama server — Ollama Cloud's
    hosted API serves large chat models only, not embedding models. If you're
    using cloud=True for the chat/code-gen model, you still need a local
    Ollama running with `ollama pull nomic-embed-text` for RAG to work."""
    headers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkis_headers")
    return _build_api_index(headers_dir)


def retrieve_api(retriever, step: dict) -> str:
    """Retrieve relevant API methods for this step — one query per unique shape type,
    built from that type's own shapes (placement + color), not just the shared step-level
    action/description, so different shape types in the same step get distinctly targeted queries."""
    if retriever is None:
        return _KRITA_QUIRKS

    shapes      = step.get("shapes", [])
    action      = step.get("action", "")
    description = step.get("description", "")

    unique_types = list(dict.fromkeys(s.get("type", "") for s in shapes))
    k_per_type   = max(3, RAG_TOP_K // len(unique_types)) if unique_types else RAG_TOP_K

    seen_sigs = set()
    all_methods = []

    try:
        for shape_type in unique_types:
            type_shapes = [s for s in shapes if s.get("type") == shape_type]
            placements  = " ".join(s.get("placement", "") for s in type_shapes)
            colors      = " ".join(s.get("color", "") for s in type_shapes)
            query = f"{action} {shape_type} {description} {placements} {colors}"
            docs  = retriever.invoke(query)[:k_per_type]
            for d in docs:
                sig = d.metadata.get("sig", d.page_content[:60])
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    all_methods.append(d.page_content)

        methods = "\n\n".join(all_methods)
        print(f"  RAG: {len(unique_types)} shape type(s) → {len(all_methods)} API methods retrieved")
        return _KRITA_QUIRKS + "\n\n# ── Retrieved API methods for this step ──\n" + methods

    except Exception as e:
        print(f"  RAG retrieval failed ({e}) — using quirks only.")
        return _KRITA_QUIRKS


# ─── Subgoal decomposition ────────────────────────────────────────────────────

_SUBGOAL_SYSTEM = """You are an experienced painter planning a canvas before touching it.

Given a painting query, break it into an ordered list of subgoals — the way a
painter actually works: background to foreground, big shapes before detail,
underdrawing/structure before refinement.

The canvas size is fixed and given to you below — do not restate it in your output.

Output ONLY valid JSON, no markdown, no explanation, in this exact shape:

{
  "steps": [
    {
      "step": <int, 1-based order>,
      "name": "<short_snake_case_id, e.g. cobblestone_street>",
      "layer": "<layer group this belongs to, e.g. background/midground/foreground>",
      "description": "<what this subgoal covers>",
      "bbox": {"x": <int>, "y": <int>, "w": <int>, "h": <int>},
      "shapes": [
        {
          "type": "stroke | path_filled | path_outline |rect_filled | rect_outline | ellipse_filled | ellipse_outline | polygon | polygon_outline ",
          "role": "<short name for this shape>",
          "placement": "<where within the bbox and how it's arranged, plain English>",
          "color": "<color description or hex>",
          "description": "<what this shape represents>",
          "fill_style": "flat | textured  (ONLY for _filled shape types: rect_filled, ellipse_filled, polygon, path_filled — omit or ignore for outline/stroke types)"
        }
      ],
      "action": "<imperative instruction describing how to paint this subgoal>"
    }
  ]
}

RULES:
- step must be sequential starting at 1, matching the order below.
- name must be unique per step and safe as a Krita layer name (snake_case, no spaces).
- bbox coordinates are pixels within the given canvas w/h — stay in bounds.
- Order steps as a painter would: distant/background elements first, then
  midground, then foreground, then small detail/accent elements last.
- Each step should be a coherent, paintable chunk (one element or tightly
  related group of elements) — not too broad, not overly fragmented.
- shapes should reflect how you'd actually construct that element with basic
  primitives (strokes, rectangles, ellipses, polygon) — 1 to 5 shapes per step.
- Use path_filled / path_outline instead of polygon for anything organic or
  curved — tree canopies, drapery, awning scallops, silhouettes, rolling
  terrain — where straight polygon edges would look wrong. Use polygon only
  for genuinely angular shapes (buildings, boxes, roofs).
- fill_style (only on _filled shapes): set to "textured" whenever the element
  should show visible brush character rather than a clean flat color — washes,
  grain, weathering, roughness, dry-brush, watercolor bleed, chalky/grainy
  surfaces, or anything the query itself describes as painterly/textured.
  Set to "flat" for clean solid-color regions with no implied texture (a
  simple wall, a plain shape, a solid background block). Decide this per
  shape based on what it actually represents, don't default to one value.
"""


def generate_subgoals(query: str, llm, canvas_w: int, canvas_h: int) -> dict:
    """Single-shot query -> ordered painter subgoals."""
    msgs = [
        SystemMessage(content=_SUBGOAL_SYSTEM),
        HumanMessage(content=(
            f"Canvas size (fixed, do not restate): {canvas_w}x{canvas_h}\n\n"
            f"Painting query:\n{query}"
        )),
    ]
    response = llm.invoke(msgs)
    raw = response.content if hasattr(response, "content") else str(response)
    process = _parse_json(raw)

    process.setdefault("steps", [])
    return process


# ─── Pass 1: Coordinate Generator ────────────────────────────────────────────

_COORD_SYSTEM = """You are a geometry planner for digital painting.

You receive a painting step with a pixel bounding box and a list of shapes.
For each shape, write Python code that computes PRECISE coordinates based on
the placement description and bbox, and stores them in a dict called 'result'.

OUTPUT FORMAT — assign to 'result' like this:
  result = {
    "role_name": <coords>
  }

Where <coords> depends on shape type:
  rect_filled / rect_outline / ellipse_filled / ellipse_outline:
      {"x": int, "y": int, "w": int, "h": int}

  stroke:
      list of (x1, y1, x2, y2) tuples — one tuple per individual stroke mark.
      Generate as many as needed to fill the described area properly.
      Think about: density, angle, spacing, length based on the placement text.

  polygon / polygon_outline:
      list of (x, y) tuples tracing the outline — enough points to define the shape.
      Think about the element's visual profile from the placement description.

  path_filled / path_outline:
      list of (x, y) anchor points tracing the curve's silhouette — same as
      polygon, but these points will be smoothed into a curved outline
      afterward, so space them along the visual profile (bumps, scallops,
      canopy lobes etc.) rather than at sharp corners.

RULES:
- Output ONLY Python code. No imports needed. No markdown.
- Use only: int, float, list, range, math (no external libs).
- The code must produce 'result = {...}' at the end.
- Stay strictly within the bbox boundaries.
- Read the placement description carefully — it describes WHERE and HOW.
  e.g. "diagonal strokes scattered across full area" → many short diagonal (x,y)→(x+15,y+15) lines
  e.g. "bottom 30% seat" → y starts at bbox.y + bbox.h * 0.7
 """


def _catmull_rom_to_bezier(points: list, closed: bool = False) -> list:
    """Convert (x, y) anchor points into cubic Bezier segments via Catmull-Rom.
    Returns a list of (p0, c1, c2, p1) tuples — start point, two control
    points, end point — ready for QPainterPath.cubicTo(). Deterministic:
    the LLM only picks anchor points, this always produces a valid smooth curve."""
    pts = [tuple(p) for p in points]
    n = len(pts)
    if n < 2:
        return []

    def get(i):
        if closed:
            return pts[i % n]
        return pts[max(0, min(n - 1, i))]

    segments = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0, p1, p2, p3 = get(i - 1), get(i), get(i + 1), get(i + 2)
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        segments.append((p1, c1, c2, p2))
    return segments


def generate_coords(step: dict, llm) -> dict:
    """Pass 1: LLM writes Python code that computes coordinates. We exec() it and capture 'result'.
    path_filled/path_outline anchor points are then smoothed into Bezier segments in Python —
    curve shape stays deterministic, the LLM never has to produce control points itself."""
    step_info = json.dumps({
        "bbox":   step.get("bbox"),
        "shapes": step.get("shapes", []),
        "description": step.get("description"),
    }, indent=2)

    msgs = [
        SystemMessage(content=_COORD_SYSTEM),
        HumanMessage(content=f"Compute coordinates for all shapes in this step:\n\n{step_info}"),
    ]
    response = llm.invoke(msgs)
    code = response.content if hasattr(response, "content") else str(response)
    code = code.replace("```python", "").replace("```", "").strip()

    namespace = {"result": {}, "math": math}
    try:
        exec(code, namespace)
        coords = namespace.get("result", {})
    except Exception as e:
        print(f"  Coord exec failed ({e}) — coords left to LLM in Pass 2.")
        return {}

    path_roles = {
        s.get("role"): s.get("type")
        for s in step.get("shapes", [])
        if s.get("type") in ("path_filled", "path_outline")
    }
    for role, shape_type in path_roles.items():
        anchors = coords.get(role)
        if isinstance(anchors, list) and len(anchors) >= 2:
            coords[role] = _catmull_rom_to_bezier(anchors, closed=(shape_type == "path_filled"))

    print(f"  Coords resolved: { {k: f'{len(v)} points' if isinstance(v, list) else v for k, v in coords.items()} }")
    return coords


# ─── Pass 2: Code Generator ───────────────────────────────────────────────────

_SYSTEM = """You are an expert Krita Python scripter for digital painting.

You receive one painting step at a time. Each step has: name, layer,
description, bbox (pixel bounding box), shapes (type/role/placement/color),
and action text.

Your job: write complete, runnable Python code for Krita's Scripter that paints this step.

Each shape in the step already has a "type" field — map it directly to the Krita API call:
  rect_filled    → node.paintRectangle(QRectF(x,y,w,h), "ForegroundColor", "ForegroundColor")
  rect_outline   → node.paintRectangle(QRectF(x,y,w,h), "ForegroundColor", "None")
  ellipse_filled → node.paintEllipse(QRectF(x,y,w,h),   "ForegroundColor", "ForegroundColor")
  ellipse_outline→ node.paintEllipse(QRectF(x,y,w,h),   "ForegroundColor", "None")
  polygon         → node.paintPolygon([QPointF(x,y) for x,y in points], "ForegroundColor", "ForegroundColor")  # filled solid (silhouette, mass, terrain)
  polygon_outline → node.paintPolygon([QPointF(x,y) for x,y in points], "ForegroundColor", "None")             # outline only (border, frame, edge)
  stroke         → for x1,y1,x2,y2 in stroke_list:
                       node.paintLine(QPoint(int(x1),int(y1)), QPoint(int(x2),int(y2)), 0.9, 0.3, "ForegroundColor")
                       # paintLine needs QPoint (int), NOT QPointF — unlike every other paint* call
  path_filled     → smooth CLOSED curve — segments already pre-computed as (p0,c1,c2,p1) Bezier tuples:
                       path = QPainterPath()
                       path.moveTo(QPointF(*segments[0][0]))
                       for p0, c1, c2, p1 in segments:
                           path.cubicTo(QPointF(*c1), QPointF(*c2), QPointF(*p1))
                       path.closeSubpath()
                       node.paintPath(path, "ForegroundColor", "ForegroundColor")
  path_outline    → same construction as path_filled but do NOT closeSubpath and fill "None":
                       node.paintPath(path, "ForegroundColor", "None")

The relevant Krita API details for this step are provided in the user message.

IMPORTANT RULES:
1. Output ONLY Python code — no markdown fences, no explanation.
2. Always start with EXACTLY these imports (correct modules — do not guess):
     from krita import Krita, ManagedColor
     from PyQt5.QtCore import QRectF, QPointF, QPoint
     from PyQt5.QtGui import QColor, QPainterPath
   NOTE: QPainterPath is in QtGui, NOT QtCore — QRectF/QPointF/QPoint are
   QtCore, QColor/QPainterPath are QtGui. Getting this wrong crashes the
   script with an ImportError before anything paints.
   Then: ki/doc/view setup, set_color() helper.
3. Always end with: doc.refreshProjection()
4. ManagedColor RGBA channel order is B, G, R, A — use the set_color() helper.
5. Convert color descriptions to appropriate hex values.
6. COORDINATES: exact coordinates per shape are pre-computed and given to you.
   Use them directly — do NOT invent or approximate coordinates.
   For stroke shapes: the list of (x1,y1,x2,y2) tuples is already computed —
   iterate over it and call paintLine for each tuple, using QPoint(int(x),int(y))
   — NOT QPointF — since paintLine specifically requires integer QPoint.
   For polygon shapes: the list of (x,y) points is already computed —
   pass it as [QPointF(x,y) for x,y in points] to paintPolygon.
   For rect/ellipse: use the given {x,y,w,h} directly in QRectF.
   For path_filled/path_outline: the list is already computed as (p0,c1,c2,p1)
   Bezier segment tuples — build the QPainterPath with moveTo + cubicTo per
   segment exactly as shown above. Do NOT recompute or approximate the curve.
7. LAYER NAMING: the Krita node name MUST be exactly the element "name" field from the step JSON.
   e.g. name="window_1" → doc.createNode("window_1", "paintlayer")
8. Choose the brush preset whose description best matches the element's look
   (texture, softness, edge quality) from the AVAILABLE BRUSH PRESETS list in
   the user message, and set it before painting:
   view.setCurrentBrushPreset(ki.resources("preset").get("<exact preset name>"))
   The AVAILABLE BRUSH PRESETS list is formatted as "name — description" per
   line — copy ONLY the name part BEFORE the " — ", never the description
   text after it, and never the whole line. Use the EXACT preset name as it
   appears, including any category prefix like "b) " or "c) " — do not
   shorten, paraphrase, or invent a name. A wrong/nonexistent name makes
   setCurrentBrushPreset() silently do nothing, leaving whatever brush was
   active from a PREVIOUS script still active (Krita Scripter scripts run in
   the same live session, so brush state persists across separately-run
   scripts). Always call resources() on the Krita INSTANCE (ki or
   Krita.instance()), never on the Krita class directly — Krita.resources(...)
   without .instance() is invalid and will error.
9. ALWAYS call view.setBrushSize(<value>) explicitly, right after setting the
   preset, with a size appropriate to THIS element. Never rely on whatever
   size happens to already be active — it may be leftover from a completely
   different element's script. Use a small size (roughly 1-3) for thin
   outlines, guide lines, and fine detail; a larger size (roughly 5-15) only
   for genuinely broad strokes or fills.
10. FLAT vs TEXTURED FILL — for FILLED shape types (rect_filled,
    ellipse_filled, polygon, path_filled): fillStyle="ForegroundColor" in
    paintRectangle/paintEllipse/paintPolygon/paintPath is ALWAYS a flat raster
    color fill — Krita's brush engine never touches the interior that way, no
    matter which preset is active (only the border/stroke uses the brush).
    Each such shape already has its own "fill_style" field set to "flat" or
    "textured" — use that field directly, do not re-judge it from the
    description yourself:
    a) fill_style == "flat" (or missing) — just use fillStyle="ForegroundColor"
       as shown in the type mapping above. Simple, always solid, brush choice
       doesn't matter here.
    b) fill_style == "textured" — get the brush to actually paint the interior
       by stroking many closely-spaced scanlines across it with paintLine
       (which DOES use the brush engine), instead of relying on fillStyle.
       Include this helper once per script:
         def brush_fill_scanlines(node, points, spacing):
             ys = [p[1] for p in points]
             y0, y1 = min(ys), max(ys)
             n = len(points)
             y = y0
             while y <= y1:
                 xs = []
                 for i in range(n):
                     ax, ay = points[i]
                     bx, by = points[(i + 1) % n]
                     if (ay <= y < by) or (by <= y < ay):
                         t = (y - ay) / (by - ay)
                         xs.append(ax + t * (bx - ax))
                 xs.sort()
                 for i in range(0, len(xs) - 1, 2):
                     node.paintLine(QPoint(int(xs[i]), int(y)), QPoint(int(xs[i + 1]), int(y)), 1.0, 1.0, "ForegroundColor")
                 y += spacing
       Then set the brush preset/size per rules 8-9, and build a point list
       for this shape before calling it:
         rect_filled:    [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
         ellipse_filled: sample ~36 points around the ellipse's perimeter
         polygon:        use its points directly
         path_filled:    build ONE continuous QPainterPath through ALL
                          segments first (moveTo the first segment's p0, then
                          cubicTo for every segment in order, exactly like the
                          normal path_filled construction), THEN flatten that
                          single finished path ONCE:
                            points = [(p.x(), p.y()) for p in path.toFillPolygon()]
                          then call brush_fill_scanlines(node, points, spacing)
                          exactly ONCE on that one flattened list.
       CRITICAL: for path_filled, do NOT loop over the raw segments list
       calling brush_fill_scanlines once per segment — each segment is just
       4 control points of ONE larger curve, not a shape of its own. Treating
       each segment as its own little quad and filling it independently
       produces broken results (stray diagonal bands, spiky slivers) because
       a Bezier segment's control points are not real polygon corners.
       Always build the whole path first, flatten once, fill once.
       spacing should be tied to brush size, e.g. spacing = max(2, brush_size * 0.6),
       so consecutive strokes overlap enough to leave no gaps.

The relevant Krita API methods, the available brush presets, and the
pre-computed coordinates will be in the user message.
"""


def _build_code_messages(step: dict, system: str, api_context: str = "",
                         coords: dict | None = None, brush_list: str = "") -> list:
    step_json = json.dumps({
        k: step.get(k)
        for k in ("name", "layer", "description", "bbox", "shapes", "action")
    }, indent=2)

    coords_block = ""
    if coords:
        coords_block = (
            "\n\n# ── PRE-COMPUTED COORDINATES (use these exactly) ──\n"
            + json.dumps(coords, indent=2)
            + "\n# For strokes: each entry is a list of (x1,y1,x2,y2) tuples → use paintLine per tuple"
            + "\n# For polygons: each entry is a list of (x,y) tuples → use paintPolygon with QPointF list"
            + "\n# For rects/ellipses: {x,y,w,h} dict → use QRectF(x,y,w,h)"
            + "\n# For path_filled/path_outline: each entry is a list of (p0,c1,c2,p1) Bezier "
              "segment tuples → build QPainterPath with moveTo + cubicTo, use paintPath"
        )

    api_block   = f"\n\n# ── RELEVANT KRITA API ──\n{api_context}" if api_context else ""
    brush_block = f"\n\n# ── AVAILABLE BRUSH PRESETS ──\n{brush_list}" if brush_list else ""
    task_text = (
        f"Generate Krita Scripter Python code for this step."
        f"{coords_block}{api_block}{brush_block}\n\nSTEP:\n{step_json}"
    )

    return [SystemMessage(content=system), HumanMessage(content=task_text)]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _divider(title: str):
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)


_DOC_INIT_CODE = """\
from krita import Krita, InfoObject
ki  = Krita.instance()
doc = ki.createDocument({w}, {h}, "Painting", "RGBA", "U8", "", 72.0)
ki.activeWindow().addView(doc)
doc.refreshProjection()
print("Canvas created: {w}x{h}")
"""


# ─── Main loop ────────────────────────────────────────────────────────────────

def run(query: str, canvas_w: int = 900, canvas_h: int = 600, cloud: bool = False):
    llm             = _make_llm(cloud)
    retriever       = _make_retriever()
    brush_list      = _load_brush_presets()
    brush_retriever = _make_brush_retriever()
    print(f"  RAG: {'enabled' if retriever else 'disabled (fallback to built-in quirks)'}")
    print(f"  Brush presets: {len(brush_list.splitlines()) if brush_list else 0} loaded")
    print(f"  Brush RAG: {'enabled' if brush_retriever else 'disabled (full list used every time)'}")

    _divider("Breaking query into subgoals (qwen2.5:7b)")
    process = generate_subgoals(query, llm, canvas_w, canvas_h)
    steps   = process.get("steps", [])

    print(f"\nCanvas : {canvas_w} x {canvas_h}   Subgoals: {len(steps)}\n")
    for step in steps:
        print(f"  {step.get('step','?')}. [{step.get('layer','')}] {step['name']} — {step.get('description','')}")

    _divider("FULL SUBGOAL OUTPUT (review before code generation)")
    print(json.dumps(steps, indent=2))

    input("\nReview the subgoals above. Press Enter to start code generation...")

    # ── Step 0: create document ───────────────────────────────────────────
    _divider("STEP 0 — Create Krita canvas")
    print("\nPaste this in Krita Scripter FIRST:\n")
    print(_DOC_INIT_CODE.format(w=canvas_w, h=canvas_h))
    input("Press Enter once the canvas is open in Krita...")

    for i, step in enumerate(steps, start=1):
        _divider(f"STEP {step.get('step', i)}/{len(steps)} — {step['name']}  [{step.get('layer','')}]")
        print(f"Action : {step.get('action', '')}\n")

        print("  Pass 1: computing coordinates...")
        coords = generate_coords(step, llm)

        print("  Pass 2: generating Krita code...")
        api_context   = retrieve_api(retriever, step)
        brush_context = retrieve_brushes(brush_retriever, step, brush_list)
        messages = _build_code_messages(step, _SYSTEM, api_context=api_context,
                                        coords=coords, brush_list=brush_context)
        response = llm.invoke(messages)
        code = response.content if hasattr(response, "content") else str(response)
        code = code.replace("```python", "").replace("```", "").strip()
        code = _validate_brush_presets(code, brush_list)   # validated against the FULL list, not the narrowed one
        code = _fix_krita_class_calls(code)

        print("\n" + "─" * 64)
        print("PASTE INTO KRITA SCRIPTER:")
        print("─" * 64 + "\n")
        print(code)
        print("\n" + "─" * 64)

        if i < len(steps):
            input("\nPress Enter to generate the next step...")

    _divider(f"DONE — all {len(steps)} subgoals complete")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print('Usage: python revised_code_action.py "<query>" [--canvas=WxH] [--cloud]')
        sys.exit(1)

    cloud = "--cloud" in args
    canvas_w, canvas_h = 900, 600
    for a in args:
        if a.startswith("--canvas="):
            w, h = a.split("=", 1)[1].lower().split("x")
            canvas_w, canvas_h = int(w), int(h)

    if args[0].startswith("--"):
        print("Error: query must be the first argument.")
        sys.exit(1)
    query = args[0]

    run(query, canvas_w=canvas_w, canvas_h=canvas_h, cloud=cloud)
