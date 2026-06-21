"""
plan_generator_v2.py  —  Iterative Krita plan generator
(Qwen-VL + Ollama + LangChain)

PIPELINE OVERVIEW
─────────────────────────────────────────────────────────────────────────────
Stage 0 │ Configuration   — model name, Ollama URL, canvas image size cap
Stage 1 │ PIL renderer    — draws the accumulated manifest as a PNG so Qwen
         │                  can literally see what's already on the canvas
Stage 2 │ LLM call        — Qwen-VL reads the image + JSON manifest and
         │                  outputs draw operations for the current step only
Stage 3 │ Stroke refiner  — takes Qwen's rough 5-15 point paths and produces
         │                  60-point mathematically smooth strokes with a
         │                  natural pressure envelope (no hardcoded shapes)
Stage 4 │ Action expander — converts each draw op dict into the exact sequence
         │                  of Krita JSON actions (set_brush, set_color, paint_*)
         │                  Brush preset is chosen per-draw by Qwen based on the
         │                  element description + krita_presets.json
Stage 5 │ Orchestrator    — loops over every process step, grows the manifest
         │                  after each step, saves the final plan.json

One LangChain call per process step.  Each call sends Qwen-VL:
  ① PIL-rendered PNG    — visual snapshot of the current canvas state
  ② Zone context        — mandatory pixel bounds for this step derived from
                          process_generator's zone map + full scene layout
  ③ JSON manifest       — compact coordinates of everything painted so far

Usage:
    python plan_generator_v2.py <process.json> [<output.json>] [--canvas WxH]

Environment variables (optional):
    QWEN_MODEL  — Ollama model tag  (default: qwen2.5vl:7b)
    OLLAMA_URL  — Ollama base URL   (default: http://localhost:11434)
"""

from __future__ import annotations
import base64
import io
import json
import math
import os
import random
import re
import sys

from PIL import Image, ImageDraw

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


def _in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ════════════════════════════════════════════════════════════════════════════
# Stage 0 — Configuration
# ════════════════════════════════════════════════════════════════════════════

# Ollama model tag — must support vision input (multimodal).
# Vision-capable Qwen tags: qwen2.5vl:7b | qwen2.5vl:72b | qwen2-vl:7b
# Set QWEN_MODEL in your environment to match whatever you pulled with Ollama.
MODEL      = os.environ.get("QWEN_MODEL", "qwen2.5vl:7b")

# Ollama server address — default for a local installation.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Maximum image width sent to Qwen-VL.
# Smaller = fewer tokens used by the image, leaving more room for the
# JSON manifest and the model's output.  512 px is enough for spatial reasoning.
IMAGE_MAX_W = 512

# How many points the stroke refiner (Stage 3) expands each rough path to.
# 60 matches the quality used in stroke_templates.py's parametric templates.
STROKE_TARGET_POINTS = 60


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 — PIL canvas renderer
# Converts the accumulated draw manifest into a real image so Qwen-VL can
# *see* the current canvas state instead of just reading text coordinates.
# ════════════════════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_str: str, opacity: float = 1.0) -> tuple[int, int, int]:
    """
    Convert a CSS hex color (#rrggbb) and an opacity float (0-1) to an
    RGB tuple suitable for PIL.

    Opacity is approximated by blending toward black — this is a visual
    approximation for the canvas preview only.  Krita uses real alpha.
    """
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (int(r * opacity), int(g * opacity), int(b * opacity))


def _draw_pil(draw: ImageDraw.ImageDraw, d: dict) -> None:
    """
    Render a single draw-op dict onto a PIL ImageDraw surface.

    Supported draw types and their PIL equivalents:
      filled_rect   → draw.rectangle(fill)
      outlined_rect → draw.rectangle(outline)
      strokes       → draw.line (polyline through all path points)
      polygon       → draw.polygon
      ellipse       → draw.ellipse
    """
    dtype  = d.get("type", "")
    rgb    = _hex_to_rgb(d.get("color", "#ffffff"), d.get("opacity", 1.0))
    bsize  = max(1, int(d.get("brush_size", 4)))

    if dtype == "filled_rect":
        draw.rectangle(
            [d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]],
            fill=rgb,
        )

    elif dtype == "outlined_rect":
        draw.rectangle(
            [d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]],
            outline=rgb, width=bsize,
        )

    elif dtype == "strokes":
        # Draw the full path as a polyline.
        # At render time we draw the REFINED path if available (Stage 3 already
        # ran), otherwise the raw Qwen path — both work fine for the preview.
        path = [(int(p[0]), int(p[1])) for p in d.get("path", [])]
        if len(path) >= 2:
            draw.line(path, fill=rgb, width=bsize)

    elif dtype == "polygon":
        pts = [(int(p[0]), int(p[1])) for p in d.get("points", [])]
        if len(pts) >= 3:
            if d.get("fill", True):
                draw.polygon(pts, fill=rgb)
            else:
                draw.polygon(pts, outline=rgb, width=bsize)

    elif dtype == "ellipse":
        bb = [d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]]
        if d.get("fill", True):
            draw.ellipse(bb, fill=rgb)
        else:
            draw.ellipse(bb, outline=rgb, width=bsize)


def manifest_to_image(manifest: list[dict],
                      canvas_w: int,
                      canvas_h: int,
                      bg_color: str = "#111111") -> Image.Image:
    """
    Stage 1 entry point.

    Render every draw-op in the entire manifest (all previous steps) onto a
    PIL Image in the correct paint order (step 1 at the bottom, latest on top).
    The resulting image is what Qwen-VL will see as the current canvas.

    Returns a PIL Image downscaled to IMAGE_MAX_W wide if the canvas is larger.
    """
    img  = Image.new("RGB", (canvas_w, canvas_h), _hex_to_rgb(bg_color))
    draw = ImageDraw.Draw(img)

    # Paint each step's draws in order so later steps overlay earlier ones.
    for entry in manifest:
        for d in entry.get("draws", []):
            _draw_pil(draw, d)

    # Downscale to keep the base64 payload small.
    if canvas_w > IMAGE_MAX_W:
        scale = IMAGE_MAX_W / canvas_w
        img   = img.resize(
            (IMAGE_MAX_W, int(canvas_h * scale)),
            Image.LANCZOS,
        )
    return img


def _image_to_b64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 PNG string for embedding in the LLM prompt."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _compact_manifest(manifest: list[dict]) -> list[dict]:
    """
    Return a token-efficient summary of the manifest for the JSON context.

    Long stroke paths and polygon point-lists are truncated to a few
    representative points with a note explaining the full length.
    This keeps the JSON prompt short without losing spatial information.
    """
    out = []
    for entry in manifest:
        compact_draws = []
        for d in entry.get("draws", []):
            c = {k: v for k, v in d.items()}

            # Truncate long stroke paths — show first 3 and last 2 points.
            if d.get("type") == "strokes":
                path = d.get("path", [])
                if len(path) > 6:
                    c["path"]      = path[:3] + path[-2:]
                    c["path_note"] = f"{len(path)} points total (first 3 + last 2 shown)"

            # Truncate long polygon point-lists similarly.
            elif d.get("type") == "polygon":
                pts = d.get("points", [])
                if len(pts) > 6:
                    c["points"]      = pts[:4] + [pts[-1]]
                    c["points_note"] = f"{len(pts)} points total (first 4 + last shown)"

            compact_draws.append(c)
        out.append({
            **{k: v for k, v in entry.items() if k != "draws"},
            "draws": compact_draws,
        })
    return out


# ════════════════════════════════════════════════════════════════════════════
# Stage 1b — SVG renderer
# Produces a human-readable SVG snapshot of the canvas at any point in time.
# Saved to  <output_dir>/<basename>_steps/step_N.svg  after every step so
# you can open each file in a browser and watch the painting build up.
# ════════════════════════════════════════════════════════════════════════════

def _draw_to_svg(d: dict) -> str:
    """
    Convert one draw-op dict to an SVG element string.

    Each draw type maps to the closest SVG primitive:
      filled_rect   → <rect fill="...">
      outlined_rect → <rect fill="none" stroke="...">
      strokes       → <polyline>
      polygon       → <polygon>
      ellipse       → <ellipse>
    """
    dtype   = d.get("type", "")
    color   = d.get("color", "#ffffff")
    opacity = d.get("opacity", 1.0)
    bsize   = d.get("brush_size", 4)

    if dtype == "filled_rect":
        return (f'<rect x="{d["x"]}" y="{d["y"]}" '
                f'width="{d["w"]}" height="{d["h"]}" '
                f'fill="{color}" opacity="{opacity:.2f}"/>')

    if dtype == "outlined_rect":
        return (f'<rect x="{d["x"]}" y="{d["y"]}" '
                f'width="{d["w"]}" height="{d["h"]}" '
                f'fill="none" stroke="{color}" '
                f'stroke-width="{bsize}" opacity="{opacity:.2f}"/>')

    if dtype == "strokes":
        # Join all path points as "x,y" pairs separated by spaces.
        pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in d.get("path", []))
        return (f'<polyline points="{pts}" stroke="{color}" '
                f'stroke-width="{bsize}" fill="none" opacity="{opacity:.2f}"/>')

    if dtype == "polygon":
        pts  = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in d.get("points", []))
        fill = d.get("fill", True)
        if fill:
            return f'<polygon points="{pts}" fill="{color}" opacity="{opacity:.2f}"/>'
        return (f'<polygon points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{bsize}" opacity="{opacity:.2f}"/>')

    if dtype == "ellipse":
        # SVG ellipse uses cx/cy/rx/ry; our dict stores x/y/w/h bounding box.
        cx = d["x"] + d["w"] / 2
        cy = d["y"] + d["h"] / 2
        rx = d["w"] / 2
        ry = d["h"] / 2
        fill = d.get("fill", True)
        if fill:
            return (f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" '
                    f'rx="{rx:.1f}" ry="{ry:.1f}" '
                    f'fill="{color}" opacity="{opacity:.2f}"/>')
        return (f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" '
                f'rx="{rx:.1f}" ry="{ry:.1f}" fill="none" '
                f'stroke="{color}" stroke-width="{bsize}" opacity="{opacity:.2f}"/>')

    return ""   # unknown type — skip


def manifest_to_svg(manifest: list[dict],
                    canvas_w: int,
                    canvas_h: int,
                    bg_color: str = "#111111") -> str:
    """
    Stage 1b entry point.

    Render the full accumulated manifest as an SVG string.
    Steps are painted in order (earlier steps at the bottom).
    Each step's draws are wrapped in an SVG <g> group with an id and label
    so you can inspect individual layers when opening in a browser.

    Returns a complete SVG document string ready to write to a .svg file.
    """
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas_w} {canvas_h}" '
        f'width="{canvas_w}" height="{canvas_h}">',

        # Background rectangle.
        f'  <rect width="{canvas_w}" height="{canvas_h}" fill="{bg_color}"/>',
    ]

    for entry in manifest:
        step_num   = entry.get("step", "?")
        layer_name = entry.get("element", "layer")

        # Wrap each step in a named group — id carries the layer name.
        # inkscape:label is omitted because it requires a namespace declaration
        # that Python's XML parser rejects when rendering SVG in Colab.
        lines.append(f'  <g id="step-{step_num}-{layer_name.replace(" ", "_")}">')

        for d in entry.get("draws", []):
            svg_elem = _draw_to_svg(d)
            if svg_elem:
                lines.append(f'    {svg_elem}')

        lines.append("  </g>")

    lines.append("</svg>")
    return "\n".join(lines)


def save_step_svg(manifest: list[dict],
                  canvas_w: int,
                  canvas_h: int,
                  bg_color: str,
                  steps_dir: str,
                  step_num: int) -> str:
    """
    Save an SVG snapshot for the current step to `steps_dir/step_NN.svg`.
    Creates the directory if it doesn't exist.
    Returns the full path of the saved file.
    """
    os.makedirs(steps_dir, exist_ok=True)
    svg_path = os.path.join(steps_dir, f"step_{step_num:02d}.svg")
    svg_str  = manifest_to_svg(manifest, canvas_w, canvas_h, bg_color)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    return svg_path


# ════════════════════════════════════════════════════════════════════════════
# Zone → pixel utilities
# Converts the 3×3 zone grid from process.json into pixel bounding boxes
# that Qwen can use directly as draw coordinates.
# ════════════════════════════════════════════════════════════════════════════

def _zone_to_pixels(zones: list[str], canvas_w: int, canvas_h: int) -> dict:
    """
    Convert a list of zone labels (e.g. ["lower-left","lower-center"])
    into a single bounding-box dict {x, y, w, h} covering all of them.

    Grid:
      rows  → upper (top third)  | mid (middle third) | lower (bottom third)
      cols  → left (left third)  | center (mid third) | right (right third)
    """
    if not zones:
        return {}

    row_bounds = {
        "upper": (0,              canvas_h // 3),
        "mid":   (canvas_h // 3,  2 * canvas_h // 3),
        "lower": (2 * canvas_h // 3, canvas_h),
    }
    col_bounds = {
        "left":   (0,              canvas_w // 3),
        "center": (canvas_w // 3,  2 * canvas_w // 3),
        "right":  (2 * canvas_w // 3, canvas_w),
    }

    x_min, y_min = canvas_w, canvas_h
    x_max, y_max = 0, 0
    matched = False

    for zone in zones:
        parts = zone.split("-", 1)
        if len(parts) != 2:
            continue
        row, col = parts[0], parts[1]
        if row in row_bounds and col in col_bounds:
            y1, y2 = row_bounds[row]
            x1, x2 = col_bounds[col]
            x_min = min(x_min, x1)
            y_min = min(y_min, y1)
            x_max = max(x_max, x2)
            y_max = max(y_max, y2)
            matched = True

    if matched:
        return {"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min}
    return {}


def _build_zone_context(step_zones: list[str],
                        element_zones: list[dict],
                        canvas_w: int,
                        canvas_h: int) -> str:
    """
    Build the zone context block that gets appended to every step's LLM prompt.

    Includes:
      ① The current step's pixel bounding box — Qwen must draw inside this region.
      ② A full scene zone map — so Qwen knows where every OTHER element lives
         and can avoid overlapping them incorrectly.
    """
    lines = ["\n── ZONE ASSIGNMENTS ──────────────────────────────────────────────────────"]

    # ① Current step's pixel region — where this element belongs on the canvas
    if step_zones:
        bounds = _zone_to_pixels(step_zones, canvas_w, canvas_h)
        if bounds:
            x, y, w, h = bounds["x"], bounds["y"], bounds["w"], bounds["h"]
            lines.append(
                f"THIS STEP occupies zones: {', '.join(step_zones)}"
            )
            lines.append(
                f"  Pixel region: x={x}  y={y}  w={w}  h={h}  "
                f"(right edge={x+w}, bottom edge={y+h})"
            )
            lines.append(
                f"  Place all draws for this element inside these pixel bounds."
                f" Use the step description to decide what shapes to draw and how large —"
                f" the zone is a boundary, not a fill instruction."
            )

    # ② Full scene map so Qwen knows where every other element sits
    if element_zones:
        lines.append("\nFull scene layout (reference — avoid wrong overlaps):")
        for ez in element_zones:
            b = _zone_to_pixels(ez.get("zones", []), canvas_w, canvas_h)
            if b:
                label = (f"x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}")
            else:
                label = "?"
            lines.append(
                f"  {ez['name']:<28} [{', '.join(ez.get('zones', []))}]  →  {label}"
            )

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 — LLM call (Qwen-VL via Ollama + LangChain)
# One call per process step.  Qwen receives THREE context formats:
#   ① PIL-rendered PNG image   — visual snapshot of the current canvas state
#   ② Zone assignment text     — mandatory pixel bounds for this step + full
#                                scene layout showing where every element goes
#   ③ JSON geometry manifest   — compact coordinates of everything drawn so far
# ════════════════════════════════════════════════════════════════════════════

# --- Output schema Qwen must follow ---
# NOTE: No // comments here — they are not valid JSON and cause parse errors
#       when the model reproduces them verbatim in its output.
_DRAW_SCHEMA = """
Output ONE valid JSON object only. No markdown fences. No comments. No trailing commas.

{
  "layer_name": "string",
  "layer_hint": "background or midground or foreground",
  "draws": [
    {"type": "filled_rect",   "x": 0, "y": 0, "w": 100, "h": 100, "color": "#rrggbb", "opacity": 1.0, "brush_size": 200, "preset": "brush name from list"},
    {"type": "strokes",       "path": [[x,y],[x,y]], "pressures": [0.8, 0.6], "color": "#rrggbb", "opacity": 0.8, "brush_size": 30, "preset": "brush name from list"},
    {"type": "polygon",       "points": [[x,y],[x,y],[x,y]], "fill": true, "color": "#rrggbb", "opacity": 1.0, "brush_size": 4, "preset": "brush name from list"},
    {"type": "ellipse",       "x": 0, "y": 0, "w": 50, "h": 50, "fill": true, "color": "#rrggbb", "opacity": 1.0, "brush_size": 4, "preset": "brush name from list"},
    {"type": "outlined_rect", "x": 0, "y": 0, "w": 100, "h": 100, "color": "#rrggbb", "opacity": 1.0, "brush_size": 3, "preset": "brush name from list"}
  ]
}

The "preset" field is REQUIRED on every draw. Choose the brush that best matches
what you are painting — not the draw type, but the visual character of the element.
Examples: grass→stamp or bristle brush, wood trunk→rough ink brush,
sky wash→basic or watercolor brush, silhouette→precision ink brush.

Draw type guide (choose whichever fits the element best):
- filled_rect:   large flat solid areas — walls, floors, sky base, background fills
- strokes:       organic or textured marks — trunks, branches, grass, hair, water, texture
- polygon:       clean silhouettes and geometric shapes — figures, buildings, leaves
- ellipse:       round or oval forms — blossoms, stars, halos, bubbles
- outlined_rect: box outlines with no fill — windows, screens, frames, shelves

You are NOT restricted to the suggested type — match the draw type to the visual
character of the element, not just its category.
"""

# --- System prompt sent on every step ---
_SYSTEM = (
    "You are a digital painting assistant generating precise Krita brush strokes.\n"
    "Each turn you handle ONE step of a painting process.\n\n"
    "You receive:\n"
    "  • A canvas image   — look at it to understand spatial layout and existing colors.\n"
    "  • Zone assignments — mandatory pixel bounds telling you exactly where to draw.\n"
    "  • A JSON manifest  — exact pixel coordinates of everything painted so far.\n\n"
    "Canvas coordinate system: (0,0) = TOP-LEFT.  X increases RIGHT.  Y increases DOWN.\n\n"
    + _DRAW_SCHEMA
    + "\nPainting rules:\n"
    "- Use the IMAGE for spatial context; use the JSON for exact coordinates.\n"
    "- Choose colors that harmonise with what is already visible in the image.\n"
    "- Do not cover completed regions unless the step explicitly describes layering.\n"
    "- For strokes: give 5-15 key points that describe the shape — "
    "  Python will automatically smooth them into 60 high-quality points.\n"
    "- All coordinates must stay within the canvas bounds.\n"
    "- opacity: 0.0-1.0.  brush_size: 1-500 pixels.\n"
)


def decide_background_color(process: dict, llm: ChatOllama) -> str:
    """
    Stage 2 — pre-loop call.

    Ask Qwen to choose a canvas background color based on the painting summary
    and first step.  Returns a lowercase hex string like '#1a1a2e'.

    """
    summary    = process.get("summary", "")
    steps      = process.get("steps", [])
    first_step = ""
    if steps:
        s = steps[0]
        first_step = " | ".join(
            str(s[k]) for k in ("action", "observation", "rule", "hypothesis")
            if s.get(k)
        )

    prompt = (
        f"Painting summary: {summary}\n"
        f"First step: {first_step}\n\n"
        "What single background color should the canvas start with?\n"
        "For indoor room scenes choose the dominant wall color (not white, not black).\n"
        "For outdoor scenes choose the sky or ground color.\n"
        "Reply with ONLY a hex color code like #c8b89a — nothing else."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw      = response.content.strip()

    # Extract the first valid #rrggbb pattern from the response.
    match = re.search(r"#[0-9a-fA-F]{6}", raw)
    if match:
        return match.group(0).lower()

    # Fallback if the model returned something unparseable.
    print(f"  [bg] Could not parse '{raw}' — using #111111")
    return "#111111"


def _parse_llm_json(raw: str) -> dict:
    """
    Robustly parse JSON from an LLM response.

    LLMs commonly produce JSON that is slightly invalid:
      - // line comments  (our old schema had these — Qwen reproduced them)
      - /* block comments */
      - Trailing commas before } or ]
      - Markdown code fences (```json ... ```)

    This function strips all of the above before calling json.loads().
    If it still fails it raises JSONDecodeError with the cleaned text
    included so the caller can log it and retry.
    """
    # 1. Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text.rstrip())

    # 2. Try parsing as-is first (fast path for well-formed output)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Strip // line comments (invalid JSON, but LLMs love them)
    text = re.sub(r"//[^\n]*", "", text)

    # 4. Strip /* block comments */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # 5. Remove trailing commas before } or ] (also invalid JSON)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 6. Try again after cleaning
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 7. Last resort — extract the outermost {...} block and try that
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Nothing worked — raise with the cleaned text so the retry loop can log it
    raise json.JSONDecodeError(
        f"Could not parse LLM output after cleaning.\nCleaned text:\n{text[:500]}",
        text, 0
    )


def step_to_draws(step: dict,
                  canvas_w: int,
                  canvas_h: int,
                  manifest: list[dict],
                  bg_color: str,
                  llm: ChatOllama,
                  element_zones: list[dict] | None = None,
                  brush_list: str = "") -> dict:
    """
    Stage 2 — main per-step LLM call.

    Sends three context formats to Qwen-VL so it can make spatially accurate
    decisions about what to paint:

      ① PNG image     (Stage 1 output)  — visual snapshot of current canvas state
      ② Zone context  (process.json)    — mandatory pixel bounds for this step
                                          + full scene layout of all elements
      ③ JSON manifest (compact)         — exact pixel coordinates of prior draws

    Returns a dict:
      { "layer_name": str, "layer_hint": str, "draws": [...] }
    """

    # --- Build the image context (Stage 1) ---
    img = manifest_to_image(manifest, canvas_w, canvas_h, bg_color)
    b64 = _image_to_b64(img)

    # --- Build the JSON context ---
    json_ctx = (
        json.dumps(_compact_manifest(manifest), indent=2)
        if manifest
        else "[]  ← canvas is empty; this is the first element"
    )

    # --- Extract step description (process.json format: action + reason) ---
    step_num  = step.get("step", "?")
    step_text = step.get("action", "")
    reason    = step.get("reason", "")

    # --- Compose the user message ---
    # /no_think disables Qwen3's internal chain-of-thought reasoning phase.
    # Without this, Qwen3-VL burns thousands of context tokens on hidden
    # reasoning before generating any output, leaving too little room for
    # long detailed JSON responses and causing mid-output truncation.
    # Safe to include even for non-Qwen3 models — they ignore the prefix.

    # Build zone context from process.json zone assignments.
    # This tells Qwen exactly which pixel region this step belongs to
    # and where every other element sits — prevents wrong placement.
    step_zones   = step.get("zones", [])
    zone_context = _build_zone_context(
        step_zones, element_zones or [], canvas_w, canvas_h
    )

    text = (
        "/no_think\n"
        f"STEP {step_num}: {step_text}\n"
        + (f"Reason: {reason}\n" if reason else "")
        + f"\nCanvas: {canvas_w}×{canvas_h} pixels\n"
        + zone_context + "\n"
        + (f"\n── Available Krita brushes (use exact name in each draw's preset field) ──\n{brush_list}\n"
           if brush_list else "")
        + "\n── JSON geometry manifest (exact coordinates of what's already painted) ──\n"
        f"{json_ctx}\n"
        "\n── The attached image shows the current canvas state visually ──\n"
        "\nGenerate the draw operations for THIS step only."
    )

    # --- Send to Qwen-VL via LangChain ---
    # Image is placed first so the vision encoder processes it before the text.
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text",      "text": text},
        ]),
    ]

    # Stream the response token by token — keeps the HTTP connection alive
    # for long detailed outputs without hitting connection timeouts.
    # On retry we always use the ORIGINAL messages — never append the failed
    # response, because that grows the context and causes overflow errors.
    last_err = None
    for attempt in range(1, 4):
        raw = ""
        for chunk in llm.stream(messages):
            raw += chunk.content

        raw = raw.strip()
        try:
            return _parse_llm_json(raw)
        except (json.JSONDecodeError, ValueError) as err:
            last_err = err
            # Log to JSON file instead of printing to console
            if attempt < 3:
                # Small nudge in the system context — does NOT append to messages
                # to avoid growing the context window and causing overflow
                pass

    raise ValueError(
        f"Step {step_num}: Qwen returned invalid JSON after 3 attempts. "
        f"Last error: {last_err}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Stage 3 — Generic stroke refiner
#
# Problem: Qwen outputs only 5-15 rough coordinate points per stroke.
# Those are enough for Qwen to communicate *where and what shape*, but they
# produce choppy, mechanical strokes in Krita.
#
# Solution: Python takes Qwen's key points and:
#   A. Smooths the path  → Catmull-Rom spline → 60 evenly-spaced points
#   B. Builds pressure   → sin(π·t) envelope  → swells in middle, tapers
#                          at both ends, like a real brush lifted down and up
#   C. Adds jitter       → ±5% random noise per point → hand-painted feel
#
# This is GENERIC — it works for any shape (sky stroke, desk edge, hair,
# circuit trace) without knowing what kind of stroke it is.
# ════════════════════════════════════════════════════════════════════════════

def _catmull_rom_segment(p0: list, p1: list, p2: list, p3: list,
                         steps: int) -> list[list[float]]:
    """
    Compute `steps` points along the Catmull-Rom spline segment between
    p1 and p2, using p0 and p3 as the surrounding control points.

    Catmull-Rom is chosen because:
      - It passes THROUGH p1 and p2 (the path points Qwen gave us)
      - It produces smooth, continuous curves at every joint
      - It needs no external library — pure Python math
    """
    points = []
    for i in range(steps):
        t  = i / steps
        t2 = t * t
        t3 = t * t * t
        # Catmull-Rom formula for x and y independently.
        x = 0.5 * (
            2 * p1[0]
            + (-p0[0] + p2[0]) * t
            + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
            + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
        )
        y = 0.5 * (
            2 * p1[1]
            + (-p0[1] + p2[1]) * t
            + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
            + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
        )
        points.append([x, y])
    return points


def _smooth_path(path: list, target: int = STROKE_TARGET_POINTS) -> list[list[float]]:
    """
    Stage 3A — Path smoothing.

    Expands a rough list of [x,y] key points into a smooth curve with
    `target` evenly spaced points using Catmull-Rom spline interpolation.

    Edge handling: phantom endpoints are added by reflecting the first and
    last real points so the spline doesn't collapse at the ends.
    """
    n = len(path)

    # Only 1 point — nothing to smooth.
    if n < 2:
        return [list(p) for p in path]

    # Only 2 points — linear interpolation is the smoothest possible.
    if n == 2:
        p0, p1 = path[0], path[1]
        return [
            [p0[0] + (p1[0] - p0[0]) * i / (target - 1),
             p0[1] + (p1[1] - p0[1]) * i / (target - 1)]
            for i in range(target)
        ]

    # 3+ points — full Catmull-Rom with phantom endpoints.
    # Phantom start: reflect path[1] through path[0].
    # Phantom end:   reflect path[-2] through path[-1].
    ext = [
        [2*path[0][0] - path[1][0],   2*path[0][1] - path[1][1]],
        *[list(p) for p in path],
        [2*path[-1][0] - path[-2][0], 2*path[-1][1] - path[-2][1]],
    ]

    n_segments       = n - 1
    steps_per_seg    = max(2, target // n_segments)
    smooth: list     = []

    for i in range(n_segments):
        # Each real segment uses the two surrounding points as controls.
        p0, p1, p2, p3 = ext[i], ext[i+1], ext[i+2], ext[i+3]
        smooth.extend(_catmull_rom_segment(p0, p1, p2, p3, steps_per_seg))

    # Append the final endpoint which the loop above skips.
    smooth.append(list(path[-1]))
    return smooth


def _pressure_envelope(n: int,
                       existing: list[float] | None = None,
                       lo: float = 0.4,
                       hi: float = 1.0,
                       jitter: float = 0.05) -> list[float]:
    """
    Stage 3B+C — Pressure envelope + jitter.

    Generates `n` pressure values for a smooth stroke.

    Two paths:
      • If Qwen supplied pressures: interpolate them to length n (preserves
        the artistic intent Qwen expressed), then add jitter.
      • If Qwen gave no pressures: generate a sin(π·t) envelope from scratch
        — pressure starts low (brush touching canvas), peaks at the midpoint
        (brush pushed hardest), and fades back to low (brush lifting off).

    After either path, ±jitter random noise is added to every point so the
    stroke looks hand-painted rather than mechanically perfect.
    """
    rng = random.Random()   # fresh instance — no global state side effects

    if existing and len(existing) >= 2:
        # --- Interpolate Qwen's pressure values to length n ---
        src_n  = len(existing)
        base   = []
        for i in range(n):
            t      = i / (n - 1) if n > 1 else 0.0
            src_t  = t * (src_n - 1)
            src_i  = int(src_t)
            src_f  = src_t - src_i
            if src_i >= src_n - 1:
                val = existing[-1]
            else:
                # Linear interpolation between adjacent source pressure values.
                val = existing[src_i] * (1 - src_f) + existing[src_i + 1] * src_f
            base.append(val)
    else:
        # --- Generate sin(π·t) envelope ---
        # t goes 0 → 1 along the stroke.
        # sin(π·t) goes 0 → 1 → 0 — a natural brush-down/push/lift-off shape.
        # Scale from [0,1] into [lo, hi] so the stroke never fully disappears.
        base = []
        for i in range(n):
            t   = i / (n - 1) if n > 1 else 0.5
            env = math.sin(math.pi * t)          # pure 0→1→0 shape
            val = lo + (hi - lo) * env            # scale into [lo, hi]
            base.append(val)

    # --- Add jitter (Stage 3C) ---
    # ±jitter random noise per point so each stroke looks unique and hand-made.
    result = []
    for p in base:
        noise = rng.uniform(-jitter, jitter)
        result.append(max(0.05, min(1.0, p + noise)))   # clamp to valid range

    return result


def refine_stroke(draw: dict) -> dict:
    """
    Stage 3 entry point.

    Takes a raw 'strokes' draw-op dict from Qwen (5-15 rough points) and
    returns an improved version with:
      - 60 smooth points via Catmull-Rom spline   (Stage 3A)
      - Natural pressure curve per point           (Stage 3B)
      - ±5% random jitter per pressure value       (Stage 3C)

    Works for ANY stroke shape — sky swirl, desk edge, hair, cloud —
    because no shape-specific knowledge is used; only the key points
    Qwen provided are smoothed mathematically.

    Non-stroke draw types (filled_rect, polygon, ellipse, outlined_rect)
    are returned unchanged — they don't have a path to smooth.
    """
    if draw.get("type") != "strokes":
        return draw   # only strokes are refined

    path = draw.get("path", [])
    if len(path) < 2:
        return draw   # nothing to smooth with fewer than 2 points

    # Stage 3A: smooth the path.
    smooth_pts = _smooth_path(path, target=STROKE_TARGET_POINTS)

    # Stage 3B+C: build pressure — uses Qwen's values if present, otherwise
    # generates the sin(π·t) envelope, then adds jitter either way.
    pressures = _pressure_envelope(
        n        = len(smooth_pts),
        existing = draw.get("pressures"),   # None if Qwen didn't supply them
    )

    # Return a new draw dict with the refined path and pressure replacing the raw ones.
    return {**draw, "path": smooth_pts, "pressures": pressures}


# ════════════════════════════════════════════════════════════════════════════
# Stage 4a — Brush preset selector
# One LLM call before the main loop.  Reads krita_presets.json (exported
# from Krita's Scripter) and asks the model to choose the best brush for
# each draw type given the painting's style and subject matter.
# ════════════════════════════════════════════════════════════════════════════

# Prefix letters that correspond to actual painting brushes.
# Skips erasers (a), blenders (k), adjustment (l), pixel art (u),
# distort/experimental (v), normal-map (w), filter (x), screentones (y),
# stamps (z) — these are not useful for painting shapes.
_PAINTING_PREFIXES = {"b", "c", "d", "e", "f", "g", "h", "i", "j", "m"}


def select_brush_presets(presets: list[dict],
                         summary: str,
                         llm: ChatOllama) -> dict:
    """
    Stage 4a — one LLM call before the step loop.

    Given the installed Krita brush list and the painting summary, asks the
    model to pick the best preset for each of the five draw types.

    Returns a mapping dict:
      {"filled_rect": "b) Basic-5 Size", "strokes": "f) Bristles-2 Flat Rough", ...}

    Falls back to b) Basic-5 Size for any preset the model invents that does
    not exist in the installed list.
    """
    # Only show painting-relevant brushes to keep the prompt short.
    painting_presets = [
        p for p in presets
        if p.get("name", "")[0].lower() in _PAINTING_PREFIXES
    ]

    preset_lines = "\n".join(
        f'  {p["name"]:<38} — {p["description"]}'
        for p in painting_presets
    )

    prompt = (
        f'Painting: "{summary}"\n\n'
        "Available Krita brush presets:\n"
        f"{preset_lines}\n\n"
        "Choose the single best preset for each draw type that suits this "
        "painting's style and subject matter.\n"
        "Output ONLY valid JSON — no markdown, no comments:\n"
        "{\n"
        '  "filled_rect":   "exact preset name",\n'
        '  "strokes":       "exact preset name",\n'
        '  "polygon":       "exact preset name",\n'
        '  "ellipse":       "exact preset name",\n'
        '  "outlined_rect": "exact preset name"\n'
        "}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw      = response.content.strip()

    try:
        preset_map = _parse_llm_json(raw)
    except (json.JSONDecodeError, ValueError):
        preset_map = {}

    # Validate every chosen name against the actual installed list.
    valid_names = {p["name"] for p in presets}
    fallback    = "b) Basic-5 Size"
    for key in ("filled_rect", "strokes", "polygon", "ellipse", "outlined_rect"):
        chosen = preset_map.get(key, "")
        if chosen not in valid_names:
            print(f"  [presets] '{chosen}' not installed — using {fallback}")
            preset_map[key] = fallback

    return preset_map


# ════════════════════════════════════════════════════════════════════════════
# Stage 4 — Krita action expander
# Converts each draw-op dict into the specific sequence of Krita JSON actions
# that runner.py will execute inside Krita.  No LLM involved here — pure
# deterministic Python.
# ════════════════════════════════════════════════════════════════════════════

def _expand_draw(actions: list[dict], node_ref: str, draw: dict,
                 valid_presets: set | None = None) -> None:
    """
    Stage 4 — expand one draw-op dict into Krita actions.

    Every draw type produces 4 setup actions then 1 paint action:
      1. set_brush_preset   — selects the brush tip
      2. set_brush_size     — sets the size in pixels
      3. set_painting_opacity — sets layer opacity
      4. set_foreground_color — sets the paint color
      5. paint_*            — the actual painting action

    For 'strokes', the refiner (Stage 3) is called first to smooth the path
    and generate a natural pressure curve before emitting paint_strokes.
    """
    # --- Run Stage 3 refiner for stroke paths ---
    # This must happen before we read the path/pressures below.
    draw = refine_stroke(draw)

    dtype      = draw.get("type", "")
    color      = draw.get("color", "#ffffff")
    opacity    = float(draw.get("opacity", 1.0))
    brush_size = int(draw.get("brush_size", 50))

    # --- 4 setup actions emitted before every paint call ---
    # Use the per-draw preset Qwen chose; validate it exists, else fall back.
    per_draw_preset = draw.get("preset", "")
    if valid_presets and per_draw_preset not in valid_presets:
        per_draw_preset = "b) Basic-5 Size"
    elif not per_draw_preset:
        per_draw_preset = "b) Basic-5 Size"
    actions.append({"type": "set_brush_preset",
                     "view_ref": "view:active",
                     "preset_name": per_draw_preset})
    actions.append({"type": "set_brush_size",
                     "view_ref": "view:active",
                     "size": float(brush_size)})
    actions.append({"type": "set_painting_opacity",
                     "view_ref": "view:active",
                     "opacity": opacity})
    actions.append({"type": "set_foreground_color",
                     "view_ref": "view:active",
                     "color": {"hex": color}})

    # --- 1 paint action depending on draw type ---

    if dtype == "filled_rect":
        # Solid filled rectangle — used for large background washes.
        actions.append({
            "type": "paint_rectangle", "node_ref": node_ref,
            "x": draw["x"], "y": draw["y"], "w": draw["w"], "h": draw["h"],
            "stroke_style": "None", "fill_style": "ForegroundColor",
        })

    elif dtype == "outlined_rect":
        # Outlined rectangle — stroke only, no fill.
        actions.append({
            "type": "paint_rectangle", "node_ref": node_ref,
            "x": draw["x"], "y": draw["y"], "w": draw["w"], "h": draw["h"],
            "stroke_style": "ForegroundColor", "fill_style": "None",
        })

    elif dtype == "strokes":
        # Brush stroke — path was smoothed by Stage 3 refiner above.
        path = draw.get("path", [])
        if len(path) < 2:
            return  # safety guard: can't paint a stroke with fewer than 2 points

        pressures = list(draw.get("pressures") or [1.0] * len(path))
        # Ensure pressures list matches path length (pad with 1.0 if short).
        while len(pressures) < len(path):
            pressures.append(1.0)

        actions.append({
            "type": "paint_strokes", "node_ref": node_ref,
            "path": [list(p) for p in path],
            "pressures": pressures[: len(path)],
            "stroke_style": "ForegroundColor",
        })

    elif dtype == "polygon":
        # Closed polygon — filled or outline only.
        points = draw.get("points", [])
        if len(points) < 3:
            return  # a polygon needs at least 3 points

        fill = draw.get("fill", True)
        actions.append({
            "type": "paint_polygon", "node_ref": node_ref,
            "points": [list(p) for p in points],
            "stroke_style": "None"            if fill else "ForegroundColor",
            "fill_style":   "ForegroundColor" if fill else "None",
        })

    elif dtype == "ellipse":
        # Ellipse — filled or outline only.
        fill = draw.get("fill", True)
        actions.append({
            "type": "paint_ellipse", "node_ref": node_ref,
            "x": draw["x"], "y": draw["y"], "w": draw["w"], "h": draw["h"],
            "stroke_style": "None"            if fill else "ForegroundColor",
            "fill_style":   "ForegroundColor" if fill else "None",
        })


# ════════════════════════════════════════════════════════════════════════════
# Stage 5 — Main pipeline / orchestrator
# Ties all stages together: reads process.json, loops over steps, grows the
# manifest, and writes the final plan.json that runner.py executes in Krita.
# ════════════════════════════════════════════════════════════════════════════

def generate(process_path: str,
             output_path: str = "",
             canvas_w: int = 1200,
             canvas_h: int = 950,
             presets_path: str = "") -> str:
    """
    Stage 5 entry point.

    Full pipeline: process.json → plan.json.
    Returns the path to the written plan file.

    The manifest starts empty and grows by one entry after each step.
    Every subsequent LLM call sees the full history as both an image and JSON,
    so each step is painted with awareness of everything that came before.
    """

    # --- Resolve output path ---
    if not output_path:
        base = os.path.splitext(os.path.basename(process_path))[0]
        if base.endswith("_process"):
            base = base[: -len("_process")]
        out_dir     = os.path.dirname(os.path.abspath(process_path))
        output_path = os.path.join(out_dir, f"{base}_plan_v2.json")

    basename = os.path.splitext(os.path.basename(output_path))[0]
    out_dir  = os.path.dirname(os.path.abspath(output_path))

    # --- Load the process JSON ---
    with open(process_path, "r", encoding="utf-8") as f:
        process = json.load(f)

    title         = process.get("summary", basename.replace("_", " ").title())[:80]
    steps         = process.get("steps", [])
    # Zone map produced by process_generator2 Stage 5 — tells plan_generator
    # which canvas region each element belongs to so Qwen draws in the right place.
    element_zones = process.get("element_zones", [])

    # --- Initialise the LangChain LLM (reused for all steps) ---
    # temperature=0  — deterministic geometry output
    # num_ctx        — context window; default 4096 is too small once we add
    #                  the base64 image + manifest + system prompt.  16384 is safe.
    # num_predict    — max output tokens; default is often 128 which truncates
    #                  the JSON mid-object.  2048 is plenty for one step's draws.
    llm = ChatOllama(
        model       = MODEL,
        base_url    = OLLAMA_URL,
        temperature = 0,
        num_ctx     = 32768,  # 32k — image + manifest + prompt can exceed 16k
        num_predict = -1,     # -1 = no output token limit; model stops when done
    )

    print(f"Model={MODEL}  steps={len(steps)}  canvas={canvas_w}×{canvas_h}")

    # --- Load brush presets for per-draw selection by Qwen ---
    # krita_presets.json is loaded once; the name+description list is injected
    # into every step prompt so Qwen picks the right brush per draw based on
    # what it is painting (grass→stamp, trunk→rough ink, sky→watercolor, etc.)
    if not presets_path:
        try:
            script_dir = os.path.dirname(__file__)
        except NameError:
            script_dir = os.getcwd()
        presets_path = os.path.join(script_dir, "krita_presets.json")

    brush_list    = ""   # compact text included in each step prompt
    valid_presets: set = set()

    if os.path.isfile(presets_path):
        with open(presets_path, "r", encoding="utf-8") as f:
            presets = json.load(f)
        # Only painting-relevant categories
        painting_presets = [p for p in presets
                            if p.get("name", "")[0].lower() in _PAINTING_PREFIXES]
        brush_list    = "\n".join(
            f'  {p["name"]:<38} — {p["description"]}'
            for p in painting_presets
        )
        valid_presets = {p["name"] for p in presets}
        print(f"Loaded {len(painting_presets)} brushes from {presets_path}")
    else:
        print(f"  [presets] {presets_path} not found — Qwen will use default brush")

    bg_color = decide_background_color(process, llm)
    print(f"Background: {bg_color}")

    # manifest  — grows after each step; fed as context to the next LLM call
    # actions   — accumulates all Krita actions across all steps
    manifest: list[dict] = []
    actions:  list[dict] = []

    # Folder where per-step SVG snapshots are saved.
    # e.g.  Art_Orch/nocturnal_steps/step_01.svg, step_02.svg, ...
    steps_dir = os.path.join(out_dir, f"{basename}_steps")

    # --- Krita document setup (prepended before any painting) ---
    actions.append({
        "type": "set_batchmode", "value": True,
        # Suppress all dialogs while the plan runs inside Krita.
    })
    actions.append({
        "type": "create_document", "ref": "doc:main",
        "width": canvas_w, "height": canvas_h, "name": title,
        "color_model": "RGBA", "color_depth": "U8",
        "profile": "sRGB-elle-V2-srgbtrc.icc", "resolution": 300.0,
    })
    actions.append({
        "type": "set_background_color", "doc_ref": "doc:main",
        "color": {"hex": bg_color},
        # Background color was chosen by Qwen based on the painting mood.
    })

    # ─── Main step loop ───────────────────────────────────────────────────
    for i, step in enumerate(steps):
        step_num = step.get("step", i + 1)
        print(f"[{step_num}/{len(steps)}] generating…", end=" ", flush=True)

        # Stage 2 — ask Qwen what to paint for this step.
        # Pass element_zones so the prompt includes pixel bounds for this step.
        result = step_to_draws(step, canvas_w, canvas_h, manifest, bg_color, llm,
                                element_zones=element_zones,
                                brush_list=brush_list)

        layer_name = result.get("layer_name", f"Step {step_num}")
        layer_hint = result.get("layer_hint", "midground")
        draws      = result.get("draws", [])
        layer_ref  = f"node:step{step_num}"

        print(f"'{layer_name}' — {len(draws)} draws")

        # --- Create the Krita layer for this step ---
        actions.append({
            "type": "method_call", "target_ref": "noop", "method": "noop",
            "comment": f"=== Step {step_num}: {layer_name} ({layer_hint}) ===",
        })
        actions.append({
            "type": "create_node", "ref": layer_ref, "doc_ref": "doc:main",
            "name": layer_name, "node_type": "paintlayer",
        })
        actions.append({"type": "set_node_blending_mode",
                          "node_ref": layer_ref, "mode": "normal"})
        actions.append({"type": "set_node_opacity",
                          "node_ref": layer_ref, "value": 255})
        actions.append({"type": "set_active_node",
                          "doc_ref": "doc:main", "node_ref": layer_ref})

        # Stage 3+4 — refine each draw op and expand to Krita actions.
        for d in draws:
            _expand_draw(actions, layer_ref, d, valid_presets)

        # --- Update the manifest ---
        # This step's draws are stored in the manifest so Stage 1 can render
        # them into the canvas image for the next LLM call.
        manifest.append({
            "step":       step_num,
            "element":    layer_name,
            "layer_hint": layer_hint,
            "draws":      draws,   # raw Qwen draws (not refined) — for image rendering
        })

        # --- Stage 1b: save SVG snapshot for this step ---
        # The SVG shows every element painted so far (including this step).
        # Open the file in any browser to inspect the canvas state after each step.
        svg_path = save_step_svg(manifest, canvas_w, canvas_h, bg_color,
                                 steps_dir, step_num)

        # Show SVG inline in Colab after every step.
        if _in_colab():
            from IPython.display import SVG as _SVG, display as _display, HTML as _HTML
            _display(_HTML(f"<b>Step {step_num}: {layer_name}</b>"))
            _display(_SVG(filename=svg_path))

    # --- Finalisation actions ---
    actions.append({"type": "refresh_projection", "doc_ref": "doc:main"})
    actions.append({"type": "wait_for_done",       "doc_ref": "doc:main"})

    kra = os.path.join(out_dir, f"{basename}.kra")
    png = os.path.join(out_dir, f"{basename}.png")
    actions.append({
        "type": "save_as", "doc_ref": "doc:main", "filename": kra,
        # Save native .kra so all layers are preserved for further editing.
    })
    actions.append({
        "type": "export_image", "doc_ref": "doc:main", "filename": png,
        "export_config": {"alpha": True, "compression": 1},
        # Export a flat PNG for quick sharing.
    })

    # --- Assemble and write the plan JSON ---
    plan = {
        "title":    title,
        "summary":  process.get("summary", ""),
        "metadata": {
            "canvas":   [canvas_w, canvas_h],
            "manifest": manifest,   # full draw history for reference
        },
        "actions": actions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(f"\nDone — plan saved to {output_path}  ({len(actions)} actions)")

    # Optional: validate the plan against dryrun.py if available.
    _try_validate(plan)
    return output_path


def _try_validate(plan: dict) -> None:
    """
    Optional post-generation validation using dryrun.py.

    Checks that every action has the required fields, refs are produced before
    they're consumed, and stroke paths have at least 2 points.  Prints a
    summary and any errors found.  Safe to skip if dryrun.py isn't present.
    """
    # __file__ is undefined when code is pasted directly into a Colab cell,
    # so fall back to the current working directory.
    try:
        base = os.path.dirname(__file__)
    except NameError:
        base = os.getcwd()
    dryrun_dir = os.path.join(base, "Process_to_Action-main", "Process_to_Action-main")
    if not os.path.isdir(dryrun_dir):
        return
    if dryrun_dir not in sys.path:
        sys.path.insert(0, dryrun_dir)
    try:
        from dryrun import summarize, validate  # type: ignore
        summarize(plan)
        errors = validate(plan)
        if errors:
            print(f"\n{len(errors)} validation issue(s):")
            for e in errors[:10]:
                print(f"  - {e}")
        else:
            print("\nValidation passed.")
    except ImportError:
        pass   # dryrun not available — skip silently


# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Iterative Krita plan generator — Qwen-VL + Ollama + LangChain"
    )
    parser.add_argument("process", help="Path to process.json")
    parser.add_argument("output",  nargs="?", default="",
                        help="Output plan.json path (default: next to process.json)")
    parser.add_argument("--canvas", default="1200x950",
                        help="Canvas WxH in pixels (default: 1200x950)")
    args = parser.parse_args()

    w, h = (int(x) for x in args.canvas.lower().split("x"))
    generate(args.process, args.output, canvas_w=w, canvas_h=h)
