# outline_generator.py
#
# Deterministic converter for a FULLY pre-specified plan (canvas, one global
# brush_type/brush_size, and steps whose elements already carry exact
# coordinates) straight into Krita Scripter code. No LLM calls at all — this
# is for hand-crafted or externally-generated plans, unlike
# revised_code_action.py's subgoal-decomposition + coordinate-generation
# pipeline (which uses an LLM for both of those).
#
# Expected JSON shape:
#   {
#     "canvas": {"w": int, "h": int},
#     "brush_type": "<exact preset name>",
#     "brush_size": int,
#     "steps": [
#       {
#         "step": int, "name": str, "layer": str, "description": str,
#         "bbox": {...},
#         "elements": [
#           {"role": str, "type": "stroke|path_filled|path_outline|polygon|polygon_outline|"
#                                  "rect_filled|rect_outline|ellipse_filled|ellipse_outline",
#            "placement": str, "coordinates": [...]}
#         ]
#       }
#     ]
#   }
#
# Usage:
#   python outline_generator.py plan.json [--bridge=URL] [--color=#hex]

import json
import sys
import urllib.error
import urllib.request

DEFAULT_COLOR = "#000000"   # this plan format has no per-element color, so one
                            # consistent ink color is used throughout unless overridden


def _catmull_rom_to_bezier(points: list, closed: bool = False) -> list:
    """Same deterministic curve-smoothing used by revised_code_action.py —
    converts (x,y) anchor points into cubic Bezier (p0,c1,c2,p1) segments,
    so path_filled/path_outline elements render as smooth curves rather
    than straight-edged polygons."""
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


def _send_to_krita(bridge_url: str, code: str, timeout: float = 240.0) -> tuple[bool, str]:
    """POST code to the Art Orch Bridge Krita plugin — same mechanism as
    revised_code_action.py, so an existing ngrok tunnel/bridge setup works
    unchanged with this script too."""
    try:
        req = urllib.request.Request(bridge_url, data=code.encode("utf-8"), method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return False, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return False, str(e)


_EXPORT_IMAGE_CODE = """\
import base64, os, tempfile
from krita import Krita, InfoObject
doc = Krita.instance().activeDocument()
doc.setBatchmode(True)
_tmp_path = os.path.join(tempfile.gettempdir(), "art_orch_export.png")
doc.exportImage(_tmp_path, InfoObject())
with open(_tmp_path, "rb") as _f:
    __bridge_result__ = base64.b64encode(_f.read()).decode("ascii")
os.remove(_tmp_path)
"""


def fetch_canvas_image(bridge_url: str, out_path: str = "generated.png", timeout: float = 60.0) -> str:
    """Ask the running Krita instance (via the bridge) to export its active
    document to PNG, send the bytes back as base64 in the HTTP response body,
    and write them to out_path here in Colab. Returns out_path."""
    import base64
    ok, payload = _send_to_krita(bridge_url, _EXPORT_IMAGE_CODE, timeout=timeout)
    if not ok:
        raise RuntimeError(f"Failed to export canvas from Krita: {payload}")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(payload))
    return out_path


def compare_images(generated_path: str, ground_truth_path: str) -> float:
    """Structural similarity (SSIM) between the generated canvas export and a
    ground-truth reference image, both resized to match before comparing.
    Returns a score in [-1, 1] (practically usually [0, 1]); 1.0 = identical.
    Requires: pip install scikit-image pillow numpy"""
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    gen = Image.open(generated_path).convert("L")
    gt = Image.open(ground_truth_path).convert("L").resize(gen.size)
    score = ssim(np.array(gen), np.array(gt))
    print(f"SSIM similarity: {score:.4f}  ({generated_path} vs {ground_truth_path})")
    return score


_DOC_INIT_CODE = """\
from krita import Krita, InfoObject
ki  = Krita.instance()
doc = ki.createDocument({w}, {h}, "Painting", "RGBA", "U8", "", 72.0)
ki.activeWindow().addView(doc)
doc.refreshProjection()
print("Canvas created: {w}x{h}")
"""


def _build_element_lines(element: dict) -> list:
    """Python source lines that paint one element, using its own already-exact coordinates."""
    etype = element["type"]
    coords = element["coordinates"]
    lines = [f'# {element.get("role", "")}: {element.get("placement", "")}']

    if etype == "stroke":
        lines.append(f"stroke_list = {coords!r}")
        lines.append("for x1, y1, x2, y2 in stroke_list:")
        lines.append('    node.paintLine(QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2)), 0.9, 0.3, "ForegroundColor")')

    elif etype in ("rect_filled", "rect_outline"):
        fill = "ForegroundColor" if etype == "rect_filled" else "None"
        lines.append(f"r = {coords!r}")
        lines.append(f'node.paintRectangle(QRectF(r["x"], r["y"], r["w"], r["h"]), "ForegroundColor", "{fill}")')

    elif etype in ("ellipse_filled", "ellipse_outline"):
        fill = "ForegroundColor" if etype == "ellipse_filled" else "None"
        lines.append(f"r = {coords!r}")
        lines.append(f'node.paintEllipse(QRectF(r["x"], r["y"], r["w"], r["h"]), "ForegroundColor", "{fill}")')

    elif etype in ("polygon", "polygon_outline"):
        fill = "ForegroundColor" if etype == "polygon" else "None"
        lines.append(f"points = {coords!r}")
        lines.append(f'node.paintPolygon([QPointF(x, y) for x, y in points], "ForegroundColor", "{fill}")')

    elif etype in ("path_filled", "path_outline"):
        closed = (etype == "path_filled")
        segments = _catmull_rom_to_bezier(coords, closed=closed)
        fill = "ForegroundColor" if closed else "None"
        lines.append(f"segments = {segments!r}")
        lines.append("path = QPainterPath()")
        lines.append("path.moveTo(QPointF(*segments[0][0]))")
        lines.append("for p0, c1, c2, p1 in segments:")
        lines.append("    path.cubicTo(QPointF(*c1), QPointF(*c2), QPointF(*p1))")
        if closed:
            lines.append("path.closeSubpath()")
        lines.append(f'node.paintPath(path, "ForegroundColor", "{fill}")')

    else:
        raise ValueError(f"Unsupported element type in this plan: {etype!r}")

    return lines


def build_step_code(step: dict, brush_type: str, brush_size: int, color: str = DEFAULT_COLOR) -> str:
    """Full Krita Scripter script for one step: create its layer, set the
    (single, plan-wide) brush once, paint every element using coordinates
    that are already exact — no LLM, no coordinate generation needed."""
    lines = [
        "from krita import Krita, ManagedColor",
        "from PyQt5.QtCore import QRectF, QPointF, QPoint",
        "from PyQt5.QtGui import QColor, QPainterPath",
        "",
        "def set_color(hex_str):",
        "    qc = QColor(hex_str)",
        '    mc = ManagedColor("RGBA", "U8", "")',
        "    mc.setComponents([qc.blueF(), qc.greenF(), qc.redF(), 1.0])",
        "    view.setForeGroundColor(mc)",
        "",
        "ki = Krita.instance()",
        "doc = ki.activeDocument()",
        "view = ki.activeWindow().activeView()",
        "",
        f'node = doc.createNode({step["name"]!r}, "paintlayer")',
        "doc.rootNode().addChildNode(node, None)",
        "doc.setActiveNode(node)",
        "",
        f'view.setCurrentBrushPreset(ki.resources("preset").get({brush_type!r}))',
        f"view.setBrushSize({brush_size})",
        f'set_color({color!r})',
        "",
    ]
    for element in step["elements"]:
        lines.extend(_build_element_lines(element))
        lines.append("")
    lines.append("doc.refreshProjection()")
    return "\n".join(lines)


def run(plan_path: str, bridge_url: str | None = None, color: str = DEFAULT_COLOR):
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    canvas     = plan["canvas"]
    brush_type = plan.get("brush_type", "b) Basic-5 Size")
    brush_size = plan.get("brush_size", 3)
    steps      = plan["steps"]

    print(f"Canvas: {canvas['w']}x{canvas['h']}   Brush: {brush_type} @ {brush_size}   Steps: {len(steps)}\n")

    init_code = _DOC_INIT_CODE.format(w=canvas["w"], h=canvas["h"])
    print("─" * 64)
    print(init_code)
    print("─" * 64)
    if bridge_url:
        ok, msg = _send_to_krita(bridge_url, init_code)
        print(f"  {'OK' if ok else 'ERROR'}: {msg[:300]}")
    else:
        input("Paste the canvas-creation code into Krita Scripter, then press Enter...")

    for i, step in enumerate(steps, start=1):
        print(f"\n=== STEP {step.get('step', i)}/{len(steps)} — {step['name']} ({step.get('description','')}) ===")
        code = build_step_code(step, brush_type, brush_size, color=color)
        print("─" * 64)
        print(code)
        print("─" * 64)
        if bridge_url:
            ok, msg = _send_to_krita(bridge_url, code)
            print(f"  {'OK' if ok else 'ERROR'}: {msg[:300]}")
        else:
            input("Paste into Krita Scripter, then press Enter for the next step...")

    print("\nDONE.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python outline_generator.py plan.json [--bridge=URL] [--color=#hex]")
        sys.exit(1)

    bridge_url = None
    color = DEFAULT_COLOR
    for a in args[1:]:
        if a.startswith("--bridge="):
            bridge_url = a.split("=", 1)[1]
        if a.startswith("--color="):
            color = a.split("=", 1)[1]

    run(args[0], bridge_url=bridge_url, color=color)
