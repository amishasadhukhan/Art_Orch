"""
Plan runner.

Loads an `ActionPlan` JSON and executes each action by calling the libkis
Python API. Must be run *inside* Krita — either pasted into Tools → Scripts →
Scripter, or invoked from a plugin's extension. The `krita` module only
exists in that embedded interpreter.

Usage from Krita's Scripter:

import sys
sys.path.insert(0, r"C:\\Users\\acer\\TCS\\Art_Orch")
from runner import run_plan
run_plan(r"C:\\Users\\acer\\TCS\\Art_Orch\\plan.json")

The runner maintains a symbol table mapping plan-side refs (e.g.
"doc:main", "node:sky", "view:active") to live libkis objects.
"""

from __future__ import annotations
import base64
import json
import traceback
from PyQt5.QtCore import QPoint
from typing import Any, Dict, List, Optional, Tuple

try:
    from krita import Krita, ManagedColor, Selection, Filter, InfoObject  # type: ignore
    from PyQt5.QtCore import QPointF, QRectF                                # type: ignore
    from PyQt5.QtGui import QColor, QPainterPath                             # type: ignore
    INSIDE_KRITA = True
except ImportError:
    INSIDE_KRITA = False


# ----- Helpers -------------------------------------------------------------

def _hex_to_managed_color(hex_str: str) -> "ManagedColor":
    """Convert '#rrggbb' to a ManagedColor in sRGB U8."""
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    mc = ManagedColor("RGBA", "U8", "sRGB-elle-V2-srgbtrc.icc")
    # Krita's component order is [B, G, R, A] for RGBA8 in many bindings;
    # use componentsOrdered if available, else fall back.
    try:
        mc.setComponents([b, g, r, 1.0])
    except Exception:
        mc.setComponents([r, g, b, 1.0])
    return mc


def _color_from_spec(spec: Dict[str, Any]) -> "ManagedColor":
    if "hex" in spec:
        return _hex_to_managed_color(spec["hex"])
    mc = ManagedColor(
        spec.get("model", "RGBA"),
        spec.get("depth", "U8"),
        spec.get("profile", "sRGB-elle-V2-srgbtrc.icc"),
    )
    mc.setComponents(spec["components"])
    return mc


def _qcolor_from_hex(hex_str: str) -> "QColor":
    hex_str = hex_str.lstrip("#")
    return QColor(
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16),
    )


# ----- Runner --------------------------------------------------------------

class Runner:
    def __init__(self) -> None:
        if not INSIDE_KRITA:
            raise RuntimeError("runner.py must be executed inside Krita.")
        self.k = Krita.instance()
        self.symbols: Dict[str, Any] = {"krita": self.k}

    # ----- ref resolution -----

    def _doc(self, ref: str):
        if ref in self.symbols:
            return self.symbols[ref]
        return self.k.activeDocument()

    def _view(self, ref: str = "view:active"):
        if ref in self.symbols:
            return self.symbols[ref]
        win = self.k.activeWindow()
        return win.activeView() if win else None

    def _node(self, ref: str):
        if ref in self.symbols:
            return self.symbols[ref]
        doc = self.k.activeDocument()
        return doc.activeNode() if doc else None

    def _canvas(self, ref: str = "canvas:active"):
        if ref in self.symbols:
            return self.symbols[ref]
        view = self._view()
        return view.canvas() if view else None

    # ----- dispatch -----

    def run(self, plan: Dict[str, Any]) -> None:
        actions = plan.get("actions", [])
        for i, act in enumerate(actions):
            try:
                self._dispatch(act)
            except Exception as e:
                print(f"[runner] action #{i} ({act.get('type')}) failed: {e}")
                traceback.print_exc()

    def _dispatch(self, act: Dict[str, Any]) -> None:
        t = act.get("type")
        fn = self._handlers.get(t)
        if fn is None:
            print(f"[runner] unknown action type: {t} — skipped")
            return
        fn(self, act)

    # ----- handlers (one per action type) -----

    def _h_set_batchmode(self, act):
        self.k.setBatchmode(act.get("value", True))

    def _h_create_document(self, act):
        doc = self.k.createDocument(
            act["width"], act["height"], act.get("name", "Untitled"),
            act.get("color_model", "RGBA"), act.get("color_depth", "U8"),
            act.get("profile", "sRGB-elle-V2-srgbtrc.icc"),
            act.get("resolution", 72.0),
        )
        self.k.activeWindow().addView(doc)
        self.symbols[act.get("ref", "doc:main")] = doc

    def _h_set_active_document(self, act):
        doc = self._doc(act.get("ref", "doc:main"))
        if doc:
            self.k.setActiveDocument(doc)

    def _h_set_background_color(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        qc = _qcolor_from_hex(act["color"].get("hex", "#000000"))
        doc.setBackgroundColor(qc)

    def _h_set_color_space(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        doc.setColorSpace(act["color_model"], act["color_depth"], act["profile"])

    def _h_resize_image(self, act):
        self._doc(act.get("doc_ref", "doc:main")).resizeImage(
            act["x"], act["y"], act["w"], act["h"])

    def _h_crop_image(self, act):
        self._doc(act.get("doc_ref", "doc:main")).crop(
            act["x"], act["y"], act["w"], act["h"])

    def _h_rotate_image(self, act):
        self._doc(act.get("doc_ref", "doc:main")).rotateImage(act["radians"])

    def _h_scale_image(self, act):
        self._doc(act.get("doc_ref", "doc:main")).scaleImage(
            act["w"], act["h"], act["xres"], act["yres"], act.get("strategy", "Bicubic"))

    def _h_flatten(self, act):
        self._doc(act.get("doc_ref", "doc:main")).flatten()

    def _h_refresh_projection(self, act):
        self._doc(act.get("doc_ref", "doc:main")).refreshProjection()

    def _h_wait_for_done(self, act):
        self._doc(act.get("doc_ref", "doc:main")).waitForDone()

    def _h_save(self, act):
        self._doc(act.get("doc_ref", "doc:main")).save()

    def _h_save_as(self, act):
        self._doc(act.get("doc_ref", "doc:main")).saveAs(act["filename"])

    def _h_export_image(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        info = InfoObject()
        for k, v in (act.get("export_config") or {}).items():
            info.setProperty(k, v)
        doc.exportImage(act["filename"], info)

    # ----- node tree -----

    def _h_create_node(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        node = doc.createNode(act.get("name", "Layer"),
                              act.get("node_type", "paintlayer"))
        parent_ref = act.get("parent_ref")
        parent = self.symbols.get(parent_ref) if parent_ref else doc.rootNode()
        above = self.symbols.get(act.get("above_ref")) if act.get("above_ref") else None
        parent.addChildNode(node, above)
        self.symbols[act.get("ref", "node:new")] = node

    def _h_set_active_node(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        node = self._node(act.get("node_ref"))
        if node and doc:
            doc.setActiveNode(node)

    def _h_set_node_opacity(self, act):
        self._node(act["node_ref"]).setOpacity(int(act["value"]))

    def _h_set_node_blending_mode(self, act):
        self._node(act["node_ref"]).setBlendingMode(act["mode"])

    def _h_set_node_visible(self, act):
        self._node(act["node_ref"]).setVisible(bool(act["visible"]))

    def _h_set_node_locked(self, act):
        self._node(act["node_ref"]).setLocked(bool(act["locked"]))

    def _h_set_node_alpha_locked(self, act):
        self._node(act["node_ref"]).setAlphaLocked(bool(act["locked"]))

    def _h_set_node_inherit_alpha(self, act):
        self._node(act["node_ref"]).setInheritAlpha(bool(act["inherit"]))

    def _h_set_node_color_label(self, act):
        self._node(act["node_ref"]).setColorLabel(int(act["index"]))

    def _h_move_node(self, act):
        self._node(act["node_ref"]).move(int(act["x"]), int(act["y"]))

    def _h_rotate_node(self, act):
        self._node(act["node_ref"]).rotateNode(float(act["radians"]))

    def _h_scale_node(self, act):
        self._node(act["node_ref"]).scaleNode(
            QPointF(act["origin_x"], act["origin_y"]),
            act["width"], act["height"], act.get("strategy", "Bicubic"))

    def _h_duplicate_node(self, act):
        n = self._node(act["node_ref"]).duplicate()
        self.symbols[act.get("new_ref", "node:copy")] = n

    def _h_merge_down(self, act):
        self._node(act["node_ref"]).mergeDown()

    def _h_remove_node(self, act):
        self._node(act["node_ref"]).remove()

    def _h_set_fill_layer_generator(self, act):
        node = self._node(act["node_ref"])
        info = InfoObject()
        for k, v in (act.get("config") or {}).items():
            info.setProperty(k, v)
        node.setGenerator(act["generator_name"], info)

    def _h_add_shapes_from_svg(self, act):
        self._node(act["node_ref"]).addShapesFromSvg(act["svg"])

    # ----- view / painting context -----

    def _h_set_foreground_color(self, act):
        v = self._view(act.get("view_ref", "view:active"))
        v.setForeGroundColor(_color_from_spec(act["color"]))

    def _h_set_view_background_color(self, act):
        v = self._view(act.get("view_ref", "view:active"))
        v.setBackGroundColor(_color_from_spec(act["color"]))

    def _h_set_brush_preset(self, act):
        name = act["preset_name"]
        presets = self.k.resources("preset")
        resource = presets.get(name)
        if resource is None:
            print(f"[runner] brush preset '{name}' not found. Available: {list(presets.keys())[:5]}")
            return
        self._view(act.get("view_ref", "view:active")).setCurrentBrushPreset(resource)

    def _h_set_brush_size(self, act):
        self._view(act.get("view_ref", "view:active")).setBrushSize(float(act["size"]))

    def _h_set_brush_rotation(self, act):
        self._view(act.get("view_ref", "view:active")).setBrushRotation(float(act["rotation"]))

    def _h_set_brush_fade(self, act):
        self._view(act.get("view_ref", "view:active")).setBrushFade(float(act["fade"]))

    def _h_set_painting_opacity(self, act):
        self._view(act.get("view_ref", "view:active")).setPaintingOpacity(float(act["opacity"]))

    def _h_set_painting_flow(self, act):
        self._view(act.get("view_ref", "view:active")).setPaintingFlow(float(act["flow"]))

    def _h_set_view_blending_mode(self, act):
        self._view(act.get("view_ref", "view:active")).setCurrentBlendingMode(act["mode"])

    def _h_set_eraser_mode(self, act):
        self._view(act.get("view_ref", "view:active")).setEraserMode(bool(act["enabled"]))

    def _h_set_disable_pressure(self, act):
        self._view(act.get("view_ref", "view:active")).setDisablePressure(bool(act["disabled"]))

    def _h_set_current_pattern(self, act):
        patterns = self.k.resources("pattern")
        r = patterns.get(act["pattern_name"])
        if r is None:
            print(f"[runner] pattern '{act['pattern_name']}' not installed")
            return
        self._view(act.get("view_ref", "view:active")).setCurrentPattern(r)

    def _h_set_current_gradient(self, act):
        gradients = self.k.resources("gradient")
        r = gradients.get(act["gradient_name"])
        if r is None:
            print(f"[runner] gradient '{act['gradient_name']}' not installed")
            return
        self._view(act.get("view_ref", "view:active")).setCurrentGradient(r)

    # ----- stroke primitives -----

    def _h_paint_line(self, act):
        n = self._node(act["node_ref"])
        n.paintLine(QPointF(*act["p1"]), QPointF(*act["p2"]),
                    float(act["pressure1"]), float(act["pressure2"]),
                    act.get("stroke_style", "ForegroundColor"))

    def _h_paint_path(self, act):
        path = QPainterPath()
        pts = act["path"]
        if not pts:
            return
        path.moveTo(QPointF(*pts[0]))
        for p in pts[1:]:
            path.lineTo(QPointF(*p))
        self._node(act["node_ref"]).paintPath(
            path, act.get("stroke_style", "ForegroundColor"),
            act.get("fill_style", "None"))

    def _h_paint_ellipse(self, act):
        rect = QRectF(act["x"], act["y"], act["w"], act["h"])
        self._node(act["node_ref"]).paintEllipse(
            rect, act.get("stroke_style", "ForegroundColor"),
            act.get("fill_style", "None"))

    def _h_paint_rectangle(self, act):
        rect = QRectF(act["x"], act["y"], act["w"], act["h"])
        self._node(act["node_ref"]).paintRectangle(
            rect, act.get("stroke_style", "ForegroundColor"),
            act.get("fill_style", "None"))

    def _h_paint_polygon(self, act):
        pts = [QPointF(*p) for p in act["points"]]
        self._node(act["node_ref"]).paintPolygon(
            pts, act.get("stroke_style", "ForegroundColor"),
            act.get("fill_style", "None"))

    def _h_paint_strokes(self, act):
        """Expand a path into a chain of paintLine calls."""
        n = self._node(act["node_ref"])
        path = act.get("path") or []
        pressures = act.get("pressures") or [1.0] * len(path)
        style = act.get("stroke_style", "ForegroundColor")

        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]

            pr1 = pressures[i] if i < len(pressures) else 1.0
            pr2 = pressures[i + 1] if i + 1 < len(pressures) else 1.0

            q1 = QPoint(int(round(p1[0])), int(round(p1[1])))
            q2 = QPoint(int(round(p2[0])), int(round(p2[1])))

            n.paintLine(q1, q2, pr1, pr2, style)

    # ----- selection -----

    def _h_selection_rect(self, act):
        sel = self.symbols.get(act.get("selection_ref"))
        if sel is None:
            sel = Selection()
            self.symbols[act.get("selection_ref", "sel:current")] = sel
        sel.select(act["x"], act["y"], act["w"], act["h"], act.get("value", 255))

    def _h_selection_invert(self, act):
        self.symbols[act["selection_ref"]].invert()

    def _h_selection_feather(self, act):
        self.symbols[act["selection_ref"]].feather(int(act["radius"]))

    def _h_selection_grow(self, act):
        self.symbols[act["selection_ref"]].grow(int(act["x_radius"]), int(act["y_radius"]))

    def _h_selection_shrink(self, act):
        self.symbols[act["selection_ref"]].shrink(
            int(act["x_radius"]), int(act["y_radius"]), bool(act.get("edge_lock", False)))

    def _h_selection_clear(self, act):
        self.symbols[act["selection_ref"]].clear()

    def _h_apply_document_selection(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        sel = self.symbols[act["selection_ref"]]
        doc.setSelection(sel)

    # ----- filters -----

    def _h_apply_filter(self, act):
        f = Filter()
        f.setName(act["filter_name"])
        info = InfoObject()
        for k, v in (act.get("config") or {}).items():
            info.setProperty(k, v)
        f.setConfiguration(info)
        n = self._node(act["node_ref"])
        f.apply(n, act.get("x", 0), act.get("y", 0),
                act.get("w", 0), act.get("h", 0))

    # ----- canvas -----

    def _h_set_canvas_zoom(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c: c.setZoomLevel(float(act["zoom"]))

    def _h_set_canvas_rotation(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c: c.setRotation(float(act["angle"]))

    # ----- generic escape hatch -----

    def _h_method_call(self, act):
        if act.get("target_ref") == "noop":
            return  # comment-only action
        target = self.symbols.get(act["target_ref"])
        if target is None:
            print(f"[runner] method_call: unknown target_ref '{act['target_ref']}'")
            return
        method = getattr(target, act["method"], None)
        if method is None:
            print(f"[runner] method_call: {act['target_ref']} has no method '{act['method']}'")
            return
        result = method(*act.get("args", []), **act.get("kwargs", {}))
        if act.get("result_ref"):
            self.symbols[act["result_ref"]] = result

    # ----- document operations (extended) -----

    # Alias: schema.py uses "flatten_document"; old runner used "flatten"
    def _h_flatten_document(self, act):
        self._doc(act.get("doc_ref", "doc:main")).flatten()

    # Alias: schema.py uses "save_document"; old runner used "save"
    def _h_save_document(self, act):
        self._doc(act.get("doc_ref", "doc:main")).save()

    def _h_open_document(self, act):
        doc = self.k.openDocument(act["filename"])
        if doc:
            self.k.activeWindow().addView(doc)
            self.symbols[act.get("ref", "doc:main")] = doc

    def _h_close_document(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        if doc:
            doc.close()

    def _h_set_document_name(self, act):
        self._doc(act.get("doc_ref", "doc:main")).setName(act["name"])

    def _h_set_document_resolution(self, act):
        self._doc(act.get("doc_ref", "doc:main")).setResolution(int(act["ppi"]))

    def _h_set_document_modified(self, act):
        self._doc(act.get("doc_ref", "doc:main")).setModified(bool(act.get("modified", True)))

    def _h_shear_image(self, act):
        self._doc(act.get("doc_ref", "doc:main")).shearImage(
            float(act["angle_x"]), float(act["angle_y"]))

    def _h_set_document_selection(self, act):
        doc = self._doc(act.get("doc_ref", "doc:main"))
        sel = self.symbols[act["selection_ref"]]
        doc.setSelection(sel)

    # ----- node / layer operations (extended) -----

    def _h_set_node_name(self, act):
        self._node(act["node_ref"]).setName(act["name"])

    def _h_set_node_collapsed(self, act):
        self._node(act["node_ref"]).setCollapsed(bool(act.get("collapsed", False)))

    def _h_crop_node(self, act):
        self._node(act["node_ref"]).cropNode(
            int(act["x"]), int(act["y"]), int(act["w"]), int(act["h"]))

    def _h_shear_node(self, act):
        self._node(act["node_ref"]).shearNode(
            float(act["angle_x"]), float(act["angle_y"]))

    def _h_save_node(self, act):
        info = InfoObject()
        for k, v in (act.get("export_config") or {}).items():
            info.setProperty(k, v)
        self._node(act["node_ref"]).save(
            act["filename"],
            float(act.get("x_res", 72.0)),
            float(act.get("y_res", 72.0)),
            info,
        )

    def _h_set_node_color_space(self, act):
        self._node(act["node_ref"]).setColorSpace(
            act["color_model"], act["color_depth"], act["profile"])

    def _h_set_pixel_data(self, act):
        data = base64.b64decode(act["data"])
        self._node(act["node_ref"]).setPixelData(
            data, int(act["x"]), int(act["y"]), int(act["w"]), int(act["h"]))

    # ----- layer-type-specific -----

    def _h_set_group_pass_through(self, act):
        self._node(act["node_ref"]).setPassThroughMode(bool(act.get("pass_through", False)))

    def _h_set_filter_layer_filter(self, act):
        f = Filter()
        f.setName(act["filter_name"])
        info = InfoObject()
        for k, v in (act.get("config") or {}).items():
            info.setProperty(k, v)
        f.setConfiguration(info)
        self._node(act["node_ref"]).setFilter(f)

    # ----- view / painting context (extended) -----

    def _h_set_global_alpha_lock(self, act):
        self._view(act.get("view_ref", "view:active")).setGlobalAlphaLock(
            bool(act.get("enabled", False)))

    def _h_activate_resource(self, act):
        resources = self.k.resources(act["resource_type"])
        r = resources.get(act["resource_name"])
        if r is None:
            print(f"[runner] activate_resource: '{act['resource_name']}' "
                  f"({act['resource_type']}) not found")
            return
        self._view(act.get("view_ref", "view:active")).activateResource(r)

    # ----- canvas controls (extended) -----

    def _h_reset_canvas_zoom(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c:
            c.resetZoom()

    def _h_reset_canvas_rotation(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c:
            c.resetRotation()

    def _h_pan_canvas(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c:
            c.pan(int(act["x"]), int(act["y"]))

    def _h_set_canvas_mirror(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c:
            c.setMirror(bool(act.get("mirror", False)))

    def _h_set_wrap_around_mode(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c:
            c.setWrapAroundMode(bool(act.get("enabled", False)))

    def _h_set_level_of_detail_mode(self, act):
        c = self._canvas(act.get("canvas_ref", "canvas:active"))
        if c:
            c.setLevelOfDetailMode(bool(act.get("enabled", False)))

    # ----- selection (extended) -----

    def _h_selection_all(self, act):
        sel = self.symbols.get(act.get("selection_ref"))
        if sel is None:
            sel = Selection()
            self.symbols[act.get("selection_ref", "sel:current")] = sel
        node = self._node(act.get("node_ref", "node:active"))
        if node:
            sel.selectAll(node, int(act.get("value", 255)))

    def _h_selection_move(self, act):
        self.symbols[act["selection_ref"]].move(int(act["x"]), int(act["y"]))

    def _h_selection_resize(self, act):
        self.symbols[act["selection_ref"]].resize(int(act["w"]), int(act["h"]))

    def _h_selection_contract(self, act):
        self.symbols[act["selection_ref"]].contract(int(act["value"]))

    def _h_selection_erode(self, act):
        self.symbols[act["selection_ref"]].erode()

    def _h_selection_dilate(self, act):
        self.symbols[act["selection_ref"]].dilate()

    def _h_selection_border(self, act):
        self.symbols[act["selection_ref"]].border(
            int(act["x_radius"]), int(act["y_radius"]))

    def _h_selection_smooth(self, act):
        self.symbols[act["selection_ref"]].smooth()

    def _h_selection_add(self, act):
        self.symbols[act["selection_ref"]].add(self.symbols[act["other_ref"]])

    def _h_selection_subtract(self, act):
        self.symbols[act["selection_ref"]].subtract(self.symbols[act["other_ref"]])

    def _h_selection_intersect(self, act):
        self.symbols[act["selection_ref"]].intersect(self.symbols[act["other_ref"]])

    def _h_selection_xor(self, act):
        self.symbols[act["selection_ref"]].symmetricdifference(
            self.symbols[act["other_ref"]])

    def _h_selection_duplicate(self, act):
        sel = self.symbols[act["selection_ref"]].duplicate()
        self.symbols[act.get("new_ref", "sel:copy")] = sel

    def _h_create_selection(self, act):
        sel = Selection()
        self.symbols[act.get("ref", "sel:new")] = sel

    def _h_copy_from_node(self, act):
        self.symbols[act["selection_ref"]].copy(self._node(act["node_ref"]))

    def _h_cut_from_node(self, act):
        self.symbols[act["selection_ref"]].cut(self._node(act["node_ref"]))

    def _h_paste_to_node(self, act):
        self.symbols[act["selection_ref"]].paste(
            self._node(act["node_ref"]),
            int(act.get("x", 0)),
            int(act.get("y", 0)),
        )

    # ----- filters (extended) -----

    def _h_start_filter(self, act):
        f = Filter()
        f.setName(act["filter_name"])
        info = InfoObject()
        for k, v in (act.get("config") or {}).items():
            info.setProperty(k, v)
        f.setConfiguration(info)
        f.startFilter(
            self._node(act["node_ref"]),
            int(act.get("x", 0)), int(act.get("y", 0)),
            int(act.get("w", 0)), int(act.get("h", 0)),
        )

    # ----- handler registry -----

    _handlers: Dict[str, Any] = {}


# Populate the handler registry — done after class body so methods exist.
for _name in dir(Runner):
    if _name.startswith("_h_"):
        Runner._handlers[_name[3:]] = getattr(Runner, _name)


def run_plan(plan_path: str) -> None:
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    Runner().run(plan)
