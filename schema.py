"""
Action plan schema.

An action plan is an ordered list of `Action` records. Each record names an
operation that the runner translates into a libkis call. The schema mirrors
the libkis surface (Krita, Document, Node, View, Window, Canvas, Channel,
ManagedColor, Selection, Filter, VectorLayer, GroupLayer, CloneLayer, PaintLayer,
FileLayer, FilterLayer, FillLayer, FilterMask, SelectionMask, TransparencyMask,
TransformMask, ColorizeMask, Palette, Swatch, Scratchpad, Resource, Shape,
InfoObject, GuidesConfig, GridConfig, ColorManager, ColorModel, ColorDepth,
ColorProfile, DockWidget, DockWidgetFactoryBase, Extension, FileDialog,
PresetChooser, PaletteView, IntParseSpinBox, DoubleParseSpinBox,
SliderSpinBox, AngleSelector, ...) so the runner can dispatch every action to a
concrete C++-backed method exposed via PyKrita. Some of those are object types
that are reachable via MethodCall rather than dedicated typed actions.

Two layers:

1. **Typed actions**: ~230 operations (create_document, create_node,
   set_foreground_color, paint_line, set_brush_preset, set_node_name, etc.)
   with explicit params. These are what the synthesizer emits by default.

2. **Generic escape hatch**: `method_call` — names any object reference and
   any libkis method, with positional/keyword args. Lets the planner reach
   the long tail of the API without bloating the schema.

Object references in the plan are symbolic strings (e.g., "doc:main",
"node:sky_layer", "view:active", "color:cobalt"). The runner maintains a
symbol table mapping these to live libkis objects.

Constants defined here (BLENDING_MODES, BRUSH_ENGINES, PRESET_CATALOG,
STROKE_STYLES, FILL_STYLES, SCALE_STRATEGIES, GRID_TYPES, LINE_TYPES,
NODE_TYPES, FILE_LAYER_SCALE_METHODS, COLOR_MODELS, COLOR_DEPTHS,
FILTER_NAMES, GENERATOR_NAMES, RESOURCE_TYPES, GRADIENT_SHAPES,
GRADIENT_REPEATS, SCRATCHPAD_MODES) are the authoritative source for valid
string values — the synthesizer should validate against them before emitting.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json


# ---------------------------------------------------------------------------
# Primitive value types
# ---------------------------------------------------------------------------

# A color is either a {model, depth, profile, components: [...]} dict (matches
# ManagedColor) or a convenience {hex: "#rrggbb"} dict the runner will expand.
Color = Dict[str, Any]

# A point is [x, y] in image pixels. A path is a list of points.
Point = Tuple[float, float]
Path = List[Point]


# ---------------------------------------------------------------------------
# Validated constant sets
# All string values used in actions that have a fixed domain are listed here.
# ---------------------------------------------------------------------------

# Valid stroke_style / fill_style values for all paint_* actions.
# Source: Node.h paintLine / paintRectangle / paintPath / paintEllipse / paintPolygon.
STROKE_STYLES: frozenset = frozenset({"ForegroundColor", "BackgroundColor", "None"})
FILL_STYLES:   frozenset = frozenset({"ForegroundColor", "BackgroundColor", "Pattern", "None"})

# Valid blending mode identifiers.
# Source: KoCompositeOpRegistry.h (libs/pigment/KoCompositeOpRegistry.h).
BLENDING_MODES: frozenset = frozenset({
    # --- Core compositing ---
    "normal",           # over / normal
    "erase",
    "in",
    "out",
    "alphadarken",
    "destination-in",
    "destination-atop",
    "behind",
    "clear",
    "dissolve",
    "displace",
    "nocomposition",
    "copy",
    "copy_red",
    "copy_green",
    "copy_blue",
    "tangent_normalmap",
    "combine_normal",
    "colorize",
    "bumpmap",
    "pass through",     # not implemented yet in Krita
    "darker color",
    "lighter color",

    # --- Arithmetic ---
    "plus",
    "minus",
    "add",
    "subtract",
    "inverse_subtract",
    "diff",
    "multiply",
    "divide",
    "arc_tangent",
    "geometric_mean",
    "additive_subtractive",
    "negation",

    # --- Modulo group ---
    "modulo",
    "modulo_continuous",
    "divisive_modulo",
    "divisive_modulo_continuous",
    "modulo_shift",
    "modulo_shift_continuous",

    # --- Misc math ---
    "equivalence",
    "allanon",
    "parallel",
    "grain_merge",
    "grain_extract",
    "exclusion",
    "marker",

    # --- Logic (bitwise) ---
    "xor",
    "or",
    "and",
    "nand",
    "nor",
    "xnor",
    "implication",
    "not_implication",
    "converse",
    "not_converse",

    # --- Hard mix family ---
    "hard mix",
    "hard_mix_hdr",
    "hard_mix_photoshop",
    "hard_mix_softer_photoshop",

    # --- Overlay family ---
    "overlay",
    "hard overlay",
    "hard_overlay_hdr",
    "interpolation",
    "interpolation 2x",

    # --- Penumbra family ---
    "penumbra a",
    "penumbra b",
    "penumbra c",
    "penumbra d",

    # --- Darken group ---
    "darken",
    "burn",
    "linear_burn",
    "gamma_dark",
    "shade_ifs_illusions",
    "fog_darken_ifs_illusions",
    "easy burn",

    # --- Lighten group ---
    "lighten",
    "dodge",
    "dodge_hdr",
    "linear_dodge",
    "screen",
    "hard_light",
    "soft_light_ifs_illusions",
    "soft_light_pegtop_delphi",
    "soft_light",           # Photoshop soft light
    "soft_light_svg",
    "gamma_light",
    "gamma_illumination",
    "vivid_light",
    "vivid_light_hdr",
    "flat_light",
    "linear light",
    "pin_light",
    "pnorm_a",
    "pnorm_b",
    "super_light",
    "tint_ifs_illusions",
    "fog_lighten_ifs_illusions",
    "easy dodge",
    "luminosity_sai",
    "greater",

    # --- HSL color model ---
    "hue",
    "color",
    "tint",
    "saturation",
    "inc_saturation",
    "dec_saturation",
    "luminize",
    "inc_luminosity",
    "dec_luminosity",

    # --- HSV color model ---
    "hue_hsv",
    "color_hsv",
    "saturation_hsv",
    "inc_saturation_hsv",
    "dec_saturation_hsv",
    "value",
    "inc_value",
    "dec_value",

    # --- HSL (second set) ---
    "hue_hsl",
    "color_hsl",
    "saturation_hsl",
    "inc_saturation_hsl",
    "dec_saturation_hsl",
    "lightness",
    "inc_lightness",
    "dec_lightness",

    # --- HSI color model ---
    "hue_hsi",
    "color_hsi",
    "saturation_hsi",
    "inc_saturation_hsi",
    "dec_saturation_hsi",
    "intensity",
    "inc_intensity",
    "dec_intensity",

    # --- Special effects ---
    "reflect",
    "glow",
    "freeze",
    "heat",
    "glow_heat",
    "heat_glow",
    "reflect_freeze",
    "freeze_reflect",
    "heat_glow_freeze_reflect_hybrid",

    # --- Lighting ---
    "lambert_lighting",
    "lambert_lighting_gamma2.2",
})

# Valid paintop engine identifiers.
# Source: plugins/paintops/*/[engine]_paintop_plugin.cpp registrations.
BRUSH_ENGINES: frozenset = frozenset({
    "paintbrush",       # Pixel Brush (default engine)
    "smudge",           # Smudge Brush
    "colorsmudge",      # Color Smudge
    "hairybrush",       # Bristle (hairy)
    "sketchbrush",      # Sketch
    "curvebrush",       # Curve
    "spraybrush",       # Spray
    "hatchingbrush",    # Hatching
    "gridbrush",        # Grid
    "particlebrush",    # Particle
    "deformbrush",      # Deform
    "roundmarker",      # Quick Brush
    "tangentnormal",    # Tangent Normal (for normal maps)
    "experimentbrush",  # Experimental
    "duplicate",        # Clone / Duplicate
    "filter",           # Filter Brush
    "mypaintbrush",     # MyPaint Brush (libmypaint, added in Krita 5)
    "dyna",             # Dynamic Brush
})

# Resampling strategy names.
# Source: KisFilterStrategy subclasses; passed to scaleImage / scaleNode / etc.
SCALE_STRATEGIES: frozenset = frozenset({
    "Hermite", "Bicubic", "Box", "Bilinear", "Bell",
    "BSpline", "Lanczos3", "Mitchell",
})

# Valid grid type identifiers for SetGridConfig.
GRID_TYPES: frozenset = frozenset({
    "rectangular", "isometric", "isometric_legacy",
})

# Valid line-style identifiers for guides and grid.
# GuidesConfig uses "dot"; GridConfig uses "dotted" — both are included.
# "none" is valid for GridConfig.lineTypeVertical (disables that line set).
LINE_TYPES: frozenset = frozenset({
    "solid", "dashed", "dotted", "dot", "none",
})

# Valid node_type strings for CreateNode.
# Source: KisLayerUtils / libkis Node.h type() documentation.
# NOTE: "colorizemask" is NOT creatable via Document.createNode() — use
# MethodCall → createColorizeMask(name) instead. It is omitted here so the
# synthesizer does not emit a CreateNode with that type.
NODE_TYPES: frozenset = frozenset({
    "paintlayer", "grouplayer", "filterlayer", "filllayer",
    "vectorlayer", "clonelayer", "filelayer",
    "filtermask", "transparencymask", "selectionmask",
    "transformmask",
})

# Valid scalingMethod values for FileLayer / SetFileLayerProperties.
# FileLayer.h uses "ToImageSize"/"ToImagePPI"; Document.h createFileLayer uses
# "ImageToSize"/"ImageToPPI". Both are accepted; prefer FileLayer.h spellings
# when calling setProperties(), and Document.h spellings when creating via createNode.
FILE_LAYER_SCALE_METHODS: frozenset = frozenset({
    "None",
    "ToImageSize", "ToImagePPI",       # FileLayer.setProperties() spellings
    "ImageToSize", "ImageToPPI",       # Document.createFileLayer() spellings
})

# Valid scalingFilter values for FileLayer / SetFileLayerProperties.
FILE_LAYER_SCALE_FILTERS: frozenset = frozenset({
    "Bicubic", "Hermite", "NearestNeighbor", "Bilinear",
    "Bell", "BSpline", "Lanczos3", "Mitchell",
})

# Valid color model identifiers.
# Source: Node.h / Document.h colorModel() documentation and Krita.colorModels().
COLOR_MODELS: frozenset = frozenset({
    "A",      # Alpha mask
    "RGBA",   # RGB with alpha (most common; internal channel order is often BGR!)
    "XYZA",   # XYZ with alpha
    "LABA",   # LAB with alpha
    "CMYKA",  # CMYK with alpha
    "GRAYA",  # Grayscale with alpha
    "YCbCrA", # YCbCr with alpha
})

# Valid color depth identifiers.
# Source: Node.h / Document.h colorDepth() documentation.
COLOR_DEPTHS: frozenset = frozenset({
    "U8",   # 8-bit unsigned integer (most common)
    "U16",  # 16-bit unsigned integer
    "F16",  # 16-bit float / half (requires OpenEXR build)
    "F32",  # 32-bit float
})

# Valid filter name identifiers passed to Krita.filter(name) / ApplyFilter / StartFilter.
# Source: Filter.h comment block (registered filter plugin IDs).
# NOTE: names are lowercase with spaces where shown — exact spelling is required.
FILTER_NAMES: frozenset = frozenset({
    "autocontrast",
    "blur",
    "bottom edge detections",
    "brightnesscontrast",
    "burn",
    "colorbalance",
    "colortoalpha",
    "colortransfer",
    "desaturate",
    "dodge",
    "emboss",
    "emboss all directions",
    "emboss horizontal and vertical",
    "emboss horizontal only",
    "emboss laplascian",
    "emboss vertical only",
    "gaussian blur",
    "gaussiannoisereducer",
    "gradientmap",
    "halftone",
    "hsvadjustment",
    "indexcolors",
    "invert",
    "left edge detections",
    "lens blur",
    "levels",
    "maximize",
    "mean removal",
    "minimize",
    "motion blur",
    "noise",
    "normalize",
    "oilpaint",
    "perchannel",
    "phongbumpmap",
    "pixelize",
    "posterize",
    "raindrops",
    "randompick",
    "right edge detections",
    "roundcorners",
    "sharpen",
    "smalltiles",
    "sobel",
    "threshold",
    "top edge detections",
    "unsharp",
    "wave",
    "waveletnoisereducer",
})

# Valid generator names for SetFillLayerGenerator / Document.createFillLayer().
# Source: plugins/generators/*/
GENERATOR_NAMES: frozenset = frozenset({
    "color",    # solid color fill
    "pattern",  # tiled pattern fill
    "gradient", # gradient fill
    "seexpr",   # SeExpr scripted fill (Krita 5+)
})

# Valid resource type strings for ActivateResource / Krita.resources().
# Source: Krita.h resources() documentation.
RESOURCE_TYPES: frozenset = frozenset({
    "paintoppresets",
    "patterns",
    "gradients",
    "palettes",
    "workspaces",
    "brushes",
})

# Valid gradient shape identifiers for FillScratchpadGradient.
# Source: Scratchpad.h fillGradient() documentation.
GRADIENT_SHAPES: frozenset = frozenset({
    "linear", "bilinear", "radial", "square",
    "conical", "conicalSymmetric",
    "spiral", "reverseSpiral", "polygonal",
})

# Valid gradient repeat modes for FillScratchpadGradient.
# Source: Scratchpad.h fillGradient() documentation.
GRADIENT_REPEATS: frozenset = frozenset({
    "none", "alternate", "forwards",
})

# Valid mode strings for SetScratchpadMode.
# Source: Scratchpad.h setMode() documentation.
SCRATCHPAD_MODES: frozenset = frozenset({
    "painting", "panning", "colorsampling",
})

# Known libkis classes used by the schema.
# Not every class has a dedicated typed action; some are intentionally
# reachable only through MethodCall or through existing object-returning APIs.
LIBKIS_CLASSES: frozenset = frozenset({
    "Canvas", "Channel", "ColorDepth", "ColorManager", "ColorModel",
    "ColorProfile", "DockWidget", "DockWidgetFactoryBase", "Document",
    "Extension", "Filter", "InfoObject", "Krita", "Node", "Notifier",
    "Resource", "Scratchpad", "Selection", "View", "Window", "Shape",
    "GroupShape", "PaintLayer", "CloneLayer", "GroupLayer", "FilterLayer",
    "FillLayer", "FileLayer", "VectorLayer", "FilterMask",
    "SelectionMask", "TransparencyMask", "TransformMask", "ColorizeMask",
    "Palette", "Swatch", "Preset", "PresetChooser", "PaletteView",
    "GuidesConfig", "GridConfig", "FileDialog",
    "DoubleParseSpinBox", "IntParseSpinBox", "SliderSpinBox",
    "AngleSelector",
})

# Catalog of brush preset names shipped with Krita.
#
# Keys confirmed from krita/data/paintoppresets/*.kpp (in-tree) and
# krita/data/bundles/Krita_4_Default_Resources.bundle.
# Pass any of these strings to SetBrushPreset.preset_name.
PRESET_CATALOG: Dict[str, List[str]] = {
    # a) Erasers
    "erasers": [
        "a) Eraser Circle",
        "a) Eraser Soft",
        "a) Eraser Small Soft",
        "a) Eraser Big Soft",
    ],

    # b) Basic
    "basic": [
        "b) Basic-5 Size",
        "b) Basic-5 Size Opacity",
        "b) Basic Wet",
        "b) Basic-5 Size Blur",
        "b) Basic Paint",
        "b) Basic Texture",
    ],

    # c) Charcoal / Chalk
    "charcoal": [
        "c) Charcoal Basic",
        "c) Chalk and Eraser",
        "c) Chalk-1",
        "c) Charcoal Texture",
    ],

    # d) Digital painting
    "digital": [
        "d) Digital Basic",
        "d) Digital Flat",
        "d) Digital Chisel Hard",
        "d) Digital Painting Hard Round",
        "d) Digital Painting Soft Round",
    ],

    # e) Effects
    "effects": [
        "e) Bristles-2 Flat Rough",
        "e) Bristles Dry",
        "e) Bristles Wet",
        "e) Bristles Opaque",
    ],

    # f) Fur / Fiber
    "fur": [
        "f) Fur-1",
        "f) Fur-2 Rake",
        "f) Fiber Clumpy",
    ],

    # g) Geometric
    "geometric": [
        "g) Grid brush",
        "g) Spray dots",
        "g) Spray basic",
    ],

    # h) Hatching
    "hatching": [
        "h) Hatching-1",
        "h) Hatching Dense",
        "h) Hatching Cross",
    ],

    # i) Inking
    "inking": [
        "i) Ink-2 Fineliner",
        "i) Ink-3 Gpen",
        "i) Ink-4 Nib Medium",
        "i) Ink-5 Brush Tip",
        "i) Ink-7 Brush Rough",
        "i) Inking",
    ],

    # j) Watercolor
    "watercolor": [
        "j) WaterC Basic Lines-Dry",
        "j) WaterC Basic Lines-Wet",
        "j) WaterC Basic Lines-Wet-Pattern",
        "j) WaterC Basic Round-Fringe 02",
        "j) WaterC Basic Round-Grain",
        "j) WaterC Basic Round-Grunge",
        "j) WaterC Flat Big-Grain Tilt",
        "j) WaterC Flat Decay Tilt",
        "j) WaterC Special Blobs",
        "j) WaterC Special Splats",
        "j) WaterC Spread",
        "j) WaterC Spread-Pattern",
        "j) WaterC Spread WideArea",
        "j) WaterC Water-Pattern",
    ],

    # p) Pencil
    "pencil": [
        "p) Pencil-1",
        "p) Pencil-2 Thick",
        "p) Pencil Soft",
        "p) Pencil Textured",
    ],

    # t) Texture / Stamp
    "texture": [
        "t) Texture - Sponge",
        "t) Basic Texture Soft Large",
        "t) Stamp - Foliage",
        "t) Stamp - Grass Clump",
        "t) Stamp - Leaves",
        "t) Stamp - Rocks",
        "t) Stamp - Scales",
        "t) Stamp - Splatter",
        "t) Stamp - Star",
    ],

    # w) Wet media
    "wet": [
        "w) Wet Paint",
        "w) Wet Blend",
        "w) Wet Painting Round",
    ],

    # Special / engine defaults (from plugins/paintops/defaultpresets/)
    "engine_defaults": [
        "colorsmudge",
        "curvebrush",
        "deformbrush",
        "duplicate",
        "eraser",
        "experimentbrush",
        "filter",
        "gridbrush",
        "hairybrush",
        "hatchingbrush",
        "paintbrush",
        "particlebrush",
        "roundmarker",
        "sketchbrush",
        "smudge",
        "spraybrush",
        "tangentnormal",
    ],
}


# ---------------------------------------------------------------------------
# Action base
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """Base class. Every action has a `type` and an optional `comment`."""
    type: str
    comment: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Strips empty strings, None, empty lists, and empty dicts.
        # False and 0 are intentionally kept — they are valid field values.
        return {k: v for k, v in d.items() if v not in ("", None, [], {})}


# ---------------------------------------------------------------------------
# Document / canvas lifecycle
# ---------------------------------------------------------------------------

@dataclass
class OpenDocument(Action):
    """Krita.instance().openDocument(filename) — loads a file into Krita."""
    type: str = "open_document"
    ref: str = "doc:main"
    filename: str = ""


@dataclass
class CreateDocument(Action):
    type: str = "create_document"
    ref: str = "doc:main"
    width: int = 1024
    height: int = 768
    name: str = "Untitled"
    color_model: str = "RGBA"           # see Krita.colorModels()
    color_depth: str = "U8"             # see Krita.colorDepths(model)
    profile: str = "sRGB-elle-V2-srgbtrc.icc"
    resolution: float = 72.0


@dataclass
class CloneDocument(Action):
    """doc.clone() — creates an independent copy of the document."""
    type: str = "clone_document"
    doc_ref: str = "doc:main"
    ref: str = "doc:clone"


@dataclass
class SetActiveDocument(Action):
    type: str = "set_active_document"
    ref: str = "doc:main"


@dataclass
class SetBatchmode(Action):
    """Suppress dialogs. Recommended True for batch generation."""
    type: str = "set_batchmode"
    value: bool = True


@dataclass
class GetActiveDocument(Action):
    """Krita.instance().activeDocument() — captures the currently active document."""
    type: str = "get_active_document"
    ref: str = "doc:active"


@dataclass
class GetActiveWindow(Action):
    """Krita.instance().activeWindow() — captures the currently active window."""
    type: str = "get_active_window"
    ref: str = "window:active"


@dataclass
class SetDocumentName(Action):
    """doc.setName(str) — changes the display title."""
    type: str = "set_document_name"
    doc_ref: str = "doc:main"
    name: str = "Untitled"


@dataclass
class SetDocumentResolution(Action):
    """doc.setResolution(int) — pixels per inch, does not rescale pixels."""
    type: str = "set_document_resolution"
    doc_ref: str = "doc:main"
    ppi: int = 72


@dataclass
class SetDocumentColorProfile(Action):
    """doc.setColorProfile(str) — assigns ICC profile without converting pixels."""
    type: str = "set_document_color_profile"
    doc_ref: str = "doc:main"
    profile: str = "sRGB-elle-V2-srgbtrc.icc"


@dataclass
class SetBackgroundColor(Action):
    type: str = "set_background_color"
    doc_ref: str = "doc:main"
    color: Color = field(default_factory=lambda: {"hex": "#0a1844"})


@dataclass
class SetColorSpace(Action):
    type: str = "set_color_space"
    doc_ref: str = "doc:main"
    color_model: str = "RGBA"
    color_depth: str = "U8"
    profile: str = "sRGB-elle-V2-srgbtrc.icc"


@dataclass
class SetDocumentInfo(Action):
    """doc.setDocumentInfo(xml) — stores Dublin-Core / Krita metadata as XML."""
    type: str = "set_document_info"
    doc_ref: str = "doc:main"
    xml: str = ""


@dataclass
class SetDocumentWidth(Action):
    """doc.setWidth(int) — directly sets the canvas width in pixels.
    Unlike resize_image, this does not reposition content.
    """
    type: str = "set_document_width"
    doc_ref: str = "doc:main"
    width: int = 1024


@dataclass
class SetDocumentHeight(Action):
    """doc.setHeight(int) — directly sets the canvas height in pixels.
    Unlike resize_image, this does not reposition content.
    """
    type: str = "set_document_height"
    doc_ref: str = "doc:main"
    height: int = 768


@dataclass
class SetDocumentFileName(Action):
    """doc.setFileName(str) — changes where the document saves on disk."""
    type: str = "set_document_filename"
    doc_ref: str = "doc:main"
    filename: str = ""


@dataclass
class SetDocumentOffset(Action):
    """doc.setXOffset(x) / setYOffset(y) — shifts the canvas origin in pixels."""
    type: str = "set_document_offset"
    doc_ref: str = "doc:main"
    x: int = 0
    y: int = 0


@dataclass
class SetDocumentXYRes(Action):
    """doc.setXRes(float) / setYRes(float) — sets horizontal and vertical DPI
    independently (distinct from setResolution which sets both equally).
    """
    type: str = "set_document_xy_res"
    doc_ref: str = "doc:main"
    x_res: float = 72.0
    y_res: float = 72.0


@dataclass
class SetDocumentBatchmode(Action):
    """doc.setBatchmode(bool) — suppresses dialogs for this specific document.
    Distinct from the Krita-global SetBatchmode.
    """
    type: str = "set_document_batchmode"
    doc_ref: str = "doc:main"
    value: bool = True


@dataclass
class SetAutosave(Action):
    """doc.setAutosave(bool) — enables or disables auto-save for this document."""
    type: str = "set_autosave"
    doc_ref: str = "doc:main"
    active: bool = True


@dataclass
class LockDocument(Action):
    """doc.lock() — acquires the document write lock (must be paired with unlock)."""
    type: str = "lock_document"
    doc_ref: str = "doc:main"


@dataclass
class UnlockDocument(Action):
    """doc.unlock() — releases the document write lock."""
    type: str = "unlock_document"
    doc_ref: str = "doc:main"


@dataclass
class ResizeImage(Action):
    type: str = "resize_image"
    doc_ref: str = "doc:main"
    x: int = 0
    y: int = 0
    w: int = 1024
    h: int = 768


@dataclass
class CropImage(Action):
    type: str = "crop_image"
    doc_ref: str = "doc:main"
    x: int = 0
    y: int = 0
    w: int = 1024
    h: int = 768


@dataclass
class RotateImage(Action):
    type: str = "rotate_image"
    doc_ref: str = "doc:main"
    radians: float = 0.0


@dataclass
class ShearImage(Action):
    """doc.shearImage(angleX, angleY) — shears the whole image. Angles in degrees."""
    type: str = "shear_image"
    doc_ref: str = "doc:main"
    angle_x: float = 0.0
    angle_y: float = 0.0


@dataclass
class ScaleImage(Action):
    """strategy must be a value from SCALE_STRATEGIES."""
    type: str = "scale_image"
    doc_ref: str = "doc:main"
    w: int = 1024
    h: int = 768
    xres: int = 72
    yres: int = 72
    strategy: str = "Bicubic"


@dataclass
class FlattenDocument(Action):
    type: str = "flatten_document"
    doc_ref: str = "doc:main"


@dataclass
class RefreshProjection(Action):
    type: str = "refresh_projection"
    doc_ref: str = "doc:main"


@dataclass
class WaitForDone(Action):
    type: str = "wait_for_done"
    doc_ref: str = "doc:main"


@dataclass
class TryBarrierLock(Action):
    """doc.tryBarrierLock() — attempts to acquire the write lock without blocking.
    Returns immediately: if background jobs are still running it returns False.
    result_ref stores the bool result ("true"/"false" string) for runner inspection.
    """
    type: str = "try_barrier_lock"
    doc_ref: str = "doc:main"
    result_ref: Optional[str] = "lock:result"


@dataclass
class CloseDocument(Action):
    """doc.close() — removes document from Krita's registry."""
    type: str = "close_document"
    doc_ref: str = "doc:main"


@dataclass
class SetDocumentModified(Action):
    """doc.setModified(bool) — controls whether Krita prompts to save on close."""
    type: str = "set_document_modified"
    doc_ref: str = "doc:main"
    modified: bool = True


@dataclass
class SaveDocument(Action):
    type: str = "save_document"
    doc_ref: str = "doc:main"


@dataclass
class SaveAs(Action):
    type: str = "save_as"
    doc_ref: str = "doc:main"
    filename: str = "output.kra"


@dataclass
class ExportImage(Action):
    type: str = "export_image"
    doc_ref: str = "doc:main"
    filename: str = "output.png"
    export_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetDocumentSelection(Action):
    """doc.setSelection(selection) — replaces the document-wide global selection."""
    type: str = "set_document_selection"
    doc_ref: str = "doc:main"
    selection_ref: str = "sel:current"


@dataclass
class SetDocumentAnnotation(Action):
    """doc.setAnnotation(type, description, data) — stores arbitrary metadata bytes."""
    type: str = "set_document_annotation"
    doc_ref: str = "doc:main"
    annotation_type: str = ""
    description: str = ""
    data: str = ""      # base64-encoded bytes


@dataclass
class RemoveDocumentAnnotation(Action):
    """doc.removeAnnotation(type)."""
    type: str = "remove_document_annotation"
    doc_ref: str = "doc:main"
    annotation_type: str = ""


# ---------------------------------------------------------------------------
# Animation (Document)
# ---------------------------------------------------------------------------

@dataclass
class SetFramesPerSecond(Action):
    """doc.setFramesPerSecond(int)."""
    type: str = "set_frames_per_second"
    doc_ref: str = "doc:main"
    fps: int = 24


@dataclass
class SetAnimationRange(Action):
    """doc.setFullClipRangeStartTime / setFullClipRangeEndTime."""
    type: str = "set_animation_range"
    doc_ref: str = "doc:main"
    start_time: int = 0
    end_time: int = 24


@dataclass
class SetPlaybackRange(Action):
    """doc.setPlayBackRange(start, stop) — temporary in-out range for preview."""
    type: str = "set_playback_range"
    doc_ref: str = "doc:main"
    start: int = 0
    stop: int = 24


@dataclass
class SetCurrentTime(Action):
    """doc.setCurrentTime(int) — moves the animation cursor to a frame."""
    type: str = "set_current_time"
    doc_ref: str = "doc:main"
    time: int = 0


@dataclass
class ImportAnimation(Action):
    """doc.importAnimation(files, firstFrame, step) — loads image sequence as frames."""
    type: str = "import_animation"
    doc_ref: str = "doc:main"
    files: List[str] = field(default_factory=list)
    first_frame: int = 0
    step: int = 1


@dataclass
class SetAudioTracks(Action):
    """doc.setAudioTracks(files) — assigns audio files to the animation."""
    type: str = "set_audio_tracks"
    doc_ref: str = "doc:main"
    files: List[str] = field(default_factory=list)


@dataclass
class SetAudioLevel(Action):
    """doc.setAudioLevel(level) — playback volume 0.0..1.0."""
    type: str = "set_audio_level"
    doc_ref: str = "doc:main"
    level: float = 1.0


# ---------------------------------------------------------------------------
# Layer / node tree
# ---------------------------------------------------------------------------

@dataclass
class CreateNode(Action):
    """Creates a node and attaches it under `parent_ref` (None = top of doc).
    node_type must be a value from NODE_TYPES.
    """
    type: str = "create_node"
    ref: str = "node:new"
    doc_ref: str = "doc:main"
    parent_ref: Optional[str] = None
    above_ref: Optional[str] = None
    name: str = "Layer"
    node_type: str = "paintlayer"


@dataclass
class SetActiveNode(Action):
    type: str = "set_active_node"
    doc_ref: str = "doc:main"
    node_ref: str = "node:active"


@dataclass
class SetNodeName(Action):
    """node.setName(str) — renames the layer."""
    type: str = "set_node_name"
    node_ref: str = "node:active"
    name: str = "Layer"


@dataclass
class SetNodeOpacity(Action):
    type: str = "set_node_opacity"
    node_ref: str = "node:active"
    value: int = 255    # 0..255


@dataclass
class SetNodeBlendingMode(Action):
    """node.setBlendingMode(str). Use a value from BLENDING_MODES."""
    type: str = "set_node_blending_mode"
    node_ref: str = "node:active"
    mode: str = "normal"


@dataclass
class SetNodeVisible(Action):
    type: str = "set_node_visible"
    node_ref: str = "node:active"
    visible: bool = True


@dataclass
class SetNodeLocked(Action):
    type: str = "set_node_locked"
    node_ref: str = "node:active"
    locked: bool = False


@dataclass
class SetNodeAlphaLocked(Action):
    type: str = "set_node_alpha_locked"
    node_ref: str = "node:active"
    locked: bool = False


@dataclass
class SetNodeInheritAlpha(Action):
    type: str = "set_node_inherit_alpha"
    node_ref: str = "node:active"
    inherit: bool = False


@dataclass
class SetNodeColorLabel(Action):
    type: str = "set_node_color_label"
    node_ref: str = "node:active"
    index: int = 0      # 0..8 in Krita UI


@dataclass
class SetNodeCollapsed(Action):
    """node.setCollapsed(bool) — collapses/expands a group layer in the UI."""
    type: str = "set_node_collapsed"
    node_ref: str = "node:active"
    collapsed: bool = False


@dataclass
class SetNodePinnedToTimeline(Action):
    """node.setPinnedToTimeline(bool) — keeps layer visible in Timeline docker."""
    type: str = "set_node_pinned_to_timeline"
    node_ref: str = "node:active"
    pinned: bool = True


@dataclass
class MoveNode(Action):
    type: str = "move_node"
    node_ref: str = "node:active"
    x: int = 0
    y: int = 0


@dataclass
class RotateNode(Action):
    """node.rotateNode(radians) — rotates this layer in radians."""
    type: str = "rotate_node"
    node_ref: str = "node:active"
    radians: float = 0.0


@dataclass
class ScaleNode(Action):
    """strategy must be a value from SCALE_STRATEGIES."""
    type: str = "scale_node"
    node_ref: str = "node:active"
    origin_x: float = 0.0
    origin_y: float = 0.0
    width: int = 1024
    height: int = 768
    strategy: str = "Bicubic"


@dataclass
class CropNode(Action):
    """node.cropNode(x, y, w, h) — crops this layer only, not the canvas."""
    type: str = "crop_node"
    node_ref: str = "node:active"
    x: int = 0
    y: int = 0
    w: int = 1024
    h: int = 768


@dataclass
class ShearNode(Action):
    """node.shearNode(angleX, angleY) — shears this layer. Angles in degrees."""
    type: str = "shear_node"
    node_ref: str = "node:active"
    angle_x: float = 0.0
    angle_y: float = 0.0


@dataclass
class DuplicateNode(Action):
    type: str = "duplicate_node"
    node_ref: str = "node:active"
    new_ref: str = "node:copy"


@dataclass
class MergeDown(Action):
    """node.mergeDown() — merges this node with the first visible node below it.
    The merged result is a new Node; store it under result_ref to reference it later.
    """
    type: str = "merge_down"
    node_ref: str = "node:active"
    result_ref: Optional[str] = None


@dataclass
class RemoveNode(Action):
    type: str = "remove_node"
    node_ref: str = "node:active"


@dataclass
class EnableNodeAnimation(Action):
    """node.enableAnimation() — makes this layer animated so it can have frames."""
    type: str = "enable_node_animation"
    node_ref: str = "node:active"


@dataclass
class SetNodeColorSpace(Action):
    """node.setColorSpace(model, depth, profile) — converts this node's color space."""
    type: str = "set_node_color_space"
    node_ref: str = "node:active"
    color_model: str = "RGBA"
    color_depth: str = "U8"
    profile: str = "sRGB-elle-V2-srgbtrc.icc"


@dataclass
class SetNodeColorProfile(Action):
    """node.setColorProfile(str) — assigns ICC profile without converting pixel data."""
    type: str = "set_node_color_profile"
    node_ref: str = "node:active"
    profile: str = "sRGB-elle-V2-srgbtrc.icc"


@dataclass
class SetLayerStyle(Action):
    """node.setLayerStyleFromAsl(str) — applies a layer style from ASL (Photoshop) XML.
    Obtain ASL content via node.layerStyleToAsl() on a styled layer.
    """
    type: str = "set_layer_style"
    node_ref: str = "node:active"
    asl_content: str = ""


@dataclass
class SetPixelData(Action):
    """node.setPixelData(data, x, y, w, h) — writes raw pixel bytes to the layer.
    data is a base64-encoded byte string matching the node's color depth.
    Only works on paintlayer, filtermask, selectionmask, etc. — not groups.
    """
    type: str = "set_pixel_data"
    node_ref: str = "node:active"
    data: str = ""      # base64-encoded raw pixel bytes
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


@dataclass
class SetPixelDataAtTime(Action):
    """node.setPixelData(data, x, y, w, h) at a specific animation frame.
    Runner must call doc.setCurrentTime(time) before writing, then restore.
    """
    type: str = "set_pixel_data_at_time"
    node_ref: str = "node:active"
    data: str = ""      # base64-encoded raw pixel bytes
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    time: int = 0


@dataclass
class SaveNode(Action):
    """node.save(filename, xRes, yRes, config) — exports a single layer to file."""
    type: str = "save_node"
    node_ref: str = "node:active"
    filename: str = "layer.png"
    x_res: float = 72.0
    y_res: float = 72.0
    export_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetFillLayerGenerator(Action):
    """Configures a FillLayer's procedural generator (gradient, color, pattern)."""
    type: str = "set_fill_layer_generator"
    node_ref: str = "node:fill"
    generator_name: str = "color"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AddShapesFromSvg(Action):
    """Drops SVG geometry into a VectorLayer via vectorLayer.addShapesFromSvg(svg)."""
    type: str = "add_shapes_from_svg"
    node_ref: str = "node:vector"
    svg: str = ""


@dataclass
class AddChildNode(Action):
    """node.addChildNode(child, above) — reparents an existing node into `node_ref`.
    `child_ref` is moved; `above_ref` is the sibling it will sit above (None = top).
    Use this to restructure the layer tree without recreating nodes.
    """
    type: str = "add_child_node"
    node_ref: str = "node:group"
    child_ref: str = "node:active"
    above_ref: Optional[str] = None


@dataclass
class RemoveChildNode(Action):
    """node.removeChildNode(child) — detaches a child from a parent node.
    The child is removed from the tree but not destroyed; use remove_node to delete it.
    """
    type: str = "remove_child_node"
    node_ref: str = "node:group"
    child_ref: str = "node:active"


@dataclass
class SetChildNodes(Action):
    """node.setChildNodes(nodes) — replaces the entire ordered child list of a group.
    node_refs is a list of node refs ordered bottom-up (first = bottommost in stack).
    All refs must already exist in the symbol table.
    """
    type: str = "set_child_nodes"
    node_ref: str = "node:group"
    node_refs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer-type-specific operations
# ---------------------------------------------------------------------------

@dataclass
class SetGroupPassThrough(Action):
    """groupLayer.setPassThroughMode(bool) — blends children directly into parent."""
    type: str = "set_group_pass_through"
    node_ref: str = "node:group"
    pass_through: bool = False


@dataclass
class SetCloneSource(Action):
    """cloneLayer.setSourceNode(node) — changes what the clone layer mirrors."""
    type: str = "set_clone_source"
    node_ref: str = "node:clone"
    source_ref: str = "node:active"


@dataclass
class SetFileLayerProperties(Action):
    """fileLayer.setProperties(fileName, scalingMethod, scalingFilter).
    scalingMethod ∈ FILE_LAYER_SCALE_METHODS.
    scalingFilter ∈ FILE_LAYER_SCALE_FILTERS.
    """
    type: str = "set_file_layer_properties"
    node_ref: str = "node:file"
    filename: str = ""
    scaling_method: str = "None"
    scaling_filter: str = "Bicubic"


@dataclass
class ResetFileLayerCache(Action):
    """fileLayer.resetCache() — forces the file layer to reload its source from disk."""
    type: str = "reset_file_layer_cache"
    node_ref: str = "node:file"


@dataclass
class SetFilterLayerFilter(Action):
    """filterLayer.setFilter(filter) — replaces the filter applied by a filter layer."""
    type: str = "set_filter_layer_filter"
    node_ref: str = "node:filterlayer"
    filter_name: str = "blur"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetFilterMaskFilter(Action):
    """filterMask.setFilter(filter) — replaces the filter applied by a filter mask."""
    type: str = "set_filter_mask_filter"
    node_ref: str = "node:filtermask"
    filter_name: str = "blur"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetSelectionMaskSelection(Action):
    """selectionMask.setSelection(selection) — binds a Selection to a selection mask."""
    type: str = "set_selection_mask_selection"
    node_ref: str = "node:selectionmask"
    selection_ref: str = "sel:current"


@dataclass
class SetTransparencyMaskSelection(Action):
    """transparencyMask.setSelection(selection) — defines the mask from a Selection."""
    type: str = "set_transparency_mask_selection"
    node_ref: str = "node:transparencymask"
    selection_ref: str = "sel:current"


@dataclass
class SetTransformMaskXML(Action):
    """transformMask.fromXML(xml) — applies a transform defined as Krita XML.
    Obtain valid XML via transformMask.toXML() on a configured mask.
    """
    type: str = "set_transform_mask_xml"
    node_ref: str = "node:transformmask"
    xml: str = ""


@dataclass
class SetColorizeMaskSettings(Action):
    """Configures the colorize mask edge-detection and cleanup parameters.
    clean_up_amount is on a 0.0–100.0 scale (libkis convention), not 0.0–1.0.
    """
    type: str = "set_colorize_mask_settings"
    node_ref: str = "node:colorizemask"
    use_edge_detection: bool = True
    edge_detection_size: float = 4.0
    clean_up_amount: float = 70.0
    limit_to_device_bounds: bool = False
    show_output: bool = True


@dataclass
class InitColorizeMaskColors(Action):
    """colorizeMask.initializeKeyStrokeColors(colors, transparentIndex) — seeds
    key-stroke color slots. transparent_index (-1 = none) marks which slot is
    the transparent/eraser key stroke.
    """
    type: str = "init_colorize_mask_colors"
    node_ref: str = "node:colorizemask"
    colors: List[Color] = field(default_factory=list)
    transparent_index: int = -1


@dataclass
class SetColorizeMaskKeyStroke(Action):
    """colorizeMask.setKeyStrokePixelData(data, color, x, y, w, h) — writes
    the key-stroke mask for one color slot. data is base64-encoded pixel bytes.
    """
    type: str = "set_colorize_mask_keystroke"
    node_ref: str = "node:colorizemask"
    color: Color = field(default_factory=lambda: {"hex": "#ff0000"})
    data: str = ""      # base64-encoded grayscale mask bytes
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


@dataclass
class UpdateColorizeMask(Action):
    """colorizeMask.updateMask() — triggers a re-colorize pass."""
    type: str = "update_colorize_mask"
    node_ref: str = "node:colorizemask"


@dataclass
class ResetColorizeMaskCache(Action):
    """colorizeMask.resetCache() — clears internal cache, forcing full recalculation."""
    type: str = "reset_colorize_mask_cache"
    node_ref: str = "node:colorizemask"


@dataclass
class SetColorizeMaskEditKeyStrokes(Action):
    """colorizeMask.setEditKeyStrokes(bool) — toggles key-stroke editing mode.
    True = painting strokes edits key-stroke layer; False = viewing output.
    """
    type: str = "set_colorize_mask_edit_keystrokes"
    node_ref: str = "node:colorizemask"
    enabled: bool = True


@dataclass
class RemoveColorizeMaskKeyStroke(Action):
    """colorizeMask.removeKeyStroke(color) — deletes the key-stroke slot for a color."""
    type: str = "remove_colorize_mask_keystroke"
    node_ref: str = "node:colorizemask"
    color: Color = field(default_factory=lambda: {"hex": "#ff0000"})


@dataclass
class SetVectorLayerAntialiased(Action):
    """vectorLayer.setAntialiased(bool) — controls shape edge anti-aliasing."""
    type: str = "set_vector_layer_antialiased"
    node_ref: str = "node:vector"
    antialiased: bool = True


@dataclass
class CreateGroupShape(Action):
    """vectorLayer.createGroupShape(name, shapes) — combines existing top-level
    shapes into a GroupShape. shape_refs must be refs of shapes already in the layer.
    Stores the resulting GroupShape under result_ref.
    """
    type: str = "create_group_shape"
    node_ref: str = "node:vector"
    name: str = "Group"
    shape_refs: List[str] = field(default_factory=list)
    result_ref: Optional[str] = "shape:group"


# ---------------------------------------------------------------------------
# View / painting context  (brush, color, opacity)
# ---------------------------------------------------------------------------

@dataclass
class SetForegroundColor(Action):
    """view.setForeGroundColor(color) — note capital G in the libkis method name."""
    type: str = "set_foreground_color"
    view_ref: str = "view:active"
    color: Color = field(default_factory=lambda: {"hex": "#ffffff"})


@dataclass
class SetViewBackgroundColor(Action):
    """view.setBackGroundColor(color) — note capital G in the libkis method name."""
    type: str = "set_view_background_color"
    view_ref: str = "view:active"
    color: Color = field(default_factory=lambda: {"hex": "#000000"})


@dataclass
class SetBrushPreset(Action):
    """Selects a brush by resource name via Krita.resources('paintoppresets')[name].
    Use a name from PRESET_CATALOG for guaranteed availability.
    """
    type: str = "set_brush_preset"
    view_ref: str = "view:active"
    preset_name: str = "b) Basic-5 Size"


@dataclass
class ModifyPreset(Action):
    """Modifies settings of the currently active brush preset via Preset.fromXML().
    `xml` should be a full valid Krita preset XML string (obtain via Preset.toXML()).
    `engine` must be a value from BRUSH_ENGINES.
    """
    type: str = "modify_preset"
    view_ref: str = "view:active"
    xml: str = ""
    engine: str = "paintbrush"


@dataclass
class SetBrushSize(Action):
    """view.setBrushSize(pixels) — overrides the preset's size at runtime."""
    type: str = "set_brush_size"
    view_ref: str = "view:active"
    size: float = 25.0


@dataclass
class SetBrushRotation(Action):
    """view.setBrushRotation(degrees) — tip rotation override."""
    type: str = "set_brush_rotation"
    view_ref: str = "view:active"
    rotation: float = 0.0


@dataclass
class SetBrushFade(Action):
    """view.setBrushFade(0.0–1.0) — how quickly the brush fades along a stroke."""
    type: str = "set_brush_fade"
    view_ref: str = "view:active"
    fade: float = 1.0


@dataclass
class SetPatternSize(Action):
    """view.setPatternSize(float) — controls the current pattern tile size."""
    type: str = "set_pattern_size"
    view_ref: str = "view:active"
    size: float = 1.0


@dataclass
class SetPaintingOpacity(Action):
    type: str = "set_painting_opacity"
    view_ref: str = "view:active"
    opacity: float = 1.0


@dataclass
class SetPaintingFlow(Action):
    type: str = "set_painting_flow"
    view_ref: str = "view:active"
    flow: float = 1.0


@dataclass
class SetViewBlendingMode(Action):
    """Brush blending mode. Use a value from BLENDING_MODES."""
    type: str = "set_view_blending_mode"
    view_ref: str = "view:active"
    mode: str = "normal"


@dataclass
class SetEraserMode(Action):
    type: str = "set_eraser_mode"
    view_ref: str = "view:active"
    enabled: bool = False


@dataclass
class SetGlobalAlphaLock(Action):
    """view.setGlobalAlphaLock(bool) — locks alpha across all layers globally."""
    type: str = "set_global_alpha_lock"
    view_ref: str = "view:active"
    enabled: bool = False


@dataclass
class SetDisablePressure(Action):
    type: str = "set_disable_pressure"
    view_ref: str = "view:active"
    disabled: bool = False


@dataclass
class SetHDRExposure(Action):
    """view.setHDRExposure(float) — HDR exposure for the canvas view."""
    type: str = "set_hdr_exposure"
    view_ref: str = "view:active"
    exposure: float = 0.0


@dataclass
class SetHDRGamma(Action):
    """view.setHDRGamma(float) — HDR gamma correction for the canvas view."""
    type: str = "set_hdr_gamma"
    view_ref: str = "view:active"
    gamma: float = 1.0


@dataclass
class SetCurrentPattern(Action):
    type: str = "set_current_pattern"
    view_ref: str = "view:active"
    pattern_name: str = ""


@dataclass
class SetCurrentGradient(Action):
    type: str = "set_current_gradient"
    view_ref: str = "view:active"
    gradient_name: str = ""


@dataclass
class ActivateResource(Action):
    """view.activateResource(resource) — sets a resource (preset, pattern, gradient,
    palette) as the current resource by name and type.
    resource_type must be a value from RESOURCE_TYPES.
    """
    type: str = "activate_resource"
    view_ref: str = "view:active"
    resource_type: str = "paintoppresets"
    resource_name: str = ""


@dataclass
class SetViewDocument(Action):
    """view.setDocument(document) — switches which document this view displays."""
    type: str = "set_view_document"
    view_ref: str = "view:active"
    doc_ref: str = "doc:main"


@dataclass
class GetViewWindow(Action):
    """view.window() — captures the Krita window that owns this view."""
    type: str = "get_view_window"
    view_ref: str = "view:active"
    ref: str = "window:active"


@dataclass
class GetViewDocument(Action):
    """view.document() — captures the document currently shown in this view."""
    type: str = "get_view_document"
    view_ref: str = "view:active"
    ref: str = "doc:main"


@dataclass
class GetViewCanvas(Action):
    """view.canvas() — captures the canvas backing this view."""
    type: str = "get_view_canvas"
    view_ref: str = "view:active"
    ref: str = "canvas:active"


@dataclass
class ShowFloatingMessage(Action):
    """view.showFloatingMessage(message, icon, timeout, priority).
    priority: 0=High (replaces lower), 1=Medium, 2=Low.
    """
    type: str = "show_floating_message"
    view_ref: str = "view:active"
    message: str = ""
    icon: str = ""          # icon name (e.g. "krita") or "" for no icon
    timeout: int = 2000     # milliseconds
    priority: int = 1       # 0=high, 1=medium, 2=low


# ---------------------------------------------------------------------------
# Window management
# ---------------------------------------------------------------------------

@dataclass
class OpenWindow(Action):
    """Krita.instance().openWindow() — opens a new top-level Krita window."""
    type: str = "open_window"
    ref: str = "window:new"


@dataclass
class ActivateWindow(Action):
    """window.activate() — brings a Krita window to the foreground."""
    type: str = "activate_window"
    window_ref: str = "window:active"


@dataclass
class CloseWindow(Action):
    """window.close() — closes a Krita window (may prompt to save unsaved docs)."""
    type: str = "close_window"
    window_ref: str = "window:active"


@dataclass
class GetWindowActiveView(Action):
    """window.activeView() — captures the active view in a Krita window."""
    type: str = "get_window_active_view"
    window_ref: str = "window:active"
    ref: str = "view:active"


@dataclass
class GetWindowQWindow(Action):
    """window.qwindow() — captures the underlying QMainWindow for a Krita window.
    This is a raw Qt handle, not a libkis-wrapped object; use MethodCall for Qt-only follow-up.
    """
    type: str = "get_window_qwindow"
    window_ref: str = "window:active"
    ref: str = "qwindow:active"


@dataclass
class AddView(Action):
    """window.addView(document) — opens document in an additional tab/view."""
    type: str = "add_view"
    window_ref: str = "window:active"
    doc_ref: str = "doc:main"
    ref: str = "view:new"


@dataclass
class ShowView(Action):
    """window.showView(view) — brings a specific view tab to the foreground."""
    type: str = "show_view"
    window_ref: str = "window:active"
    view_ref: str = "view:active"


@dataclass
class SetViewVisible(Action):
    """view.setVisible() — makes a view tab visible in the window.
    Note: libkis View.setVisible() takes no arguments and only shows the view;
    there is no API to hide a view tab programmatically.
    """
    type: str = "set_view_visible"
    view_ref: str = "view:active"


# ---------------------------------------------------------------------------
# Canvas controls
# ---------------------------------------------------------------------------

@dataclass
class SetCanvasZoom(Action):
    """canvas.setZoomLevel(float). 1.0 = 100%."""
    type: str = "set_canvas_zoom"
    canvas_ref: str = "canvas:active"
    zoom: float = 1.0


@dataclass
class ResetCanvasZoom(Action):
    """canvas.resetZoom() — returns zoom to 100%."""
    type: str = "reset_canvas_zoom"
    canvas_ref: str = "canvas:active"


@dataclass
class SetCanvasRotation(Action):
    """canvas.setRotation(degrees)."""
    type: str = "set_canvas_rotation"
    canvas_ref: str = "canvas:active"
    angle: float = 0.0


@dataclass
class ResetCanvasRotation(Action):
    """canvas.resetRotation() — returns rotation to 0°."""
    type: str = "reset_canvas_rotation"
    canvas_ref: str = "canvas:active"


@dataclass
class PanCanvas(Action):
    """canvas.pan(x, y) — pans the viewport by pixel offset."""
    type: str = "pan_canvas"
    canvas_ref: str = "canvas:active"
    x: int = 0
    y: int = 0


@dataclass
class SetCanvasCenter(Action):
    """canvas.setPreferredCenter(x, y) — centers image pixel at viewport center."""
    type: str = "set_canvas_center"
    canvas_ref: str = "canvas:active"
    x: float = 0.0
    y: float = 0.0


@dataclass
class SetCanvasMirror(Action):
    """canvas.setMirror(bool) — horizontal mirror mode."""
    type: str = "set_canvas_mirror"
    canvas_ref: str = "canvas:active"
    mirror: bool = False


@dataclass
class SetWrapAroundMode(Action):
    """canvas.setWrapAroundMode(bool) — tile-wrap painting (requires OpenGL)."""
    type: str = "set_wrap_around_mode"
    canvas_ref: str = "canvas:active"
    enabled: bool = False


@dataclass
class SetLevelOfDetailMode(Action):
    """canvas.setLevelOfDetailMode(bool) — Instant Preview mode (requires OpenGL)."""
    type: str = "set_level_of_detail_mode"
    canvas_ref: str = "canvas:active"
    enabled: bool = False


# ---------------------------------------------------------------------------
# Stroke primitives  (Node.paint*)
# ---------------------------------------------------------------------------

@dataclass
class PaintLine(Action):
    """node.paintLine(p1, p2, pressure1, pressure2, strokeStyle).
    stroke_style ∈ STROKE_STYLES.
    """
    type: str = "paint_line"
    node_ref: str = "node:active"
    p1: Point = (0.0, 0.0)
    p2: Point = (0.0, 0.0)
    pressure1: float = 1.0
    pressure2: float = 1.0
    stroke_style: str = "ForegroundColor"


@dataclass
class PaintPath(Action):
    """node.paintPath(path, strokeStyle, fillStyle).
    stroke_style ∈ STROKE_STYLES, fill_style ∈ FILL_STYLES.
    """
    type: str = "paint_path"
    node_ref: str = "node:active"
    path: Path = field(default_factory=list)
    stroke_style: str = "ForegroundColor"
    fill_style: str = "None"


@dataclass
class PaintEllipse(Action):
    """node.paintEllipse(rect, strokeStyle, fillStyle).
    fill_style ∈ FILL_STYLES — "Pattern" fills with the current pattern resource.
    """
    type: str = "paint_ellipse"
    node_ref: str = "node:active"
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    stroke_style: str = "ForegroundColor"
    fill_style: str = "None"


@dataclass
class PaintRectangle(Action):
    """fill_style ∈ FILL_STYLES — "Pattern" fills with the current pattern resource."""
    type: str = "paint_rectangle"
    node_ref: str = "node:active"
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    stroke_style: str = "ForegroundColor"
    fill_style: str = "None"


@dataclass
class PaintPolygon(Action):
    """fill_style ∈ FILL_STYLES — "Pattern" fills with the current pattern resource."""
    type: str = "paint_polygon"
    node_ref: str = "node:active"
    points: List[Point] = field(default_factory=list)
    stroke_style: str = "ForegroundColor"
    fill_style: str = "None"


@dataclass
class PaintStrokes(Action):
    """Convenience batch: runner expands into N paint_line calls along `path`
    with per-point pressures. stroke_style ∈ STROKE_STYLES.
    """
    type: str = "paint_strokes"
    node_ref: str = "node:active"
    path: Path = field(default_factory=list)
    pressures: List[float] = field(default_factory=list)
    stroke_style: str = "ForegroundColor"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass
class SelectionRect(Action):
    """selection.select(x, y, w, h, value). value 0..255 = selection strength."""
    type: str = "selection_rect"
    selection_ref: str = "sel:current"
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    value: int = 255


@dataclass
class SelectAll(Action):
    """selection.selectAll(node, value) — selects the full node extent."""
    type: str = "selection_all"
    selection_ref: str = "sel:current"
    node_ref: str = "node:active"
    value: int = 255


@dataclass
class SelectionInvert(Action):
    type: str = "selection_invert"
    selection_ref: str = "sel:current"


@dataclass
class MoveSelection(Action):
    """selection.move(x, y) — repositions the selection mask."""
    type: str = "selection_move"
    selection_ref: str = "sel:current"
    x: int = 0
    y: int = 0


@dataclass
class ResizeSelection(Action):
    """selection.resize(w, h) — resizes the selection mask dimensions."""
    type: str = "selection_resize"
    selection_ref: str = "sel:current"
    w: int = 0
    h: int = 0


@dataclass
class SelectionFeather(Action):
    type: str = "selection_feather"
    selection_ref: str = "sel:current"
    radius: int = 0


@dataclass
class SelectionGrow(Action):
    type: str = "selection_grow"
    selection_ref: str = "sel:current"
    x_radius: int = 0
    y_radius: int = 0


@dataclass
class SelectionShrink(Action):
    type: str = "selection_shrink"
    selection_ref: str = "sel:current"
    x_radius: int = 0
    y_radius: int = 0
    edge_lock: bool = False


@dataclass
class ContractSelection(Action):
    """selection.contract(value) — shrinks selection width/height uniformly."""
    type: str = "selection_contract"
    selection_ref: str = "sel:current"
    value: int = 0


@dataclass
class ErodeSelection(Action):
    """selection.erode() — morphological erosion by 1 pixel."""
    type: str = "selection_erode"
    selection_ref: str = "sel:current"


@dataclass
class DilateSelection(Action):
    """selection.dilate() — morphological dilation by 1 pixel."""
    type: str = "selection_dilate"
    selection_ref: str = "sel:current"


@dataclass
class BorderSelection(Action):
    """selection.border(xRadius, yRadius) — keeps only the border ring of a selection."""
    type: str = "selection_border"
    selection_ref: str = "sel:current"
    x_radius: int = 0
    y_radius: int = 0


@dataclass
class SmoothSelection(Action):
    """selection.smooth() — smooths jagged edges of the selection."""
    type: str = "selection_smooth"
    selection_ref: str = "sel:current"


@dataclass
class SelectionClear(Action):
    """selection.clear() — deselects everything."""
    type: str = "selection_clear"
    selection_ref: str = "sel:current"


@dataclass
class SelectionAdd(Action):
    """selection.add(other) — boolean OR with another selection."""
    type: str = "selection_add"
    selection_ref: str = "sel:current"
    other_ref: str = "sel:other"


@dataclass
class SelectionSubtract(Action):
    """selection.subtract(other) — removes other's pixels from this selection."""
    type: str = "selection_subtract"
    selection_ref: str = "sel:current"
    other_ref: str = "sel:other"


@dataclass
class SelectionIntersect(Action):
    """selection.intersect(other) — boolean AND with another selection."""
    type: str = "selection_intersect"
    selection_ref: str = "sel:current"
    other_ref: str = "sel:other"


@dataclass
class SelectionXor(Action):
    """selection.symmetricdifference(other) — symmetric difference (XOR)."""
    type: str = "selection_xor"
    selection_ref: str = "sel:current"
    other_ref: str = "sel:other"


@dataclass
class ReplaceSelection(Action):
    """selection.replace(other) — replaces this selection's pixels with other's."""
    type: str = "selection_replace"
    selection_ref: str = "sel:current"
    other_ref: str = "sel:other"


@dataclass
class DuplicateSelection(Action):
    """selection.duplicate() — creates an independent copy of this selection."""
    type: str = "selection_duplicate"
    selection_ref: str = "sel:current"
    new_ref: str = "sel:copy"


@dataclass
class SetSelectionPixelData(Action):
    """selection.setPixelData(data, x, y, w, h) — writes raw grayscale mask bytes.
    data is base64-encoded; 255 = fully selected, 0 = fully deselected.
    """
    type: str = "set_selection_pixel_data"
    selection_ref: str = "sel:current"
    data: str = ""      # base64-encoded grayscale bytes
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


@dataclass
class CopyFromNode(Action):
    """selection.copy(node) — copies the selected area of node to clipboard."""
    type: str = "copy_from_node"
    selection_ref: str = "sel:current"
    node_ref: str = "node:active"


@dataclass
class CutFromNode(Action):
    """selection.cut(node) — cuts the selected area of node to clipboard."""
    type: str = "cut_from_node"
    selection_ref: str = "sel:current"
    node_ref: str = "node:active"


@dataclass
class PasteToNode(Action):
    """selection.paste(node, x, y) — pastes clipboard content into node at position."""
    type: str = "paste_to_node"
    selection_ref: str = "sel:current"
    node_ref: str = "node:active"
    x: int = 0
    y: int = 0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

@dataclass
class ApplyFilter(Action):
    """filter_name must be a value from FILTER_NAMES.
    Rect (x, y, w, h) = 0,0,0,0 applies to the full layer extent.
    """
    type: str = "apply_filter"
    node_ref: str = "node:active"
    filter_name: str = "blur"
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StartFilter(Action):
    """filter.startFilter(node, x, y, w, h) — starts filter application asynchronously.
    filter_name must be a value from FILTER_NAMES.
    Use wait_for_done after to sync. Rect 0,0,0,0 = full layer extent.
    """
    type: str = "start_filter"
    node_ref: str = "node:active"
    filter_name: str = "blur"
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Palette operations
# ---------------------------------------------------------------------------

@dataclass
class AddPaletteGroup(Action):
    """palette.addGroup(name) — creates a named group inside a palette resource."""
    type: str = "add_palette_group"
    palette_name: str = ""
    group_name: str = ""


@dataclass
class RemovePaletteGroup(Action):
    """palette.removeGroup(name, keepColors) — removes a group; keepColors moves
    its entries to the default group instead of deleting them.
    """
    type: str = "remove_palette_group"
    palette_name: str = ""
    group_name: str = ""
    keep_colors: bool = True


@dataclass
class RenamePaletteGroup(Action):
    """palette.renameGroup(oldName, newName)."""
    type: str = "rename_palette_group"
    palette_name: str = ""
    old_group_name: str = ""
    new_group_name: str = ""


@dataclass
class MovePaletteGroup(Action):
    """palette.moveGroup(groupName, groupNameInsertBefore) — reorders groups."""
    type: str = "move_palette_group"
    palette_name: str = ""
    group_name: str = ""
    insert_before: str = ""


@dataclass
class AddPaletteEntry(Action):
    """palette.addEntry(swatch, groupName) — appends a color swatch to a palette group.
    color follows the standard Color dict format.
    """
    type: str = "add_palette_entry"
    palette_name: str = ""
    group_name: str = ""
    entry_name: str = ""
    color: Color = field(default_factory=lambda: {"hex": "#000000"})
    spot_color: bool = False


@dataclass
class RemovePaletteEntry(Action):
    """palette.removeEntry(index) — removes swatch at flat index from default group."""
    type: str = "remove_palette_entry"
    palette_name: str = ""
    index: int = 0


@dataclass
class RemovePaletteEntryFromGroup(Action):
    """palette.removeEntryFromGroup(index, groupName) — removes swatch at index
    within the specified group.
    """
    type: str = "remove_palette_entry_from_group"
    palette_name: str = ""
    group_name: str = ""
    index: int = 0


@dataclass
class SetPaletteColumnCount(Action):
    """palette.setColumnCount(columns) — changes the palette grid width."""
    type: str = "set_palette_column_count"
    palette_name: str = ""
    columns: int = 16


@dataclass
class SetPaletteComment(Action):
    """palette.setComment(str) — sets the description/comment metadata on a palette."""
    type: str = "set_palette_comment"
    palette_name: str = ""
    palette_comment: str = ""


@dataclass
class SetPaletteGroupRowCount(Action):
    """palette.setRowCountGroup(rows, name) — sets the row count for a named group."""
    type: str = "set_palette_group_row_count"
    palette_name: str = ""
    group_name: str = ""
    rows: int = 1


@dataclass
class SavePalette(Action):
    """palette.save() — persists palette changes back to disk."""
    type: str = "save_palette"
    palette_name: str = ""


# ---------------------------------------------------------------------------
# Krita root-level operations
# ---------------------------------------------------------------------------

@dataclass
class WriteSetting(Action):
    """Krita.instance().writeSetting(group, name, value) — persists a config key."""
    type: str = "write_setting"
    group: str = ""
    name: str = ""
    value: str = ""


@dataclass
class AddColorProfile(Action):
    """Krita.instance().addProfile(profilePath) — loads an ICC profile from file."""
    type: str = "add_color_profile"
    profile_path: str = ""


# ---------------------------------------------------------------------------
# Object constructors  (create named objects in the runner symbol table)
# ---------------------------------------------------------------------------

@dataclass
class CreateFilter(Action):
    """Krita.instance().filter(name) — instantiates a Filter object and stores it
    under filter_ref. Required before SetFilterName / SetFilterConfiguration /
    assigning to a FilterLayer or FilterMask via MethodCall.
    filter_name must be a value from Krita.filters() (e.g. 'blur', 'sharpen').
    """
    type: str = "create_filter"
    ref: str = "filter:active"
    filter_name: str = "blur"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreateManagedColor(Action):
    """Creates a ManagedColor object and stores it under color_ref.
    Use SetManagedColor* actions afterwards to adjust its values.
    color provides an initial value via the standard Color dict format.
    """
    type: str = "create_managed_color"
    ref: str = "color:active"
    color: Color = field(default_factory=lambda: {"hex": "#000000"})
    color_model: str = "RGBA"
    color_depth: str = "U8"
    profile: str = "sRGB-elle-V2-srgbtrc.icc"


@dataclass
class CreateSelection(Action):
    """Creates an empty Selection object and stores it under selection_ref.
    Required when building a selection from scratch via SetSelectionPixelData
    or boolean ops without starting from SelectionRect / SelectAll.
    """
    type: str = "create_selection"
    ref: str = "sel:new"
    doc_ref: str = "doc:main"


# ---------------------------------------------------------------------------
# ManagedColor  (color object manipulation)
# ---------------------------------------------------------------------------

@dataclass
class SetManagedColorComponents(Action):
    """color.setComponents([float, ...]) — sets raw channel values on a ManagedColor.
    Component order and count depend on the color model (e.g. RGBA = [R, G, B, A]).
    """
    type: str = "set_managed_color_components"
    color_ref: str = "color:active"
    components: List[float] = field(default_factory=list)


@dataclass
class SetManagedColorProfile(Action):
    """color.setColorProfile(str) — assigns an ICC profile to the color object."""
    type: str = "set_managed_color_profile"
    color_ref: str = "color:active"
    profile: str = "sRGB-elle-V2-srgbtrc.icc"


@dataclass
class SetManagedColorSpace(Action):
    """color.setColorSpace(model, depth, profile) — converts the color to a new space."""
    type: str = "set_managed_color_space"
    color_ref: str = "color:active"
    color_model: str = "RGBA"
    color_depth: str = "U8"
    profile: str = "sRGB-elle-V2-srgbtrc.icc"


@dataclass
class SetManagedColorFromXML(Action):
    """color.fromXML(str) — loads a ManagedColor from its XML serialization.
    Obtain valid XML via color.toXML() on an existing color object.
    """
    type: str = "set_managed_color_from_xml"
    color_ref: str = "color:active"
    xml: str = ""


# ---------------------------------------------------------------------------
# Channel  (per-channel visibility and pixel data)
# ---------------------------------------------------------------------------

@dataclass
class SetChannelVisible(Action):
    """channel.setVisible(bool) — shows or hides an individual color channel.
    channel_ref must point to a Channel object stored in the symbol table.
    """
    type: str = "set_channel_visible"
    channel_ref: str = "channel:active"
    visible: bool = True


@dataclass
class SetChannelPixelData(Action):
    """channel.setPixelData(data, rect) — writes raw bytes into one channel plane.
    data is base64-encoded; rect is [x, y, w, h] in image pixels.
    Only meaningful on paintlayer or mask nodes.
    """
    type: str = "set_channel_pixel_data"
    channel_ref: str = "channel:active"
    data: str = ""      # base64-encoded channel bytes
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


# ---------------------------------------------------------------------------
# Filter object configuration
# ---------------------------------------------------------------------------

@dataclass
class SetFilterName(Action):
    """filter.setName(str) — renames a Filter object in the symbol table."""
    type: str = "set_filter_name"
    filter_ref: str = "filter:active"
    name: str = ""


@dataclass
class SetFilterConfiguration(Action):
    """filter.setConfiguration(InfoObject) — replaces the filter's parameter bundle.
    config is a flat key→value dict matching the filter's InfoObject schema.
    """
    type: str = "set_filter_configuration"
    filter_ref: str = "filter:active"
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resource object mutation
# ---------------------------------------------------------------------------

@dataclass
class SetResourceName(Action):
    """resource.setName(str) — renames a resource (preset, pattern, gradient, etc.)."""
    type: str = "set_resource_name"
    resource_ref: str = "resource:active"
    name: str = ""


@dataclass
class SetResourceImage(Action):
    """resource.setImage(QImage) — replaces the resource's thumbnail/display image.
    data is a base64-encoded PNG byte string.
    """
    type: str = "set_resource_image"
    resource_ref: str = "resource:active"
    data: str = ""      # base64-encoded PNG bytes


# ---------------------------------------------------------------------------
# Scratchpad
# ---------------------------------------------------------------------------

@dataclass
class ClearScratchpad(Action):
    """scratchpad.clear() — clears all content from the scratchpad."""
    type: str = "clear_scratchpad"
    scratchpad_ref: str = "scratchpad:active"


@dataclass
class FillScratchpadDefault(Action):
    """scratchpad.fillDefault() — fills with Krita's default checker pattern."""
    type: str = "fill_scratchpad_default"
    scratchpad_ref: str = "scratchpad:active"


@dataclass
class FillScratchpadBackground(Action):
    """scratchpad.fillBackground() — fills with the current background color."""
    type: str = "fill_scratchpad_background"
    scratchpad_ref: str = "scratchpad:active"


@dataclass
class FillScratchpadForeground(Action):
    """scratchpad.fillForeground() — fills with the current foreground color."""
    type: str = "fill_scratchpad_foreground"
    scratchpad_ref: str = "scratchpad:active"


@dataclass
class FillScratchpadTransparent(Action):
    """scratchpad.fillTransparent() — fills the scratchpad with transparency."""
    type: str = "fill_scratchpad_transparent"
    scratchpad_ref: str = "scratchpad:active"


@dataclass
class FillScratchpadDocument(Action):
    """scratchpad.fillDocument(fullContent) — fills from the active document projection.
    full_content=True uses full canvas; False uses only the visible viewport area.
    """
    type: str = "fill_scratchpad_document"
    scratchpad_ref: str = "scratchpad:active"
    full_content: bool = True


@dataclass
class FillScratchpadLayer(Action):
    """scratchpad.fillLayer(fullContent) — fills from the active layer's pixel data.
    full_content=True uses full layer extent; False uses only the visible viewport area.
    """
    type: str = "fill_scratchpad_layer"
    scratchpad_ref: str = "scratchpad:active"
    full_content: bool = True


@dataclass
class FillScratchpadPattern(Action):
    """scratchpad.fillPattern(transform) — tiles the current pattern with an
    optional affine transform. transform is a 6-float list [m11,m12,m21,m22,dx,dy]
    representing a QTransform matrix (identity = [1,0,0,1,0,0]).
    """
    type: str = "fill_scratchpad_pattern"
    scratchpad_ref: str = "scratchpad:active"
    transform: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


@dataclass
class FillScratchpadGradient(Action):
    """scratchpad.fillGradient(start, end, shape, repeat, reverse, dither).
    gradient_shape must be a value from GRADIENT_SHAPES.
    gradient_repeat must be a value from GRADIENT_REPEATS.
    start / end are integer [x, y] pixel coordinates in scratchpad space.
    libkis takes QPoint (integer); runner must truncate floats to int.
    Empty point (0, 0) uses the scratchpad corners as default.
    """
    type: str = "fill_scratchpad_gradient"
    scratchpad_ref: str = "scratchpad:active"
    start: Tuple[int, int] = (0, 0)
    end: Tuple[int, int] = (100, 0)
    gradient_shape: str = "linear"
    gradient_repeat: str = "none"
    reverse: bool = False
    dither: bool = False


@dataclass
class LoadScratchpadImage(Action):
    """scratchpad.loadScratchpadImage(QImage) — loads a raster image into the scratchpad.
    data is a base64-encoded PNG byte string.
    """
    type: str = "load_scratchpad_image"
    scratchpad_ref: str = "scratchpad:active"
    data: str = ""      # base64-encoded PNG bytes


@dataclass
class SetScratchpadFillColor(Action):
    """scratchpad.setFillColor(QColor) — sets the color used by fillDefault."""
    type: str = "set_scratchpad_fill_color"
    scratchpad_ref: str = "scratchpad:active"
    color: Color = field(default_factory=lambda: {"hex": "#ffffff"})


@dataclass
class SetScratchpadMode(Action):
    """scratchpad.setMode(str) — sets painting mode manually (requires setModeManually).
    mode must be a value from SCRATCHPAD_MODES.
    """
    type: str = "set_scratchpad_mode"
    scratchpad_ref: str = "scratchpad:active"
    mode: str = "painting"


@dataclass
class SetScratchpadModeManually(Action):
    """scratchpad.setModeManually(bool) — when True, mode is fixed by set_scratchpad_mode.
    When False, Krita controls the mode based on active tool/modifiers.
    """
    type: str = "set_scratchpad_mode_manually"
    scratchpad_ref: str = "scratchpad:active"
    enabled: bool = True


@dataclass
class SetScratchpadZoomLink(Action):
    """scratchpad.setCanvasZoomLink(bool) — links scratchpad zoom to canvas zoom."""
    type: str = "set_scratchpad_zoom_link"
    scratchpad_ref: str = "scratchpad:active"
    linked: bool = True


@dataclass
class SetScratchpadScale(Action):
    """scratchpad.setScale(float) — manually sets the scratchpad zoom level."""
    type: str = "set_scratchpad_scale"
    scratchpad_ref: str = "scratchpad:active"
    scale: float = 1.0


@dataclass
class ScaleScratchpadToFit(Action):
    """scratchpad.scaleToFit() — fits the scratchpad content to the widget size."""
    type: str = "scale_scratchpad_to_fit"
    scratchpad_ref: str = "scratchpad:active"


@dataclass
class ResetScratchpadScale(Action):
    """scratchpad.scaleReset() — resets scratchpad zoom to 1:1."""
    type: str = "reset_scratchpad_scale"
    scratchpad_ref: str = "scratchpad:active"


@dataclass
class PanScratchpadTo(Action):
    """scratchpad.panTo(x, y) — pans the scratchpad viewport to an absolute position."""
    type: str = "pan_scratchpad_to"
    scratchpad_ref: str = "scratchpad:active"
    x: int = 0
    y: int = 0


@dataclass
class PanScratchpadCenter(Action):
    """scratchpad.panCenter() — centers the scratchpad content in the viewport."""
    type: str = "pan_scratchpad_center"
    scratchpad_ref: str = "scratchpad:active"


# ---------------------------------------------------------------------------
# Guides  (Document.guidesConfig())
# ---------------------------------------------------------------------------

@dataclass
class SetGuidesConfig(Action):
    """Sets all guide properties via document.guidesConfig() setters.
    horizontal: list of Y positions in pixels.
    vertical:   list of X positions in pixels.
    line_type ∈ LINE_TYPES — GuidesConfig uses "dot" (not "dotted").
    """
    type: str = "set_guides_config"
    doc_ref: str = "doc:main"
    horizontal: List[float] = field(default_factory=list)
    vertical: List[float] = field(default_factory=list)
    visible: bool = True
    locked: bool = False
    snap: bool = True
    color: Color = field(default_factory=lambda: {"hex": "#6096c8"})
    line_type: str = "solid"


@dataclass
class LoadGuidesConfigXML(Action):
    """guidesConfig.fromXml(xmlContent) — loads full guides config from Krita XML.
    Obtain valid XML via guidesConfig.toXml() on a configured GuidesConfig object.
    """
    type: str = "load_guides_config_xml"
    doc_ref: str = "doc:main"
    xml: str = ""


@dataclass
class RemoveAllGuides(Action):
    """guidesConfig.removeAllGuides() — clears every guide line."""
    type: str = "remove_all_guides"
    doc_ref: str = "doc:main"


# ---------------------------------------------------------------------------
# Grid  (Document.gridConfig())
# ---------------------------------------------------------------------------

@dataclass
class SetGridConfig(Action):
    """Sets grid properties via document.gridConfig() / document.setGridConfig().
    grid_type ∈ GRID_TYPES.
    line_type_* ∈ LINE_TYPES; line_type_vertical also accepts "none" to hide it.
    spacing_active_* and angle_*_active toggle individual grid axis visibility.
    *_aspect_locked links paired values (offset X/Y, spacing X/Y, angles).
    cell_spacing is for "isometric_legacy"; cell_size is for "isometric".
    """
    type: str = "set_grid_config"
    doc_ref: str = "doc:main"
    visible: bool = False
    snap: bool = False
    grid_type: str = "rectangular"
    spacing_x: int = 32
    spacing_y: int = 32
    spacing_active_horizontal: bool = True
    spacing_active_vertical: bool = True
    spacing_aspect_locked: bool = False
    offset_x: int = 0
    offset_y: int = 0
    offset_aspect_locked: bool = False
    subdivision: int = 2
    line_type_main: str = "solid"
    line_type_subdivision: str = "dotted"
    line_type_vertical: str = "solid"     # "none" disables vertical lines (isometric only)
    color_main: Color = field(default_factory=lambda: {"hex": "#c0c0c0"})
    color_subdivision: Color = field(default_factory=lambda: {"hex": "#e0e0e0"})
    color_vertical: Color = field(default_factory=lambda: {"hex": "#c0c0c0"})
    angle_left: float = 30.0
    angle_right: float = 30.0
    angle_left_active: bool = True
    angle_right_active: bool = True
    angle_aspect_locked: bool = False
    cell_size: int = 32
    cell_spacing: int = 32


@dataclass
class LoadGridConfigXML(Action):
    """gridConfig.fromXml(xmlContent) — loads full grid config from Krita XML.
    Obtain valid XML via gridConfig.toXml() on a configured GridConfig object.
    NOTE: requires runner to access gridConfig via MethodCall (not in Document.sip).
    """
    type: str = "load_grid_config_xml"
    doc_ref: str = "doc:main"
    xml: str = ""


# ---------------------------------------------------------------------------
# Shape  (individual vector shapes inside a VectorLayer)
# Shape objects are stored in the symbol table via MethodCall result_ref,
# e.g. by calling vectorLayer.shapes() or addShapesFromSvg().
# ---------------------------------------------------------------------------

@dataclass
class SetShapeName(Action):
    """shape.setName(str) — renames a shape within a VectorLayer."""
    type: str = "set_shape_name"
    shape_ref: str = "shape:active"
    name: str = ""


@dataclass
class SetShapeZIndex(Action):
    """shape.setZIndex(int) — sets the stacking order of a shape within its layer.
    Higher values appear on top.
    """
    type: str = "set_shape_z_index"
    shape_ref: str = "shape:active"
    z_index: int = 0


@dataclass
class SetShapeSelectable(Action):
    """shape.setSelectable(bool) — controls whether the shape can be selected in the UI."""
    type: str = "set_shape_selectable"
    shape_ref: str = "shape:active"
    selectable: bool = True


@dataclass
class SetShapeGeometryProtected(Action):
    """shape.setGeometryProtected(bool) — locks shape geometry so it cannot be
    transformed or edited interactively.
    """
    type: str = "set_shape_geometry_protected"
    shape_ref: str = "shape:active"
    protected: bool = False


@dataclass
class SetShapeVisible(Action):
    """shape.setVisible(bool) — shows or hides an individual vector shape."""
    type: str = "set_shape_visible"
    shape_ref: str = "shape:active"
    visible: bool = True


@dataclass
class SetShapePosition(Action):
    """shape.setPosition(QPointF) — moves the shape's origin to an absolute position."""
    type: str = "set_shape_position"
    shape_ref: str = "shape:active"
    x: float = 0.0
    y: float = 0.0


@dataclass
class SetShapeTransformation(Action):
    """shape.setTransformation(QTransform) — applies a full 2D affine transform.
    matrix is a 6-float list [m11, m12, m21, m22, dx, dy] (identity=[1,0,0,1,0,0]).
    """
    type: str = "set_shape_transformation"
    shape_ref: str = "shape:active"
    matrix: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


@dataclass
class RemoveShape(Action):
    """shape.remove() — deletes the shape from its parent VectorLayer."""
    type: str = "remove_shape"
    shape_ref: str = "shape:active"


@dataclass
class UpdateShape(Action):
    """shape.update() — schedules a repaint of the shape's bounding area."""
    type: str = "update_shape"
    shape_ref: str = "shape:active"


@dataclass
class UpdateShapeAbsolute(Action):
    """shape.updateAbsolute(QRectF) — repaints a specific absolute bounding rect."""
    type: str = "update_shape_absolute"
    shape_ref: str = "shape:active"
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


@dataclass
class SelectShape(Action):
    """shape.select() — adds the shape to the active selection within its VectorLayer."""
    type: str = "select_shape"
    shape_ref: str = "shape:active"


@dataclass
class DeselectShape(Action):
    """shape.deselect() — removes the shape from the active selection."""
    type: str = "deselect_shape"
    shape_ref: str = "shape:active"


# ---------------------------------------------------------------------------
# Swatch  (individual color entries in a Palette)
# Swatch objects are obtained via MethodCall result_ref from
# palette.entryByIndex() or palette.entryByIndexFromGroup().
# ---------------------------------------------------------------------------

@dataclass
class SetSwatchName(Action):
    """swatch.setName(str) — renames a color swatch entry."""
    type: str = "set_swatch_name"
    swatch_ref: str = "swatch:active"
    name: str = ""


@dataclass
class SetSwatchId(Action):
    """swatch.setId(str) — sets the internal ID string of a swatch entry."""
    type: str = "set_swatch_id"
    swatch_ref: str = "swatch:active"
    swatch_id: str = ""


@dataclass
class SetSwatchColor(Action):
    """swatch.setColor(ManagedColor) — changes the color stored in this swatch.
    color_ref must point to a ManagedColor in the symbol table.
    """
    type: str = "set_swatch_color"
    swatch_ref: str = "swatch:active"
    color_ref: str = "color:active"


@dataclass
class SetSwatchSpotColor(Action):
    """swatch.setSpotColor(bool) — marks a swatch as a spot color (not process color)."""
    type: str = "set_swatch_spot_color"
    swatch_ref: str = "swatch:active"
    spot_color: bool = False


# ---------------------------------------------------------------------------
# InfoObject  (key-value property bag used by filters, fill layers, export)
# InfoObject refs are obtained via MethodCall result_ref from
# filter.configuration() or fillLayer.filterConfig().
# ---------------------------------------------------------------------------

@dataclass
class SetInfoObjectProperties(Action):
    """infoObject.setProperties(dict) — bulk-sets all properties at once.
    Replaces the entire property map with the provided key→value dict.
    """
    type: str = "set_info_object_properties"
    info_ref: str = "infoobj:active"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SetInfoObjectProperty(Action):
    """infoObject.setProperty(key, value) — sets a single property by key.
    value must be a non-None JSON-compatible scalar (str, int, float, bool).
    Note: None is stripped by to_dict(), so it cannot be used to clear a property.
    Use MethodCall to call setProperty with a null QVariant if needed.
    """
    type: str = "set_info_object_property"
    info_ref: str = "infoobj:active"
    key: str = ""
    value: Any = ""


# ---------------------------------------------------------------------------
# Notifier  (enable / disable Krita event signals)
# ---------------------------------------------------------------------------

@dataclass
class SetNotifierActive(Action):
    """notifier.setActive(bool) — enables or disables all Krita event signals.
    Set False to suppress signals during batch operations; restore to True after.
    """
    type: str = "set_notifier_active"
    active: bool = True


# ---------------------------------------------------------------------------
# Generic escape hatch
# ---------------------------------------------------------------------------

@dataclass
class MethodCall(Action):
    """Fallback: call any libkis method on any tracked object.
    The runner looks up `target_ref` in the symbol table and invokes `method`
    with `args` (positional) and `kwargs` (keyword), then optionally stores
    the return value under `result_ref` for later reference.

    target_ref: "krita" | "doc:main" | "node:sky" | "view:active" | etc.
    """
    type: str = "method_call"
    target_ref: str = "krita"
    method: str = ""
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result_ref: Optional[str] = None


# ---------------------------------------------------------------------------
# Plan container
# ---------------------------------------------------------------------------

@dataclass
class ActionPlan:
    """Top-level container the translator emits."""
    title: str
    summary: str
    actions: List[Action] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "title": self.title,
                "summary": self.summary,
                "metadata": self.metadata,
                "actions": [a.to_dict() for a in self.actions],
            },
            indent=indent,
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# ---------------------------------------------------------------------------
# Registry of typed actions  (used by the runner & dryrun deserializer)
# ---------------------------------------------------------------------------

# Uses the field default directly instead of instantiating each class.
TYPED_ACTIONS: Dict[str, type] = {
    cls.__dataclass_fields__["type"].default: cls
    for cls in [
        # Krita root
        WriteSetting, AddColorProfile,
        GetActiveDocument, GetActiveWindow,

        # Document lifecycle
        OpenDocument, CreateDocument, CloneDocument,
        SetActiveDocument, SetBatchmode,
        SetDocumentName, SetDocumentResolution, SetDocumentColorProfile,
        SetDocumentWidth, SetDocumentHeight,
        SetDocumentFileName, SetDocumentOffset, SetDocumentXYRes,
        SetDocumentBatchmode, SetDocumentInfo, SetAutosave,
        LockDocument, UnlockDocument,
        SetBackgroundColor, SetColorSpace,
        ResizeImage, CropImage, RotateImage, ShearImage, ScaleImage,
        FlattenDocument, RefreshProjection, WaitForDone, TryBarrierLock,
        CloseDocument, SetDocumentModified,
        SaveDocument, SaveAs, ExportImage,
        SetDocumentSelection, SetDocumentAnnotation, RemoveDocumentAnnotation,

        # Animation
        SetFramesPerSecond, SetAnimationRange, SetPlaybackRange,
        SetCurrentTime, ImportAnimation, SetAudioTracks, SetAudioLevel,

        # Node tree
        CreateNode, SetActiveNode,
        SetNodeName, SetNodeOpacity, SetNodeBlendingMode,
        SetNodeVisible, SetNodeLocked, SetNodeAlphaLocked,
        SetNodeInheritAlpha, SetNodeColorLabel,
        SetNodeCollapsed, SetNodePinnedToTimeline,
        MoveNode, RotateNode, ScaleNode, CropNode, ShearNode,
        DuplicateNode, MergeDown, RemoveNode,
        EnableNodeAnimation, SetNodeColorSpace, SetNodeColorProfile,
        SetLayerStyle, SetPixelData, SetPixelDataAtTime, SaveNode,
        SetFillLayerGenerator, AddShapesFromSvg,
        AddChildNode, RemoveChildNode, SetChildNodes,

        # Layer-type-specific
        SetGroupPassThrough, SetCloneSource,
        SetFileLayerProperties, ResetFileLayerCache,
        SetFilterLayerFilter, SetFilterMaskFilter,
        SetSelectionMaskSelection, SetTransparencyMaskSelection,
        SetTransformMaskXML,
        SetColorizeMaskSettings, InitColorizeMaskColors,
        SetColorizeMaskKeyStroke, UpdateColorizeMask, ResetColorizeMaskCache,
        SetColorizeMaskEditKeyStrokes, RemoveColorizeMaskKeyStroke,
        SetVectorLayerAntialiased, CreateGroupShape,

        # View / painting context
        SetForegroundColor, SetViewBackgroundColor,
        SetBrushPreset, ModifyPreset,
        SetBrushSize, SetBrushRotation, SetBrushFade, SetPatternSize,
        SetPaintingOpacity, SetPaintingFlow, SetViewBlendingMode,
        SetEraserMode, SetGlobalAlphaLock, SetDisablePressure,
        SetHDRExposure, SetHDRGamma,
        SetCurrentPattern, SetCurrentGradient,
        ActivateResource, SetViewDocument,
        ShowFloatingMessage,
        GetViewWindow, GetViewDocument, GetViewCanvas,

        # Window
        OpenWindow, ActivateWindow, CloseWindow, AddView,
        ShowView, SetViewVisible,
        GetWindowActiveView, GetWindowQWindow,

        # Canvas
        SetCanvasZoom, ResetCanvasZoom,
        SetCanvasRotation, ResetCanvasRotation,
        PanCanvas, SetCanvasCenter,
        SetCanvasMirror, SetWrapAroundMode, SetLevelOfDetailMode,

        # Stroke primitives
        PaintLine, PaintPath, PaintEllipse, PaintRectangle,
        PaintPolygon, PaintStrokes,

        # Selection
        SelectionRect, SelectAll,
        SelectionInvert, MoveSelection, ResizeSelection,
        SelectionFeather, SelectionGrow, SelectionShrink,
        ContractSelection, ErodeSelection, DilateSelection,
        BorderSelection, SmoothSelection, SelectionClear,
        SelectionAdd, SelectionSubtract, SelectionIntersect, SelectionXor,
        ReplaceSelection, DuplicateSelection, SetSelectionPixelData,
        CopyFromNode, CutFromNode, PasteToNode,

        # Filters
        ApplyFilter, StartFilter,

        # Palette
        AddPaletteGroup, RemovePaletteGroup, RenamePaletteGroup,
        MovePaletteGroup, AddPaletteEntry, RemovePaletteEntry,
        RemovePaletteEntryFromGroup, SetPaletteColumnCount,
        SetPaletteComment, SetPaletteGroupRowCount, SavePalette,

        # Object constructors
        CreateFilter, CreateManagedColor, CreateSelection,

        # ManagedColor
        SetManagedColorComponents, SetManagedColorProfile,
        SetManagedColorSpace, SetManagedColorFromXML,

        # Channel
        SetChannelVisible, SetChannelPixelData,

        # Filter object
        SetFilterName, SetFilterConfiguration,

        # Resource
        SetResourceName, SetResourceImage,

        # Scratchpad
        ClearScratchpad, FillScratchpadDefault, FillScratchpadBackground,
        FillScratchpadForeground, FillScratchpadTransparent,
        FillScratchpadDocument, FillScratchpadLayer, FillScratchpadPattern,
        FillScratchpadGradient, LoadScratchpadImage,
        SetScratchpadFillColor, SetScratchpadMode, SetScratchpadModeManually,
        SetScratchpadZoomLink, SetScratchpadScale,
        ScaleScratchpadToFit, ResetScratchpadScale,
        PanScratchpadTo, PanScratchpadCenter,

        # Guides & Grid
        SetGuidesConfig, LoadGuidesConfigXML, RemoveAllGuides,
        SetGridConfig, LoadGridConfigXML,

        # Shape
        SetShapeName, SetShapeZIndex, SetShapeSelectable,
        SetShapeGeometryProtected, SetShapeVisible,
        SetShapePosition, SetShapeTransformation,
        RemoveShape, UpdateShape, UpdateShapeAbsolute,
        SelectShape, DeselectShape,

        # Swatch
        SetSwatchName, SetSwatchId, SetSwatchColor, SetSwatchSpotColor,

        # InfoObject
        SetInfoObjectProperties, SetInfoObjectProperty,

        # Notifier
        SetNotifierActive,

        # Escape hatch
        MethodCall,
    ]
}
