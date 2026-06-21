"""
update_preset_descriptions.py

Reads krita_presets.json and replaces the generic category-level descriptions
with more specific ones derived from each brush's own name.
Run once from the Art_Orch directory:
    python update_preset_descriptions.py
"""

import json
import os

# ── Specific descriptions keyed on substrings found in the brush name ─────────
# Checked in order — first match wins.  Keys are lowercase substrings.

SPECIFIC: list[tuple[str, str]] = [
    # Basic
    ("airbrush soft",         "airbrush — smooth soft gradients, good for glows and shading"),
    ("basic-1",               "basic hard round — simple hard-edged brush"),
    ("basic-2 opacity",       "basic round — pressure controls opacity, good for transparent washes"),
    ("basic-3 flow",          "basic round — pressure controls flow, builds colour gradually"),
    ("basic-4 flow opacity",  "basic round — pressure controls both flow and opacity, very controllable layering"),
    ("basic-5 size opacity",  "basic round — pressure controls both size and opacity"),
    ("basic-5 size",          "basic round — pressure controls size, the most versatile general-purpose brush"),
    ("basic-6 details",       "basic small detail brush — fine marks and hatching"),

    # Pencil
    ("pencil 1 sketch",       "pencil — rough sketchy mypaint pencil"),
    ("pencil 2b",             "pencil — soft 2B mypaint pencil"),
    ("pencil-1 hard",         "pencil — hard precise pencil line"),
    ("pencil-2",              "pencil — standard natural pencil texture"),
    ("pencil-3 large 4b",     "pencil — wide soft 4B pencil, good for large shading areas"),
    ("pencil-4 soft",         "pencil — soft gentle light marks"),
    ("pencil-5 tilted",       "pencil — tilted side-of-pencil shading effect"),
    ("pencil-6 quick shade",  "pencil — quick tonal shading"),

    # Ink
    ("ink pen",               "ink — mypaint ink pen, expressive variable-width line"),
    ("ink-1 precision",       "ink — ultra-precise finest line, best for intricate outlines"),
    ("ink-2 fineliner",       "ink — consistent fine liner, good for clean outlines and details"),
    ("ink-3 gpen",            "ink — calligraphy G-pen, pressure gives thick-to-thin variation"),
    ("ink-4 pen rough",       "ink — rough textured ink stroke, organic expressive lines"),
    ("ink-7 brush rough",     "ink — bristly rough ink brush, loose expressive marks"),
    ("ink-8 sumi-e",          "ink — sumi-e calligraphy brush, flowing East Asian style strokes"),

    # Marker
    ("marker chisel smooth",  "marker — chisel tip, flat angled strokes good for bold fills"),
    ("marker details",        "marker — fine-tip marker for detailed linework"),
    ("marker dry",            "marker — dry streaky marker, uneven coverage"),
    ("marker medium",         "marker — medium opaque mypaint marker"),
    ("marker plain",          "marker — plain opaque mypaint marker"),

    # Bristles
    ("bristles-1 details",      "bristle — fine detail bristle for small textured marks"),
    ("bristles-2 flat rough",   "bristle — flat rough bristle, strong directional texture"),
    ("bristles-3 large smooth", "bristle — large smooth bristle, wide painterly coverage"),
    ("bristles-4 glaze",        "bristle — semi-transparent glaze strokes, good for layering colour"),
    ("bristles-5 flat",         "bristle — flat bristle for broad smooth strokes"),
    ("charcoal rock soft",      "bristle — soft charcoal-rock texture, grainy dark coverage"),
    ("dry roller",              "bristle — dry roller, leaves a repetitive textured pattern"),

    # Dry
    ("dry bristles eroded",   "dry — heavily eroded bristle, very rough broken texture"),
    ("dry brushing",          "dry — classic dry-brush effect, rough scratchy marks"),
    ("dry textured creases",  "dry — dry texture with crease marks, complex surface texture"),
    ("dry bristles",          "dry — dry bristle, rough scratchy coverage"),

    # Chalk / charcoal
    ("chalk details",           "chalk — fine chalk detail marks"),
    ("chalk grainy",            "chalk — grainy chalk with visible texture"),
    ("chalk soft",              "chalk — smooth soft chalk coverage"),
    ("charcoal pencil medium",  "charcoal — medium-width charcoal pencil stroke"),
    ("charcoal pencil thin",    "charcoal — thin precise charcoal line"),
    ("charcoal pencil large",   "charcoal — large area charcoal shading"),

    # Wet
    ("wet bristles rough",    "wet — rough wet bristle, textured fluid stroke"),
    ("wet bristles",          "wet — smooth wet bristle brush"),
    ("wet circle",            "wet — circular wet brush, soft round blendable marks"),
    ("wet knife plus",        "wet — palette knife plus, spreading and mixing strokes"),
    ("wet knife",             "wet — palette knife effect, flat spreading strokes"),
    ("wet paint plus",        "wet — wet paint plus mypaint, very fluid blendable"),
    ("wet paint details",     "wet — fine wet paint brush for detailed wet marks"),
    ("wet paint",             "wet — general wet paint, blendable soft strokes"),
    ("wet smear",             "wet — smearing brush, blends and drags existing colour"),
    ("wet textured soft",     "wet — soft textured wet brush"),

    # Watercolour
    ("waterc basic lines-wet-pattern", "watercolor — wet lines with pattern texture"),
    ("waterc basic lines-wet",         "watercolor — wet fluid lines, colour bleeds"),
    ("waterc basic lines-dry",         "watercolor — dry crisp watercolor lines"),
    ("waterc basic round-fringe 02",   "watercolor — round brush with fringe edge"),
    ("waterc basic round-grain",       "watercolor — round brush with grain texture"),
    ("waterc basic round-grunge",      "watercolor — round brush with grunge texture"),
    ("waterc flat big-grain tilt",     "watercolor — large flat tilted brush, grain texture"),
    ("waterc flat decay tilt",         "watercolor — flat tilted decayed edge brush"),
    ("waterc special blobs",           "watercolor — blob strokes for organic shapes"),
    ("waterc special splats",          "watercolor — splat marks for loose organic texture"),
    ("waterc spread widearea",         "watercolor — wide spreading wash"),
    ("waterc spread-pattern",          "watercolor — spreading strokes with pattern"),
    ("waterc spread",                  "watercolor — spreading wash strokes"),
    ("waterc water-pattern",           "watercolor — water-like pattern wash"),
    ("watercolor fringe",              "watercolor — soft fringe edge"),
    ("watercolor texture",             "watercolor — textured wash"),
    ("waterpaint hard edges",          "watercolor — wet paint with hard dried edges"),
    ("waterpaint soft edges",          "watercolor — wet paint with soft blended edges"),

    # Impasto / RGBA
    ("rgba 01 thick-dry",      "impasto — thick dry textured paint"),
    ("rgba 02 thickpaint",     "impasto — thick opaque paint"),
    ("rgba 03 rake",           "impasto — rake comb texture"),
    ("rgba 04 impasto",        "impasto — classic thick impasto strokes"),
    ("rgba 05 impasto-details","impasto — impasto fine detail marks"),
    ("rgba 06 rock",           "impasto — rock-like rough texture"),
]

# ── Category fallbacks ─────────────────────────────────────────────────────

CATEGORY_FALLBACKS: list[tuple[str, str]] = [
    ("b)", "basic soft round brush — general purpose painting"),
    ("c)", "pencil — light marks, good for underdrawing"),
    ("d)", "ink — crisp lines, good for outlines"),
    ("e)", "marker — flat opaque coverage"),
    ("f)", "bristle — textured directional strokes"),
    ("g)", "dry brush — rough dry texture"),
    ("h)", "chalk or charcoal — grainy dusty marks"),
    ("i)", "wet media — fluid blendable strokes"),
    ("j)", "watercolor — soft wet washes and fluid strokes"),
    ("m)", "impasto — thick textured opaque paint"),
]


def _describe(name: str) -> str:
    n = name.lower()
    for keyword, desc in SPECIFIC:
        if keyword in n:
            return desc
    for prefix, desc in CATEGORY_FALLBACKS:
        if name.startswith(prefix):
            return desc
    return "general purpose brush"


def main() -> None:
    path = os.path.join(os.path.dirname(__file__), "krita_presets.json")
    with open(path, "r", encoding="utf-8") as f:
        presets = json.load(f)

    updated = [{"name": p["name"], "description": _describe(p["name"])} for p in presets]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"Updated {len(updated)} presets in {path}")
    # Show a sample to verify
    for p in updated[:5]:
        print(f"  {p['name']:<38} {p['description']}")


if __name__ == "__main__":
    main()
