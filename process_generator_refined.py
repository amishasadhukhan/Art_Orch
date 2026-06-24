# process_generator_refined.py
#
# Refined pipeline for generating structured painting process descriptions.
#
# PIPELINE
# ─────────────────────────────────────────────────────────────
# Stage 1   — Element Extraction    extract elements + descriptions from query
# Stage 2   — Element Reflection    validate, remove hallucinations
# Stage 3   — Painting Order        bg/mg/fg depth, paint order
# Stage 4   — Order Reflection      validate order logic + layer split rules
# Stage 5.1 — Canvas Layout         categorical bbox placement (zones + fractions)
# Stage 5.2 — Scene Graph           spatial relations + structural gap fill
# Stage 6   — Layout Reflection     validate spatial coherence + zone accuracy
# Stage 7   — Object Realizer       shape decomposition (type, role, placement, color)
# Stage 8   — Object Reflection     validate shapes + merge into painting_order
# Stage 9   — Step Generator        concrete action per element + pixel bbox
# Stage 10  — Step Reflection       validate action specificity + coverage

import json
import os
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─── JSON HELPER ──────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """
    Robustly extract valid JSON from an LLM response.
    Strips <think> blocks (Qwen3), markdown fences, and falls back to
    regex extraction of the outermost {...} block.
    """
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
    raise ValueError(f"Could not extract valid JSON:\n{raw[:500]}")


# ─── LAYOUT VOCABULARY LOADER ─────────────────────────────────────────────────

def _load_layout_vocab() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layout_vocab.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

_LAYOUT_VOCAB = _load_layout_vocab()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Element Extraction
# Input:  raw painting description (string)
# Output: {"scene_type": str, "elements": [{name, description}, ...]}
# ═══════════════════════════════════════════════════════════════════════════════

_S1_SYSTEM = """You are a visual element extractor for digital painting planning.

Given a painting description, extract every distinct visual element as JSON.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly two keys: "scene_type" and "elements".
- "scene_type": a short label for the overall scene (string).
- "elements": an array of objects. Each element must have:
    - "name": short label for this element (string)
    - "description": capture EVERY detail the prompt mentions about this element —
      colors, shapes, motion, texture, style, position, mood.
      Use the prompt's own words as much as possible. Do not summarize or compress.
- List every distinct visual element separately. Do not merge different objects.
- Do not add any detail NOT stated in the prompt.
- Do not assign layers — layer assignment happens in Stage 3.
- DO NOT extract abstract concepts. Reject anything that cannot be painted as a
  visible object or surface: "composition", "style", "mood", "atmosphere",
  "tension", "balance", "theme", "energy", "feeling", "contrast", "dynamic",
  "narrative". If it cannot be drawn, it is not an element.

MULTIPLE INSTANCES:
- If the description mentions more than one instance of the same object type in
  spatially distinct locations, extract each instance as a SEPARATE element with
  a disambiguating name suffix.
  Examples:
    "two chairs — one on the left, one on the right"
      → chair_left, chair_right
    "candles at each corner of the table"
      → candle_left, candle_right  (or candle_1 … candle_4 if four)
    "three windows along the wall"
      → window_1, window_2, window_3
- If the description says "two chairs" without any spatial distinction, extract as
  one element "chairs" and note "two instances" in the description —
  Stage 5.2 will handle placement of the second instance via gap fill.
"""

_S1_EXAMPLES = [
    {
        "input": (
            "The sky is painted in deep amber and burnt orange, with streaks of crimson "
            "near the horizon blending upward into a dusty rose. The sun is a flattened "
            "glowing disc in pale gold, half-submerged behind the mountains, casting a "
            "warm radiant halo. The mountains are rendered in cool purple and slate blue, "
            "three overlapping ridges receding into atmospheric haze. The lake in the "
            "foreground is perfectly still and dark, its surface a near-black mirror."
        ),
        "output": json.dumps({
            "scene_type": "sunset mountain lake",
            "elements": [
                {"name": "sky",       "description": "deep amber and burnt orange with streaks of crimson near the horizon blending upward into a dusty rose"},
                {"name": "sun",       "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo"},
                {"name": "mountains", "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze"},
                {"name": "lake",      "description": "perfectly still and dark, near-black mirror surface reflecting the warm amber sky and the pale glow of the sun"},
            ]
        }, indent=2),
    }
]


def _build_s1_prompt() -> ChatPromptTemplate:
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=_S1_EXAMPLES,
    )
    return ChatPromptTemplate.from_messages([
        ("system", _S1_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Element Reflection
# Input:  description + elements[]
# Output: {"is_valid": bool, "issues": [str], "corrected_elements": [...]}
# ═══════════════════════════════════════════════════════════════════════════════

_S2_SYSTEM = """You are a painting description validator.

You receive an original painting description and a list of extracted visual elements.
Your task: check whether the extracted elements faithfully represent the description.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly three keys: "is_valid", "issues", "corrected_elements".
- "is_valid": true if elements are accurate and complete, false if problems exist (boolean).
- "issues": list of strings describing problems. Empty list [] if none.
- "corrected_elements": the fixed elements array. If is_valid is true, return original unchanged.
- Flag ONLY real problems:
    - Missing elements that the description clearly describes as visible objects
    - Descriptions that contain details NOT present in the original prompt (hallucinations)
    - Descriptions that omit details the prompt explicitly states
    - Abstract concepts that slipped through as elements (remove them)
- For each element, re-read every sentence that mentions it and verify all details are captured.
- Do not change elements that are correctly and completely extracted.
- Do not assign layers — that happens in Stage 3.
"""


def _build_s2_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _S2_SYSTEM),
        ("human", "/no_think\nORIGINAL DESCRIPTION:\n{description}\n\nEXTRACTED ELEMENTS:\n{elements_json}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Painting Order + Layer Assignment
# Input:  scene_type + description + elements[]
# Output: {"scene_type": str, "painting_order": [{order, name, layer, description, position}, ...]}
# ═══════════════════════════════════════════════════════════════════════════════

_S3_SYSTEM = """You are a digital painting sequence planner.

You receive the original painting description and a validated list of visual elements.
Your task: assign each element a layer and a painting order ONLY.
Precise canvas positions are determined later by the Scene Layout Planner — do NOT assign them here.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly two keys: "scene_type" and "painting_order".
- "painting_order": an ordered array. Each object must have exactly four keys:
    - "order": integer starting from 1 (background elements first)
    - "name": same as input
    - "layer": a descriptive Krita layer name (see LAYER RULES below)
    - "description": same as input, unchanged
- Order strictly back to front: ALL background elements before midground, ALL
  midground before foreground.
- Within the same conceptual layer, order by what must be painted first
  (elements painted under others come first).

LAYER ASSIGNMENT GUIDE:
- "background": the scene environment — sky, walls, floor, ground plane, distant landscape,
  ceiling. Anything that forms the "room" or "world" that objects and characters inhabit.
- "midground": objects, furniture, props, vehicles — things placed within the scene that
  are neither the ground/walls nor the main subject.
- "foreground": characters, figures, or elements that are the primary subject closest
  to the viewer.

LAYER NAMING RULES:
- If a conceptual group (background / midground / foreground) has 3 or fewer elements,
  use the group name directly: "background", "midground", "foreground".
- If a conceptual group has more than 3 elements, split them into descriptive named
  sub-layers instead of stacking everything on one layer.
  Use the format "group – descriptor",
  e.g. "foreground – trees", "foreground – character", "background – sky", "background – ground".
- Do not add, remove, or rename any element. Only assign layer and order.
"""


_S3_EXAMPLES = [
    {
        "input": json.dumps({
            "scene_type": "sunset mountain lake",
            "description": "The sky is painted in deep amber and burnt orange. The sun is half-submerged behind the mountains. The mountains are in three overlapping ridges. The lake in the foreground is perfectly still.",
            "elements": [
                {"name": "sky",       "description": "deep amber and burnt orange with streaks of crimson near the horizon blending upward into a dusty rose"},
                {"name": "sun",       "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo"},
                {"name": "mountains", "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze"},
                {"name": "lake",      "description": "perfectly still and dark, near-black mirror surface reflecting the sky and sun"},
            ]
        }, indent=2),
        "output": json.dumps({
            "scene_type": "sunset mountain lake",
            "painting_order": [
                {"order": 1, "name": "sky",       "layer": "background", "description": "deep amber and burnt orange with streaks of crimson near the horizon blending upward into a dusty rose"},
                {"order": 2, "name": "sun",       "layer": "background", "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo"},
                {"order": 3, "name": "mountains", "layer": "midground",  "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze"},
                {"order": 4, "name": "lake",      "layer": "foreground", "description": "perfectly still and dark, near-black mirror surface reflecting the sky and sun"},
            ]
        }, indent=2),
    }
]


def _build_s3_prompt() -> ChatPromptTemplate:
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=_S3_EXAMPLES,
    )
    return ChatPromptTemplate.from_messages([
        ("system", _S3_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Order Reflection
# Input:  description + painting_order[]
# Output: {"is_valid": bool, "issues": [str], "corrected_order": [...]}
# ═══════════════════════════════════════════════════════════════════════════════

_S4_SYSTEM = """You are a painting order validator.

You receive the original painting description and a JSON painting order list.
Your task: check whether the painting order, layers, and positions are correct.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly three keys: "is_valid", "issues", "corrected_order".
- "is_valid": true if order and positions are correct, false otherwise (boolean).
- "issues": list of strings describing problems. Empty list [] if none.
- "corrected_order": the fixed painting_order array. If is_valid is true, return original unchanged.
- Flag these problems:
    - Any background element ordered after a midground or foreground element
    - Any midground element ordered after a foreground element
    - An element that visually sits beneath another but is assigned a higher order number
      (it would be painted AFTER the element that should cover it — wrong)
    - A conceptual group with >3 elements all on the same single layer name
      (they should be split into descriptive sub-layers)
    - A layer assignment that contradicts the element's visual role
      (e.g. a character assigned to "background", a wall assigned to "foreground")
- Do NOT flag or comment on canvas positions — positions are assigned by Stage 5.
- Do NOT flag stylistic choices — only flag structural order and layer errors.
"""


def _build_s4_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _S4_SYSTEM),
        ("human", "/no_think\nORIGINAL DESCRIPTION:\n{description}\n\nPAINTING ORDER:\n{order_json}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5.1 — Canvas Layout (Categorical BBox Placement)
# Input:  description + scene_type + elements_to_place[] + (optional) existing_layout[]
# Output: {"layout": [{name, zones, h_align, v_align, w_fraction, h_fraction}, ...]}
#
# Uses categorical labels — Python converts to pixels deterministically.
# Called twice: Pass 1 for known elements, Pass 2 for gap-fill elements.
# ═══════════════════════════════════════════════════════════════════════════════

_S51_SYSTEM = """You are a canvas layout planner for digital painting.

You receive:
  - "description": the original painting query — authoritative source for all placement decisions
  - "scene_type": overall scene label
  - "elements_to_place": list of elements that need canvas positions assigned
  - "existing_layout": elements that have already been assigned canvas positions.
    If empty, place elements freely using the description and scene knowledge.
    If not empty, place the new elements relative to what is already positioned —
    their zones and sizes should make spatial sense alongside the existing layout.

Your task: assign each element in "elements_to_place" a position on the canvas using
categorical labels only. Do NOT output raw pixel values or percentages — use the label vocabulary below.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly one key: "layout".
- "layout": an array with one entry per element, each having these keys:
    - "name": element name (string)
    - "zones": list of 3×3 grid cells this element occupies (array of strings)
    - "w_fraction": width as fraction of zone width   — "full" | "3/4" | "1/2" | "1/3" | "1/4"
    - "h_fraction": height as fraction of zone height — "full" | "3/4" | "1/2" | "1/3" | "1/4"
    - "h_align": ONLY include when w_fraction is NOT "full" — "left" | "center" | "right"
    - "v_align": ONLY include when h_fraction is NOT "full" — "top"  | "middle" | "bottom"
  Do NOT emit h_align when w_fraction="full" — it has no effect.
  Do NOT emit v_align when h_fraction="full" — it has no effect.

PLACEMENT RULES:
- Elements painted first (lower order) sit visually behind later ones — do not overlap
  unless the description explicitly describes layering.
- Use the original painting description to anchor placements that are explicitly stated.
- For everything else, apply real-world spatial knowledge for the scene type.
- The original painting description is the authoritative source — it overrides assumptions.
"""

_S51_EXAMPLE_KNOWN = {
    "input": json.dumps({
        "description": "The sky fills the upper half with deep amber light. The mountains span the full width at the horizon. The lake covers the lower third, perfectly still.",
        "scene_type": "sunset mountain lake",
        "existing_layout": [],
        "elements_to_place": [
            {"order": 1, "name": "sky",       "layer": "background", "description": "deep amber and burnt orange, streaks of crimson near the horizon blending upward into dusty rose — fills the upper half of the canvas"},
            {"order": 2, "name": "sun",       "layer": "background", "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo"},
            {"order": 3, "name": "mountains", "layer": "midground",  "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze — spans full width at the horizon"},
            {"order": 4, "name": "lake",      "layer": "foreground", "description": "perfectly still and dark, near-black mirror surface reflecting sky and sun — covers the lower third"},
        ]
    }, indent=2),
    "output": json.dumps({
        "layout": [
            {
                "name": "sky",
                "zones": ["upper-left","upper-center","upper-right","mid-left","mid-center","mid-right"],
                "w_fraction": "full", "h_fraction": "full"
            },
            {
                "name": "sun",
                "zones": ["mid-center"],
                "w_fraction": "1/4", "h_fraction": "1/3",
                "h_align": "center", "v_align": "top"
            },
            {
                "name": "mountains",
                "zones": ["mid-left","mid-center","mid-right"],
                "w_fraction": "full", "h_fraction": "1/2",
                "v_align": "bottom"
            },
            {
                "name": "lake",
                "zones": ["lower-left","lower-center","lower-right"],
                "w_fraction": "full", "h_fraction": "full"
            },
        ]
    }, indent=2),
}


def _build_s51_prompt() -> ChatPromptTemplate:
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=[_S51_EXAMPLE_KNOWN],
    )
    return ChatPromptTemplate.from_messages([
        ("system", _LAYOUT_VOCAB + "\n\n" + _S51_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5.2 — Scene Graph (Relations + Structural Gap Fill)
# Input:  description + scene_type + painting_order + layout from 5.1 Pass 1
# Output: {"nodes": [...], "edges": [...]}
#
# nodes: all known elements (status:"existing") + structurally missing ones (status:"added")
# edges: typed relation pairs — supports / sits-on / overlaps / in-front-of /
#        behind / adjacent-to / attached-to / mounted-on
# Added nodes include description + layer for use in Stage 5.1 Pass 2.
# ═══════════════════════════════════════════════════════════════════════════════

_S52_SYSTEM = """You are a scene graph builder for digital painting.

You receive:
  - "description": the original painting query — use it to ground all decisions
  - "scene_type": overall scene label
  - "painting_order": the full validated element list from the previous stage, each entry
    containing order, name, layer, description, and zones already assigned on canvas.
    Use the element descriptions to understand what each element IS and infer how it
    relates to others. Use zones to understand spatial proximity.

Your task: produce a scene graph — typed relation edges between elements, plus any
structurally missing elements implied by those relations.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly two keys: "nodes" and "edges".
- "nodes": array of all elements — existing ones plus any you add as gaps.
  Each node must have:
    - "name": element name (string)
    - "status": "existing" | "added"
    - If status is "added", also include:
        - "description": brief physical description appropriate for this scene
        - "layer": "background" | "midground" | "foreground"
        - "inferred_from": the existing element name that implies this one cannot be absent
        - "insert_after": the ORDER NUMBER of the existing element after which this should
          be drawn — must be less than the order of the element it supports or sits under
          (e.g. a shelf that holds books must be drawn before the books, so insert_after
          should be the order of the last element painted before books)
- "edges": array of directed relation pairs. Each edge must have:
    - "from": element name (string)
    - "rel": relation type — one of: supports | sits-on | overlaps | in-front-of |
              behind | adjacent-to | attached-to | mounted-on | rests-on | leans-against
    - "to": element name (string)

GAP FILL RULES:
- Only add an element if it is physically plausible or a natural structural support physically implied by an
  existing element — not merely probable or decorative.
- Check by FUNCTION not by name: a "workbench" already covers the role of a "desk".
  Do not add an element whose structural role is already served by something in
  painting_order under a different name.
- If an element already exists but the scene implies MORE than one instance
  (e.g. two seated figures → two chairs needed), adding a second is valid.
- The original painting description is authoritative — if it explicitly excludes an
  element or describes a context where it has no place (e.g. figures floating in void
  → no floor), do not add it regardless of structural logic.
- Each added node must carry an "inferred_from" field naming the existing element
  that makes it structurally necessary.
"""


_S52_EXAMPLE = {
    # Input = Stage 4 fields (order, name, layer, description)
    #       + Stage 5.1 fields (zones, w_fraction, h_fraction, h_align?, v_align?)
    #         h_align omitted when w_fraction="full"; v_align omitted when h_fraction="full"
    "input": json.dumps({
        "description": "A person reading alone in a cozy library with stacks of books and warm wooden walls.",
        "scene_type": "cozy library reading room",
        "painting_order": [
            {"order": 1, "name": "walls",  "layer": "background",
             "description": "warm amber wood-paneled walls with a deep golden tone spanning full height across the back of the scene",
             "zones": ["upper-left","upper-center","upper-right","mid-left","mid-center","mid-right"],
             "w_fraction": "full", "h_fraction": "full"},
            {"order": 2, "name": "floor",  "layer": "background",
             "description": "dark hardwood flooring with visible grain lines running horizontally",
             "zones": ["lower-left","lower-center","lower-right"],
             "w_fraction": "full", "h_fraction": "full"},
            {"order": 3, "name": "books",  "layer": "midground",
             "description": "stacks of thick hardcover books in muted reds, greens, and gold spines clustered on the left side at mid-height",
             "zones": ["mid-left"],
             "w_fraction": "3/4", "h_fraction": "1/2", "h_align": "center", "v_align": "top"},
            {"order": 4, "name": "person", "layer": "foreground",
             "description": "a figure leaning forward, face hidden behind an open book, softly lit from above, seated on the right side at mid-height",
             "zones": ["mid-right"],
             "w_fraction": "1/2", "h_fraction": "3/4", "h_align": "center", "v_align": "middle"},
        ]
    }, indent=2),
    "output": json.dumps({
        "nodes": [
            {"name": "walls",     "status": "existing"},
            {"name": "floor",     "status": "existing"},
            {"name": "books",     "status": "existing"},
            {"name": "person",    "status": "existing"},
            {"name": "bookshelf", "status": "added", "layer": "midground",
             "description": "tall wooden shelving unit with horizontal planks matching the amber wall tone, supporting the stacked books",
             "inferred_from": "books", "insert_after": 2},
            {"name": "armchair",  "status": "added", "layer": "midground",
             "description": "cushioned reading chair in warm muted fabric, low to the ground, supporting the seated figure",
             "inferred_from": "person", "insert_after": 3},
        ],
        "edges": [
            {"from": "floor",     "rel": "supports",    "to": "bookshelf"},
            {"from": "floor",     "rel": "supports",    "to": "armchair"},
            {"from": "walls",     "rel": "behind",      "to": "bookshelf"},
            {"from": "walls",     "rel": "behind",      "to": "armchair"},
            {"from": "bookshelf", "rel": "supports",    "to": "books"},
            {"from": "armchair",  "rel": "supports",    "to": "person"},
            {"from": "person",    "rel": "sits-on",     "to": "armchair"},
            {"from": "books",     "rel": "rests-on",    "to": "bookshelf"},
        ]
    }, indent=2),
}


def _build_s52_prompt() -> ChatPromptTemplate:
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=[_S52_EXAMPLE],
    )
    return ChatPromptTemplate.from_messages([
        ("system", _LAYOUT_VOCAB + "\n\n" + _S52_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Layout Reflection
# Input:  description + scene_type + painting_order (unified Stage 5 output) + edges
# Output: corrected painting_order (same structure — unchanged if no issues found)
#
# Checks:
#   1. Spatial coherence — elements linked by supports/rests-on/sits-on/attached-to
#      must share a zone or be in directly adjacent zones
#   2. Zone vs description — explicit position words in element descriptions
#      (left/right/upper/lower/center) must agree with assigned zones
#   3. Layer ordering — background order < midground order < foreground order
#   4. Size plausibility — fill elements (spanning 3+ zones) should use w/h_fraction=full;
#      small detail elements should not span the full canvas
# ═══════════════════════════════════════════════════════════════════════════════

_S6_SYSTEM = """You are a layout reviewer for digital painting pipelines.

You receive:
  - "description": the original painting query — authoritative for all placement decisions
  - "scene_type": overall scene label
  - "painting_order": the unified element list from Stage 5, each entry having:
      order, name, layer, description, zones, w_fraction, h_fraction,
      and optionally h_align (when w_fraction != full) / v_align (when h_fraction != full)
  - "edges": directed spatial relation pairs from the scene graph

Your task: review the layout for errors and return the corrected painting_order.

CHECKS TO PERFORM:
1. SPATIAL COHERENCE — for every edge where rel is one of:
   supports / rests-on / sits-on / attached-to / mounted-on
   the two elements must share at least one zone OR be in directly adjacent zones
   (e.g. mid-left and lower-left are adjacent; mid-left and mid-right are NOT adjacent).
   If violated, move the structurally dependent element to match its parent's zones.

2. ZONE VS DESCRIPTION — if an element's description contains an explicit position word
   (left / right / upper / lower / center / top / bottom), its zones must agree.
   e.g. description says "clustered on the left" → zones must include a left-column zone.
   If violated, reassign zones to match the description.

3. LAYER ORDERING — all background elements must have lower order numbers than all
   midground elements; all midground lower than foreground.
   If violated, renumber to fix the ordering while preserving relative order within
   each layer group.

4. SIZE PLAUSIBILITY — an element whose zones span an entire row or the full canvas
   should use w_fraction="full" and/or h_fraction="full".
   A single-zone small element (person, sun, chair) should NOT use w_fraction="full"
   unless its description says it fills that zone.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly one key: "painting_order".
- Return the FULL painting_order — every element, corrected or not.
- Preserve all fields. Only change the specific field that has an error.
- Do NOT add or remove elements.
- h_align only when w_fraction != full; v_align only when h_fraction != full.
"""



def _build_s6_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _LAYOUT_VOCAB + "\n\n" + _S6_SYSTEM),
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Object Realizer (Shape Decomposition)
# Input:  painting_order (unified Stage 6 output — all elements with zones/fractions)
# Output: {"objects": [{name, shapes: [...]}, ...]}
#
# Decomposes each element into the minimal set of geometric primitives needed to
# construct its visual form within its bounding box.
#
# AVAILABLE PRIMITIVES (no coordinates — type + role + color only):
#   rect_filled, rect_outline, ellipse_filled, ellipse_outline, polygon, stroke
# ═══════════════════════════════════════════════════════════════════════════════

_S7_SYSTEM = """You are an object shape planner for digital painting.

You receive a list of scene elements. Your task: for each element, output the
minimal set of geometric primitives that together construct its visual form.
Do NOT output any coordinates — only shape type, role, and color.

AVAILABLE PRIMITIVES:

  rect_filled    : {{"type":"rect_filled",     "role":str, "placement":str, "color":str}}
  rect_outline   : {{"type":"rect_outline",    "role":str, "placement":str, "color":str}}
  ellipse_filled : {{"type":"ellipse_filled",  "role":str, "placement":str, "color":str}}
  ellipse_outline: {{"type":"ellipse_outline", "role":str, "placement":str, "color":str}}
  polygon        : {{"type":"polygon",         "role":str, "placement":str, "color":str}}
  stroke         : {{"type":"stroke",          "role":str, "placement":str, "color":str, "thickness":"thin"|"medium"|"thick"}}

  role      = short label for what part of the element this shape represents
  placement = natural language description of where this shape sits within the element's
              bounding box — e.g. "full width, top 15% of element height",
              "left edge narrow strip, full height below tabletop",
              "centered, leaning-forward upper body filling most of the element area, etc..."
  color     = short descriptive string drawn directly from the element's description

RULES:
- Keep shapes minimal — 1 to 5 per element unless the description explicitly names more parts.
- Color must come from the element's own description — do not invent colors.
- Add an outline variant (rect_outline / ellipse_outline) only if the description
  mentions a visible border, rim, or glow on that element.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly one key: "objects".
- "objects": array with one entry per element, in the same order as the input.
  Each entry: {{"name": str, "shapes": [shape, ...]}}.
- Do NOT skip any element. Every element must appear in "objects".
"""

_S7_EXAMPLE = {
    "input": json.dumps({
        "description": "A person reading alone in a cozy library with stacks of books and warm wooden walls.",
        "painting_order": [
            {"order": 1, "name": "walls",     "layer": "background",
             "description": "warm amber wood-paneled walls with a deep golden tone spanning full height",
             "zones": ["upper-left","upper-center","upper-right","mid-left","mid-center","mid-right"],
             "w_fraction": "full", "h_fraction": "full"},
            {"order": 2, "name": "floor",     "layer": "background",
             "description": "dark hardwood flooring with visible grain lines running horizontally",
             "zones": ["lower-left","lower-center","lower-right"],
             "w_fraction": "full", "h_fraction": "full"},
            {"order": 3, "name": "bookshelf", "layer": "midground",
             "description": "tall wooden shelving unit with horizontal planks matching the amber wall tone",
             "zones": ["mid-left"], "w_fraction": "3/4", "h_fraction": "full"},
            {"order": 4, "name": "books",     "layer": "midground",
             "description": "stacks of thick hardcover books in muted reds, greens, and gold spines",
             "zones": ["mid-left"], "w_fraction": "3/4", "h_fraction": "1/2", "v_align": "top"},
            {"order": 5, "name": "armchair",  "layer": "midground",
             "description": "cushioned reading chair in warm muted fabric, low seat with padded back and arms",
             "zones": ["mid-right"], "w_fraction": "1/2", "h_fraction": "3/4",
             "h_align": "center", "v_align": "middle"},
            {"order": 6, "name": "person",    "layer": "foreground",
             "description": "a figure leaning forward, face hidden behind an open book, softly lit from above",
             "zones": ["mid-right"], "w_fraction": "1/2", "h_fraction": "3/4",
             "h_align": "center", "v_align": "middle"},
        ]
    }, indent=2),
    "output": json.dumps({
        "objects": [
            {
                "name": "walls",
                "shapes": [
                    {"type": "rect_filled", "role": "wall_fill",
                     "placement": "covers full element area edge to edge",
                     "color": "warm amber golden tone"}
                ]
            },
            {
                "name": "floor",
                "shapes": [
                    {"type": "rect_filled", "role": "floor_fill",
                     "placement": "covers full element area edge to edge",
                     "color": "dark hardwood brown"}
                ]
            },
            {
                "name": "bookshelf",
                "shapes": [
                    {"type": "rect_filled", "role": "back_panel",
                     "placement": "full width, full height — the flat rear wall of the shelf unit",
                     "color": "deep amber wood tone"},
                    {"type": "rect_filled", "role": "shelf_top",
                     "placement": "full width, thin horizontal strip at the top third of element",
                     "color": "amber wood, slightly lighter than back panel"},
                    {"type": "rect_filled", "role": "shelf_mid",
                     "placement": "full width, thin horizontal strip at the middle of element",
                     "color": "amber wood, slightly lighter than back panel"},
                    {"type": "rect_filled", "role": "shelf_low",
                     "placement": "full width, thin horizontal strip at the lower third of element",
                     "color": "amber wood, slightly lighter than back panel"},
                ]
            },
            {
                "name": "books",
                "shapes": [
                    {"type": "rect_filled", "role": "book_red",
                     "placement": "left portion of element, tall narrow upright spine",
                     "color": "muted red spine"},
                    {"type": "rect_filled", "role": "book_green",
                     "placement": "center portion, tall narrow upright spine adjacent to red book",
                     "color": "muted green spine"},
                    {"type": "rect_filled", "role": "book_gold",
                     "placement": "right portion, tall narrow upright spine adjacent to green book",
                     "color": "gold spine"},
                ]
            },
            {
                "name": "armchair",
                "shapes": [
                    {"type": "rect_filled", "role": "seat",
                     "placement": "full width, bottom 30% of element — low wide cushioned seat",
                     "color": "warm muted fabric"},
                    {"type": "rect_filled", "role": "back",
                     "placement": "full width, upper 50% of element — tall padded back rest",
                     "color": "warm muted fabric, slightly darker"},
                    {"type": "rect_filled", "role": "left_arm",
                     "placement": "left edge, narrow vertical strip spanning seat and back height",
                     "color": "warm muted fabric"},
                    {"type": "rect_filled", "role": "right_arm",
                     "placement": "right edge, narrow vertical strip spanning seat and back height",
                     "color": "warm muted fabric"},
                ]
            },
            {
                "name": "person",
                "shapes": [
                    {"type": "polygon", "role": "silhouette",
                     "placement": "centered in element — hunched forward posture, upper body dominant in top two thirds; head is tilted down, shoulders catch soft top-down light; knees and lower legs visible in the bottom third",
                     "color": "dark warm shadow with a subtle highlight along the top of shoulders and head"}
                ]
            }
        ]
    }, indent=2),
}


def _build_s7_prompt() -> ChatPromptTemplate:
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=[_S7_EXAMPLE],
    )
    return ChatPromptTemplate.from_messages([
        ("system", _S7_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — Object Reflection + Merge
# Input:  painting_order (with description per element) + objects (Stage 7 output)
# Output: {"objects": [...]} — corrected shapes (Python then merges into painting_order)
#
# This stage both validates Stage 7's output AND acts as the merge step.
# After Stage 8, each element in painting_order carries a validated "shapes" list.
# Stage 9 (Step Generator) receives the fully merged painting_order.
#
# Checks:
#   1. Shape type fit     — primitive type matches the element's visual nature
#   2. Color grounding    — every color traceable to the element's description
#   3. Placement coherence— placement description makes physical sense
#   4. Completeness       — every element from painting_order has shapes defined
#   5. Shape count        — simple elements should not have excessive shapes
# ═══════════════════════════════════════════════════════════════════════════════

_S8_SYSTEM = """You are a shape decomposition reviewer for digital painting.

You receive:
  - "painting_order": the full element list, each with its description
  - "objects": the Stage 7 shape decomposition — one entry per element with a shapes list

Your task: review each element's shapes and correct any errors. Return the full corrected objects array.

CHECKS TO PERFORM:

1. SHAPE TYPE FIT — does the primitive type match the element's visual nature?
   - If a shape uses the wrong primitive type, replace it with the correct one.

2. COLOR GROUNDING — every "color" field must be traceable to the element's description.
   If a color is invented or contradicts the description, replace it with a color
   that IS stated or clearly implied by the description.

3. PLACEMENT COHERENCE — does the placement description make physical sense?
   - A "tabletop" placement must describe the top portion of the element.
   - A "legs" placement must describe the bottom portion.
   - A "back panel" for a shelf must describe a full-height area behind the shelves.
   - If the placement is physically wrong or vague, rewrite it to be accurate.

4. COMPLETENESS — every element in painting_order must have a corresponding entry
   in objects. If an element is missing, add it with appropriate shapes.

5. SHAPE COUNT — A structural object should not need more than 5. If there are too many
   shapes for a simple element, consolidate.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly one key: "objects".
- Return the FULL objects array — every element, corrected or not.
- Preserve all fields (role, placement, color, thickness). Only change fields that have errors.
- Do NOT add or remove elements from the array.
"""


def _build_s8_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _S8_SYSTEM),
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — Step Generator
# Input:  description + painting_order (with shapes from Stage 8) + canvas size
# Output: process.json — ALL Stage 1-8 fields preserved per element,
#         plus two new fields added by Stage 9:
#           "action": concrete painting instruction for this element
#           "reason": why this element is painted at this point in the sequence
#         Python also computes and adds "bbox" from zones + fractions + canvas size.
#
# The LLM ONLY generates {element, action, reason} — nothing else is rewritten.
# Python merges action/reason into the existing painting_order and writes process.json.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 9A: Zone → pixel bbox resolver (pure Python) ─────────────────────────────

_ZONE_GRID = {
    "upper-left":  (0, 0), "upper-center": (1, 0), "upper-right": (2, 0),
    "mid-left":    (0, 1), "mid-center":   (1, 1), "mid-right":   (2, 1),
    "lower-left":  (0, 2), "lower-center": (1, 2), "lower-right": (2, 2),
}
_FRACTION_MAP = {"full": 1.0, "3/4": 0.75, "1/2": 0.5, "1/3": 1/3, "1/4": 0.25}
_H_ALIGN_MAP  = {"left": 0.0, "center": 0.5, "right": 1.0}
_V_ALIGN_MAP  = {"top":  0.0, "middle": 0.5, "bottom": 1.0}


def _resolve_element_bbox(elem: dict, canvas_w: int, canvas_h: int) -> dict:
    """Convert zones + fractions + alignment into a pixel bbox {x, y, w, h}."""
    zones = elem.get("zones", [])
    col_w = canvas_w / 3
    row_h = canvas_h / 3
    cols  = [_ZONE_GRID[z][0] for z in zones if z in _ZONE_GRID]
    rows  = [_ZONE_GRID[z][1] for z in zones if z in _ZONE_GRID]
    if not cols:
        return {"x": 0, "y": 0, "w": canvas_w, "h": canvas_h}

    zone_x = min(cols) * col_w
    zone_y = min(rows) * row_h
    zone_w = (max(cols) - min(cols) + 1) * col_w
    zone_h = (max(rows) - min(rows) + 1) * row_h

    wf     = _FRACTION_MAP.get(elem.get("w_fraction", "full"), 1.0)
    hf     = _FRACTION_MAP.get(elem.get("h_fraction", "full"), 1.0)
    elem_w = zone_w * wf
    elem_h = zone_h * hf

    ha = _H_ALIGN_MAP.get(elem.get("h_align", "center"), 0.5)
    va = _V_ALIGN_MAP.get(elem.get("v_align",  "top"),    0.0)

    return {
        "x": round(zone_x + (zone_w - elem_w) * ha),
        "y": round(zone_y + (zone_h - elem_h) * va),
        "w": round(elem_w),
        "h": round(elem_h),
    }


# ── 9B: LLM action generator ─────────────────────────────────────────────────

_S9_SYSTEM = """You are a painting step writer for a digital painting pipeline.

You receive a list of elements in painting order. Each element has:
  - name, layer, description: what this element is and looks like
  - zones: where on the canvas it sits
  - bbox: its pixel bounding box {{x, y, w, h}} already computed for you
  - shapes: [{{type, role, placement, color}}] — the geometric primitives

Your task: for each element write ONE action + reason pair.

  "action": concrete instruction describing how to paint this element —
            reference its shapes (type, role, placement, color), its zone location,
            and its bbox dimensions to anchor the action spatially.
            Be specific: describe brush motion, shape order, and color.
  "reason": one sentence — why this element is painted at this point in the sequence.

RULES:
- One entry per element, same order as the input.
- For simple fills (one rect_filled): one clear sweep instruction.
- For structural objects (multiple rect shapes): describe the draw order of parts.
- For polygons (silhouettes, ridges): describe the outline shape to trace.
- Reference bbox numbers to give spatial grounding.
- Use color terms from the element description — no hex codes.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly one key: "actions".
- "actions": array with one entry per element: {{"element": str, "action": str, "reason": str}}.
- Do NOT skip any element.
"""

_S9_EXAMPLE = {
    "input": json.dumps({
        "description": "A person reading alone in a cozy library with stacks of books and warm wooden walls.",
        "canvas_w": 900, "canvas_h": 600,
        "painting_order": [
            {"order": 1, "name": "walls", "layer": "background",
             "description": "warm amber wood-paneled walls spanning full height",
             "zones": ["upper-left","upper-center","upper-right","mid-left","mid-center","mid-right"],
             "bbox": {"x": 0, "y": 0, "w": 900, "h": 400},
             "shapes": [
                 {"type": "rect_filled", "role": "wall_fill",
                  "placement": "covers full element area edge to edge",
                  "color": "warm amber golden tone"}
             ]},
            {"order": 2, "name": "floor", "layer": "background",
             "description": "dark hardwood flooring with visible grain lines",
             "zones": ["lower-left","lower-center","lower-right"],
             "bbox": {"x": 0, "y": 400, "w": 900, "h": 200},
             "shapes": [
                 {"type": "rect_filled", "role": "floor_fill",
                  "placement": "covers full element area edge to edge",
                  "color": "dark hardwood brown"}
             ]},
            {"order": 3, "name": "bookshelf", "layer": "midground",
             "description": "tall wooden shelving unit with horizontal planks",
             "zones": ["mid-left"], "bbox": {"x": 0, "y": 200, "w": 225, "h": 200},
             "shapes": [
                 {"type": "rect_filled", "role": "back_panel",
                  "placement": "full width and height", "color": "deep amber wood tone"},
                 {"type": "rect_filled", "role": "shelf_top",
                  "placement": "thin horizontal strip at top third", "color": "amber, slightly lighter"},
                 {"type": "rect_filled", "role": "shelf_mid",
                  "placement": "thin horizontal strip at middle", "color": "amber, slightly lighter"},
                 {"type": "rect_filled", "role": "shelf_low",
                  "placement": "thin horizontal strip at lower third", "color": "amber, slightly lighter"},
             ]},
            {"order": 4, "name": "books", "layer": "midground",
             "description": "stacks of thick hardcover books in muted reds, greens, and gold spines",
             "zones": ["mid-left"], "bbox": {"x": 0, "y": 200, "w": 225, "h": 100},
             "shapes": [
                 {"type": "rect_filled", "role": "book_red",
                  "placement": "left portion, tall narrow upright spine", "color": "muted red spine"},
                 {"type": "rect_filled", "role": "book_green",
                  "placement": "centre portion, tall narrow upright spine", "color": "muted green spine"},
                 {"type": "rect_filled", "role": "book_gold",
                  "placement": "right portion, tall narrow upright spine", "color": "gold spine"},
             ]},
            {"order": 5, "name": "armchair", "layer": "midground",
             "description": "cushioned reading chair in warm muted fabric",
             "zones": ["mid-right"], "bbox": {"x": 562, "y": 225, "w": 150, "h": 150},
             "shapes": [
                 {"type": "rect_filled", "role": "seat",
                  "placement": "full width, bottom 30%", "color": "warm muted fabric"},
                 {"type": "rect_filled", "role": "back",
                  "placement": "full width, upper 50%", "color": "warm muted fabric, slightly darker"},
                 {"type": "rect_filled", "role": "left_arm",
                  "placement": "left edge, narrow vertical strip", "color": "warm muted fabric"},
                 {"type": "rect_filled", "role": "right_arm",
                  "placement": "right edge, narrow vertical strip", "color": "warm muted fabric"},
             ]},
            {"order": 6, "name": "person", "layer": "foreground",
             "description": "a figure leaning forward, face hidden behind an open book, softly lit from above",
             "zones": ["mid-right"], "bbox": {"x": 600, "y": 210, "w": 112, "h": 150},
             "shapes": [
                 {"type": "polygon", "role": "silhouette",
                  "placement": "centered — hunched forward, book raised to face level, knees at bottom",
                  "color": "dark warm shadow with subtle highlight on shoulders"}
             ]},
        ]
    }, indent=2),
    "output": json.dumps({
        "actions": [
            {
                "element": "walls",
                "action": "Flood the full 900×400 px wall area (x=0–900, y=0–400) with a broad horizontal wash of warm amber golden tone edge to edge, establishing the wooden-paneled backdrop.",
                "reason": "The wall fill is the background foundation — must go down first so all midground elements paint over it."
            },
            {
                "element": "floor",
                "action": "Fill the full 900×200 px floor strip (x=0–900, y=400–600) with a flat dark hardwood brown rect, using horizontal strokes to suggest wood grain.",
                "reason": "The floor is background and must be laid down before any furniture is placed on it."
            },
            {
                "element": "bookshelf",
                "action": "In the left 225×200 px region (x=0–225, y=200–400): first fill the rect_filled back_panel in deep amber wood tone. Then draw three thin rect_filled shelf planks in slightly lighter amber — shelf_top near y=267, shelf_mid near y=300, shelf_low near y=333.",
                "reason": "The bookshelf midground must be drawn before the books that rest on it."
            },
            {
                "element": "books",
                "action": "In the left zone (x=0–225, y=200–300): paint three tall narrow rect_filled spines side by side — book_red on the left in muted red, book_green in the centre in muted green, book_gold on the right in gold.",
                "reason": "Books sit on the bookshelf and must be painted after it."
            },
            {
                "element": "armchair",
                "action": "In the right-centre zone (x=562–712, y=225–375): paint the rect_filled seat (bottom 30%, full width) in warm muted fabric, then the rect_filled back (upper 50%) in slightly darker fabric, then the narrow rect_filled left_arm and right_arm strips along each side.",
                "reason": "The armchair is midground and must be drawn before the person who sits in it."
            },
            {
                "element": "person",
                "action": "In the 112×150 px region (x=600–712, y=210–360): paint a dark warm-shadow polygon silhouette — upper body hunched forward fills the top two-thirds, head tilted down, open book raised to face level. Add a soft lighter stroke along the top of the shoulders and crown to suggest overhead lighting.",
                "reason": "The person is the foreground subject and is painted last so it sits in front of all other elements."
            }
        ]
    }, indent=2),
}


def _build_s9_prompt() -> ChatPromptTemplate:
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=[_S9_EXAMPLE],
    )
    return ChatPromptTemplate.from_messages([
        ("system", _S9_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — Step Reflection
# Input:  steps from Stage 9 (each step has ALL fields: zones, bbox, shapes, action, reason)
# Output: {"corrections": [{element, action, reason}]} — only changed entries
#
# The LLM outputs ONLY corrections. Python merges them back.
# No few-shot example — same reasoning as Stage 6 and Stage 8.
#
# Checks:
#   1. Vagueness    — action must reference actual shapes (type/role) and bbox/zone numbers
#   2. Spatial      — location words in action must agree with the element's actual zones
#   3. Color        — colors in action must match the element's description and shapes
#   4. Coverage     — every shape (role) in the element's shapes list must be mentioned
# ═══════════════════════════════════════════════════════════════════════════════

_S10_SYSTEM = """You are a painting step reviewer for a digital painting pipeline.

You receive a list of painting steps. Each step contains ALL context about one element:
  - name, layer, description: what the element is
  - zones, bbox: where it sits on the canvas
  - shapes: [{{type, role, placement, color}}] — the geometric primitives
  - action: the painting instruction generated by Stage 9
  - reason: why this element is painted at this point

Your task: review each step's "action" and "reason" for errors and return corrections
for any that need fixing. Only return entries that actually need to change.

CHECKS TO PERFORM:

1. VAGUENESS — the action must be specific:
   - It must reference the actual shape types and roles (e.g. "rect_filled back_panel",
     "polygon silhouette") — not just "paint the element" or "fill the area".
   - It must anchor spatially using zones or bbox numbers (e.g. "in the left 225×200 px region").
   - If the action is too vague, rewrite it using the shapes and bbox as anchors.

2. SPATIAL ACCURACY — location words in the action must match the element's actual zones:
   - If zones=["mid-left"] the action must not say "right side" or "center".
   - If bbox shows the element is in the upper portion the action must not say "bottom".
   - Fix any mismatches.

3. COLOR ACCURACY — every color mentioned in the action must be traceable to the
   element's description or its shapes' color fields. Fix any invented colors.

4. SHAPE COVERAGE — every shape role in the element's shapes list must be referenced
   in the action. If a role is missing, add a mention of it to the action.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly one key: "corrections".
- "corrections": array of {{element, action, reason}} — ONLY for steps that need changes.
- If a step is already correct, do NOT include it in corrections.
- Do NOT change the element name or any other field.
"""


def _build_s10_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", _S10_SYSTEM),
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown.\n\n{input}"),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessGeneratorRefined:
    """
    Refined painting process generator.

    Stages 1-4 produce a validated, ordered element list ready for Stage 5
    (Scene Layout Planner) to assign precise bbox positions and spatial relations.
    """

    def __init__(
        self,
        stage1_model:   str = "qwen3:8b",
        stage2_model:   str = "qwen3:8b",
        stage3_model:   str = "qwen3:8b",
        stage4_model:   str = "qwen3:8b",
        stage51_model:  str = "qwen3:8b",
        stage52_model:  str = "qwen3:30b-a3b",   # larger — scene graph needs more reasoning
        stage6_model:   str = "qwen3:30b-a3b",   # larger — spatial reasoning for reflection
        stage7_model:   str = "qwen3:30b-a3b",   # larger — shape decomposition needs spatial reasoning
        stage8_model:   str = "qwen3:8b",
        stage9_model:   str = "qwen3:8b",
        stage10_model:  str = "qwen3:8b",
        max_reflect_rounds: int = 2,
    ):
        self.max_reflect_rounds = max_reflect_rounds
        self.stage_models = {
            "1":   stage1_model,
            "2":   stage2_model,
            "3":   stage3_model,
            "4":   stage4_model,
            "5.1": stage51_model,
            "5.2": stage52_model,
            "6":   stage6_model,
            "7":   stage7_model,
            "8":   stage8_model,
            "9":   stage9_model,
            "10":  stage10_model,
        }

        def _llm(model):
            return ChatOllama(model=model, temperature=0.3)

        self.extract_chain        = _build_s1_prompt()  | _llm(stage1_model)  | StrOutputParser()
        self.reflect_chain        = _build_s2_prompt()  | _llm(stage2_model)  | StrOutputParser()
        self.order_chain          = _build_s3_prompt()  | _llm(stage3_model)  | StrOutputParser()
        self.order_reflect_chain  = _build_s4_prompt()  | _llm(stage4_model)  | StrOutputParser()
        self.layout_chain         = _build_s51_prompt() | _llm(stage51_model) | StrOutputParser()
        self.scene_graph_chain    = _build_s52_prompt() | _llm(stage52_model) | StrOutputParser()
        self.layout_reflect_chain = _build_s6_prompt()  | _llm(stage6_model)  | StrOutputParser()
        self.object_realize_chain = _build_s7_prompt()  | _llm(stage7_model)  | StrOutputParser()
        self.object_reflect_chain = _build_s8_prompt()  | _llm(stage8_model)  | StrOutputParser()
        self.step_gen_chain       = _build_s9_prompt()  | _llm(stage9_model)  | StrOutputParser()
        self.step_reflect_chain   = _build_s10_prompt() | _llm(stage10_model) | StrOutputParser()

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def run_stage1(self, description: str) -> tuple[str, list]:
        """Extract elements from the description. Returns (scene_type, elements)."""
        print(f"\n Stage 1 — Extracting elements...  [{self.stage_models['1']}]")
        raw = self.extract_chain.invoke({"input": description})
        result = _parse_json(raw)
        scene_type = result.get("scene_type", "unknown")
        elements   = result.get("elements", [])
        print(f"  Found {len(elements)} elements  |  scene: '{scene_type}'")
        return scene_type, elements

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def run_stage2(self, description: str, elements: list) -> list:
        """Validate and correct extracted elements. Returns corrected elements."""
        for round_num in range(1, self.max_reflect_rounds + 1):
            print(f"\n Stage 2 — Element reflection round {round_num}...  [{self.stage_models['2']}]")
            try:
                raw = self.reflect_chain.invoke({
                    "description": description,
                    "elements_json": json.dumps(elements, indent=2),
                })
                result = _parse_json(raw)
            except (ValueError, Exception) as e:
                print(f"  Parse failed ({e}) — keeping current elements.")
                break

            if result.get("is_valid"):
                print("  Elements validated — no issues found.")
                break

            issues = result.get("issues", [])
            print(f"  {len(issues)} issue(s):")
            for issue in issues:
                print(f"    • {issue}")
            elements = result.get("corrected_elements", elements)
            print("  Corrections applied.")

        return elements

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def run_stage3(self, scene_type: str, description: str, elements: list) -> list:
        """Assign painting order, layers, and positions. Returns painting_order."""
        print(f"\n Stage 3 — Assigning painting order...  [{self.stage_models['3']}]")
        order_input = json.dumps({
            "scene_type":  scene_type,
            "description": description,
            "elements":    elements,
        }, indent=2)
        raw = self.order_chain.invoke({"input": order_input})
        result = _parse_json(raw)
        painting_order = result.get("painting_order", [])
        print(f"  Ordered {len(painting_order)} elements.")
        return painting_order

    # ── Stage 4 ───────────────────────────────────────────────────────────────

    def run_stage4(self, description: str, painting_order: list) -> list:
        """Validate painting order logic. Returns corrected order."""
        for round_num in range(1, self.max_reflect_rounds + 1):
            print(f"\n Stage 4 — Order reflection round {round_num}...  [{self.stage_models['4']}]")
            try:
                raw = self.order_reflect_chain.invoke({
                    "description": description,
                    "order_json":  json.dumps(painting_order, indent=2),
                })
                result = _parse_json(raw)
            except (ValueError, Exception) as e:
                print(f"  Parse failed ({e}) — keeping current order.")
                break

            if result.get("is_valid"):
                print("  Order validated — no issues found.")
                break

            issues = result.get("issues", [])
            print(f"  {len(issues)} issue(s):")
            for issue in issues:
                print(f"    • {issue}")
            painting_order = result.get("corrected_order", painting_order)
            print("  Corrections applied.")

        return painting_order

    # ── Stage 5.1 ─────────────────────────────────────────────────────────────

    def run_stage51(self, description: str, scene_type: str,
                    elements_to_place: list, existing_layout: list = None) -> list:
        """
        Place elements on canvas using categorical labels.
        Called twice: Pass 1 for known elements, Pass 2 for gap-fill elements.
        Returns list of layout entries [{name, zones, h_align, v_align, w_fraction, h_fraction}].
        """
        pass_label = "Pass 2 (gap-fill)" if existing_layout else "Pass 1 (known elements)"
        print(f"\n Stage 5.1 — Canvas layout {pass_label}...  [{self.stage_models['5.1']}]")

        raw = self.layout_chain.invoke({
            "input": json.dumps({
                "description":       description,        # original query — always passed
                "scene_type":        scene_type,
                "existing_layout":   existing_layout or [],   # Pass 2 uses Pass 1 output as context
                "elements_to_place": elements_to_place,       # full Stage 4 output, unchanged
            }, indent=2)
        })
        result = _parse_json(raw)
        layout = result.get("layout", [])
        print(f"  Placed {len(layout)} elements.")
        return layout

    # ── Stage 5.2 ─────────────────────────────────────────────────────────────

    def run_stage52(self, description: str, scene_type: str,
                    painting_order: list, layout: list) -> dict:
        """
        Build scene graph: typed relation edges between known elements + identify
        structurally missing elements (gap fill).
        Returns {"nodes": [...], "edges": [...]}.
        Nodes with status "added" are new elements needing Stage 5.1 Pass 2.
        """
        print(f"\n Stage 5.2 — Scene graph + gap fill...  [{self.stage_models['5.2']}]")

        # Enrich Stage 4 painting_order with layout zones from Stage 5.1.
        # Keep ALL Stage 4 fields (order, name, layer, description) so Stage 5.2
        # has full context for building the relation graph and spotting gaps.
        layout_by_name = {e["name"]: e for e in layout}
        enriched_order = []
        for elem in painting_order:
            layout_entry = layout_by_name.get(elem["name"], {})
            entry = {
                "order":       elem.get("order"),
                "name":        elem["name"],
                "layer":       elem["layer"],
                "description": elem.get("description", ""),
                "zones":       layout_entry.get("zones", []),
                "w_fraction":  layout_entry.get("w_fraction", ""),
                "h_fraction":  layout_entry.get("h_fraction", ""),
            }
            # h_align only meaningful when w_fraction != full
            if layout_entry.get("h_align"):
                entry["h_align"] = layout_entry["h_align"]
            # v_align only meaningful when h_fraction != full
            if layout_entry.get("v_align"):
                entry["v_align"] = layout_entry["v_align"]
            enriched_order.append(entry)

        raw = self.scene_graph_chain.invoke({
            "input": json.dumps({
                "description":    description,   # original query — always passed
                "scene_type":     scene_type,
                "painting_order": enriched_order,
            }, indent=2)
        })
        result = _parse_json(raw)

        added   = [n for n in result.get("nodes", []) if n.get("status") == "added"]
        edges   = result.get("edges", [])
        print(f"  {len(edges)} relation edges.")
        print(f"  {len(added)} missing element(s) identified: {[n['name'] for n in added]}")
        return result

    # ── Full pipeline (Stages 1-5) ────────────────────────────────────────────

    def run_stages_1_to_5(self, description: str) -> dict:
        """
        Run Stages 1-5 and return the full intermediate result ready for Stage 6.

        Returns:
          {
            "description":    str,
            "scene_type":     str,
            "painting_order": [           # unified per-element: all fields merged
              {order, name, layer, description, zones, w_fraction, h_fraction,
               h_align? (only if w_fraction != full),
               v_align? (only if h_fraction != full)}
            ],
            "edges": [                    # relation graph from Stage 5.2
              {from, rel, to}
            ]
          }
        """
        # Stages 1-4
        scene_type, elements = self.run_stage1(description)
        elements             = self.run_stage2(description, elements)
        painting_order       = self.run_stage3(scene_type, description, elements)
        painting_order       = self.run_stage4(description, painting_order)

        # Stage 5.1 Pass 1 — place known elements
        layout_pass1 = self.run_stage51(description, scene_type, painting_order)

        # Stage 5.2 — build scene graph + identify missing elements
        scene_graph = self.run_stage52(description, scene_type, painting_order, layout_pass1)

        # Insert added nodes into painting_order at correct positions and renumber
        added_nodes = [n for n in scene_graph.get("nodes", []) if n.get("status") == "added"]
        if added_nodes:
            painting_order = _insert_added_nodes(painting_order, added_nodes)

        # Stage 5.1 Pass 2 — place gap-fill elements with existing layout as context
        layout_pass2 = []
        if added_nodes:
            layout_pass2 = self.run_stage51(
                description, scene_type,
                elements_to_place=added_nodes,
                existing_layout=layout_pass1,
            )

        # Merge painting_order + layout into one unified list per element
        full_layout    = layout_pass1 + layout_pass2
        layout_by_name = {e["name"]: e for e in full_layout}
        unified_order  = []
        for elem in painting_order:
            lo = layout_by_name.get(elem["name"], {})
            entry = {
                "order":       elem["order"],
                "name":        elem["name"],
                "layer":       elem["layer"],
                "description": elem.get("description", ""),
                "zones":       lo.get("zones", []),
                "w_fraction":  lo.get("w_fraction", ""),
                "h_fraction":  lo.get("h_fraction", ""),
            }
            if lo.get("h_align"):
                entry["h_align"] = lo["h_align"]
            if lo.get("v_align"):
                entry["v_align"] = lo["v_align"]
            unified_order.append(entry)

        print(f"\n Stages 1-5 complete. Total elements: {len(unified_order)}")

        return {
            "description":    description,
            "scene_type":     scene_type,
            "painting_order": unified_order,
            "edges":          scene_graph.get("edges", []),
        }

    # ── Stage 6 ───────────────────────────────────────────────────────────────

    def run_stage6(self, description: str, scene_type: str,
                   painting_order: list, edges: list) -> list:
        """
        Layout Reflection — reviews the unified Stage 5 painting_order for:
          1. Spatial coherence (supports/rests-on edges → shared or adjacent zones)
          2. Zone vs description keywords (left/right/upper/lower in element descriptions)
          3. Layer ordering (background < midground < foreground order numbers)
          4. Size plausibility (full-span elements should use fraction=full)
        Returns the corrected painting_order (unchanged if no issues found).
        """
        print(f"\n Stage 6 — Layout reflection...  [{self.stage_models['6']}]")
        raw = self.layout_reflect_chain.invoke({
            "input": json.dumps({
                "description":    description,
                "scene_type":     scene_type,
                "painting_order": painting_order,
                "edges":          edges,
            }, indent=2)
        })
        result = _parse_json(raw)
        corrected = result.get("painting_order", painting_order)
        changes = sum(
            1 for a, b in zip(painting_order, corrected)
            if a.get("zones") != b.get("zones")
            or a.get("w_fraction") != b.get("w_fraction")
            or a.get("h_fraction") != b.get("h_fraction")
        )
        print(f"  {changes} element(s) corrected.")
        return corrected

    def run_stages_1_to_6(self, description: str) -> dict:
        """
        Full pipeline through Stage 6 (layout reflection).
        Returns the final verified layout ready for Stage 7 (Object Realizer).

        Returns:
          {
            "description":    str,
            "scene_type":     str,
            "painting_order": [ {order, name, layer, description, zones,
                                  w_fraction, h_fraction, h_align?, v_align?} ],
            "edges":          [ {from, rel, to} ]
          }
        """
        result = self.run_stages_1_to_5(description)
        corrected_order = self.run_stage6(
            description,
            result["scene_type"],
            result["painting_order"],
            result["edges"],
        )
        result["painting_order"] = corrected_order
        print(f"\n Stages 1-6 complete.")
        return result

    # ── Stage 7 ───────────────────────────────────────────────────────────────

    def run_stage7(self, description: str, painting_order: list) -> list:
        """
        Object Realizer — decomposes each element into geometric primitives.
        Returns a list of {name, shapes} objects in the same order as painting_order.
        The caller merges shapes back into painting_order by matching on name.
        """
        print(f"\n Stage 7 — Object shape decomposition...  [{self.stage_models['7']}]")
        raw = self.object_realize_chain.invoke({
            "input": json.dumps({
                "description":    description,
                "painting_order": painting_order,
            }, indent=2)
        })
        result  = _parse_json(raw)
        objects = result.get("objects", [])
        total_shapes = sum(len(o.get("shapes", [])) for o in objects)
        print(f"  {len(objects)} element(s) realized, {total_shapes} shape(s) total.")
        return objects

    def run_stages_1_to_7(self, description: str) -> dict:
        """
        Full pipeline through Stage 7 (object shape decomposition).

        Returns:
          {
            "description":    str,
            "scene_type":     str,
            "painting_order": [
              { order, name, layer, description, zones, w_fraction, h_fraction,
                h_align?, v_align?,
                "shapes": [{type, role, color, ...coords...}, ...]
              }
            ],
            "edges": [{from, rel, to}]
          }
        """
        result = self.run_stages_1_to_6(description)

        objects = self.run_stage7(description, result["painting_order"])

        # Merge shapes into painting_order by name
        shapes_by_name = {o["name"]: o.get("shapes", []) for o in objects}
        for elem in result["painting_order"]:
            elem["shapes"] = shapes_by_name.get(elem["name"], [])

        print(f"\n Stages 1-7 complete.")
        return result

    # ── Stage 8 ───────────────────────────────────────────────────────────────

    def run_stage8(self, painting_order: list, objects: list) -> list:
        """
        Object Reflection — validates Stage 7 shapes for type fit, color grounding,
        placement coherence, completeness, and shape count.
        Returns corrected objects list. Python then merges shapes into painting_order.
        """
        print(f"\n Stage 8 — Object shape reflection...  [{self.stage_models['8']}]")
        raw = self.object_reflect_chain.invoke({
            "input": json.dumps({
                "painting_order": painting_order,
                "objects":        objects,
            }, indent=2)
        })
        result   = _parse_json(raw)
        corrected = result.get("objects", objects)
        changes = sum(
            1 for a, b in zip(objects, corrected)
            if a.get("shapes") != b.get("shapes")
        )
        print(f"  {changes} element(s) had shapes corrected.")
        return corrected

    def run_stages_1_to_8(self, description: str) -> dict:
        """
        Full pipeline through Stage 8 (object reflection + merge).

        Returns the fully merged painting_order ready for Stage 9 (Step Generator):
          {
            "description":    str,
            "scene_type":     str,
            "painting_order": [
              { order, name, layer, description, zones, w_fraction, h_fraction,
                h_align?, v_align?,
                "shapes": [{type, role, placement, color, thickness?}, ...]
              }
            ],
            "edges": [{from, rel, to}]
          }
        """
        result  = self.run_stages_1_to_7(description)
        objects = [{"name": e["name"], "shapes": e.get("shapes", [])}
                   for e in result["painting_order"]]

        corrected = self.run_stage8(result["painting_order"], objects)

        # Merge validated shapes back into painting_order
        shapes_by_name = {o["name"]: o.get("shapes", []) for o in corrected}
        for elem in result["painting_order"]:
            elem["shapes"] = shapes_by_name.get(elem["name"], elem.get("shapes", []))

        print(f"\n Stages 1-8 complete.")
        return result

    # ── Stage 9 ───────────────────────────────────────────────────────────────

    def run_stage9(self, result: dict, canvas_w: int, canvas_h: int) -> dict:
        """
        Step Generator — adds bbox + action + reason to each element.
        Assembles process_json in memory only — Stage 10 writes the final file.

        A) Python computes pixel bboxes from zones + fractions + canvas size.
        B) LLM generates {{element, action, reason}} per element.
        C) Python merges action/reason into painting_order (all Stage 1-8 fields preserved).
        D) Assembles process_json dict in result (no file write here).
        """
        print(f"\n Stage 9 — Step generation...  [{self.stage_models['9']}]")

        painting_order = result["painting_order"]

        # A) Pixel bboxes
        for elem in painting_order:
            elem["bbox"] = _resolve_element_bbox(elem, canvas_w, canvas_h)

        # B) LLM generates action + reason
        raw = self.step_gen_chain.invoke({
            "input": json.dumps({
                "description":    result["description"],
                "canvas_w":       canvas_w,
                "canvas_h":       canvas_h,
                "painting_order": painting_order,
            }, indent=2)
        })
        actions_result = _parse_json(raw)
        actions        = actions_result.get("actions", [])

        # C) Merge into painting_order by element name
        action_by_name = {a["element"]: a for a in actions}
        for elem in painting_order:
            info = action_by_name.get(elem["name"], {})
            elem["action"] = info.get("action", "")
            elem["reason"] = info.get("reason", "")

        print(f"  {len(actions)} action(s) generated.")

        # D) Assemble process.json — ALL fields preserved in steps
        process_json = {
            "summary":    result["description"],
            "scene_type": result["scene_type"],
            "canvas":     {"w": canvas_w, "h": canvas_h},
            "steps":      [{**e, "step": e["order"]} for e in painting_order],
            "edges":      result.get("edges", []),
        }

        result["process_json"] = process_json
        return result

    def run_stages_1_to_9(self, description: str,
                          canvas_w: int = 900, canvas_h: int = 600) -> dict:
        """Run Stages 1-9 only (no file write). Kept for incremental testing."""
        result = self.run_stages_1_to_8(description)
        result = self.run_stage9(result, canvas_w, canvas_h)
        print(f"\n Stages 1-9 complete.")
        return result

    # ── Stage 10 ──────────────────────────────────────────────────────────────

    def run_stage10(self, result: dict, output_path: str = "process_out.json") -> dict:
        """
        Step Reflection — reviews each step's action/reason for vagueness, spatial
        accuracy, color accuracy, and shape coverage. Only returns corrections.
        Python merges them back and rewrites process.json.
        """
        print(f"\n Stage 10 — Step reflection...  [{self.stage_models['10']}]")

        steps = result["process_json"]["steps"]

        raw = self.step_reflect_chain.invoke({
            "input": json.dumps({"steps": steps}, indent=2)
        })
        corrections_result = _parse_json(raw)
        corrections        = corrections_result.get("corrections", [])

        if corrections:
            correction_by_name = {c["element"]: c for c in corrections}
            for step in steps:
                if step["name"] in correction_by_name:
                    c = correction_by_name[step["name"]]
                    step["action"] = c.get("action", step["action"])
                    step["reason"] = c.get("reason", step["reason"])

            # Also update painting_order to stay in sync
            for elem in result["painting_order"]:
                if elem["name"] in correction_by_name:
                    c = correction_by_name[elem["name"]]
                    elem["action"] = c.get("action", elem.get("action", ""))
                    elem["reason"] = c.get("reason", elem.get("reason", ""))

            print(f"  {len(corrections)} step(s) corrected.")
        else:
            print(f"  No corrections needed.")

        # Always write process.json here — Stage 9 only assembled it in memory
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result["process_json"], f, indent=2, ensure_ascii=False)
        print(f"  Saved → {output_path}")

        return result

    def run_stages_1_to_10(self, description: str,
                           canvas_w: int = 900, canvas_h: int = 600,
                           output_path: str = "process_out.json") -> dict:
        """Full pipeline through Stage 10 — final validated process.json."""
        result = self.run_stages_1_to_9(description, canvas_w, canvas_h)
        result = self.run_stage10(result, output_path)
        print(f"\n Stages 1-10 complete.")
        return result

    def run_stages_1_to_4(self, description: str) -> dict:
        """Run Stages 1-4 only (no layout). Kept for incremental testing."""
        scene_type, elements = self.run_stage1(description)
        elements             = self.run_stage2(description, elements)
        painting_order       = self.run_stage3(scene_type, description, elements)
        painting_order       = self.run_stage4(description, painting_order)
        return {
            "scene_type":     scene_type,
            "description":    description,
            "elements":       elements,
            "painting_order": painting_order,
        }


def _insert_added_nodes(painting_order: list, added_nodes: list) -> list:
    """
    Insert gap-fill nodes into painting_order at the position indicated by
    each node's insert_after field, then renumber all orders sequentially.
    Multiple nodes with the same insert_after are inserted in the order they appear.
    """
    added_nodes_sorted = sorted(added_nodes, key=lambda n: n.get("insert_after", 0))

    result = list(painting_order)
    for node in added_nodes_sorted:
        insert_after_order = node.get("insert_after", 0)
        insert_idx = next(
            (i for i, e in enumerate(result) if e.get("order") == insert_after_order),
            -1
        )
        new_entry = {
            "order":       insert_after_order,  # placeholder; renumbered below
            "name":        node["name"],
            "layer":       node.get("layer", "midground"),
            "description": node.get("description", ""),
        }
        result.insert(insert_idx + 1, new_entry)

    for i, elem in enumerate(result, start=1):
        elem["order"] = i

    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python process_generator_refined.py \"description\" [output.json] [WxH]")
        sys.exit(1)

    description  = sys.argv[1]
    output_path  = sys.argv[2] if len(sys.argv) > 2 else "process_out.json"
    canvas_w, canvas_h = 900, 600
    if len(sys.argv) > 3:
        canvas_w, canvas_h = map(int, sys.argv[3].split("x"))

    gen    = ProcessGeneratorRefined()
    result = gen.run_stages_1_to_10(description, canvas_w, canvas_h, output_path)
    print("\n" + "═" * 60)
    print(json.dumps(result.get("process_json", {}), indent=2))
