# process_generator2.py
# Stage 1: extract visual elements from the description
# Stage 2: self-reflection to verify and correct extracted elements
# Stage 3: assign painting order and positional info to each element
# Stage 4: self-reflection to verify the painting order
# Stage 5: zone layout map + structural gap fill + conflict validation
# Stage 6: generate painting steps from the final ordered element list
# Stage 7: self-reflection to verify the painting steps

import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ─── STAGE 1: ELEMENT EXTRACTION ─────────────────────────────────────────────

EXTRACT_SYSTEM = """You are a visual element extractor for digital painting planning.

Given a painting description, extract every distinct visual element as JSON.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly two keys: "scene_type" and "elements".
- "scene_type": a short label for the overall scene (string).
- "elements": an array of objects. Each element must have:
    - "name": short label for this element (string)
    - "description": capture EVERY detail the prompt mentions about this element — colors, shapes, motion, texture, style, position, mood. Use the prompt's own words as much as possible. Do not summarize or compress. If the prompt says many things about an element, include all of them (string)
- List every distinct visual element separately. Do not merge different objects.
- Do not add any detail that is not stated in the prompt. Only write what the prompt explicitly says about each element.
- Do not assign layers — layer assignment happens in a later stage.
- DO NOT extract abstract concepts as elements. Reject anything that is not a physical thing you can paint: "composition", "style", "mood", "atmosphere", "tension", "balance", "theme", "energy", "feeling", "contrast", "dynamic", "narrative", or any other concept that cannot be rendered as a visible object or surface.
"""

_EXTRACT_EXAMPLES = [
    {
        "input": "The sky is painted in deep amber and burnt orange, with streaks of crimson near the horizon blending upward into a dusty rose. The sun is a flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo. The mountains are rendered in cool purple and slate blue, arranged in three overlapping ridges that recede into atmospheric haze, each ridge lighter and smaller than the one in front. The lake in the foreground is perfectly still and dark, its surface a near-black mirror that reflects the warm amber sky and the pale glow of the sun.",
        "output": json.dumps({
            "scene_type": "sunset mountain lake",
            "elements": [
                {
                    "name": "sky",
                    "description": "deep amber and burnt orange with streaks of crimson near the horizon blending upward into a dusty rose"
                },
                {
                    "name": "sun",
                    "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo"
                },
                {
                    "name": "mountains",
                    "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze, each ridge lighter and smaller than the one in front"
                },
                {
                    "name": "lake",
                    "description": "perfectly still and dark, near-black mirror surface reflecting the warm amber sky and the pale glow of the sun"
                }
            ]
        }, indent=2)
    }
]


def _build_extract_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=_EXTRACT_EXAMPLES
    )
    return ChatPromptTemplate.from_messages([
        ("system", EXTRACT_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown, no bold text, no headers.\n\n{input}")
    ])


# ─── STAGE 2: ELEMENT SELF-REFLECTION ────────────────────────────────────────

REFLECT_SYSTEM = """You are a painting description validator.

You will receive an original painting description and a JSON list of extracted visual elements.
Each element has a "name" and "description" field.
Your task is to check whether the extracted elements faithfully represent the description.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly three keys: "is_valid", "issues", "corrected_elements".
- "is_valid": true if elements are accurate and complete, false if problems exist (boolean).
- "issues": list of strings describing problems — missing elements, hallucinated details in description (array). Empty list if none.
- "corrected_elements": the fixed elements array. If is_valid is true, return the original array unchanged.
- Flag only real problems: missing elements, descriptions that contain details not in the prompt, descriptions that are missing details the prompt explicitly states about that element, or abstract concepts that slipped through as elements — "composition", "style", "mood", "atmosphere", "tension", "balance", "theme", "energy", "feeling", "contrast", "dynamic", "narrative", or any concept that cannot be rendered as a visible object or surface on canvas (remove these from corrected_elements).
- For each element, re-read every sentence in the original description that mentions it and check that all those details appear in the element's description field.
- Do not change elements that are correctly and completely extracted.
- Do not assign or validate layers — that happens in a later stage.
"""


def _build_reflect_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", REFLECT_SYSTEM),
        ("human", "/no_think\nORIGINAL DESCRIPTION:\n{description}\n\nEXTRACTED ELEMENTS:\n{elements_json}")
    ])


# ─── STAGE 3: PAINTING ORDER + POSITION ──────────────────────────────────────

ORDER_SYSTEM = """You are a digital painting sequence planner.

You will receive the original painting description and a validated list of visual elements (name and description only).
Your task: assign each element a layer, a painting order, and a canvas position.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly two keys: "scene_type" and "painting_order".
- "painting_order": an ordered array where each object has:
    - "order": integer starting from 1 (background elements first)
    - "name": same as input
    - "layer": assign a descriptive Krita layer name for this element (string). See layer assignment and naming rules below.
    - "description": same as input
    - "position": where this element sits on the canvas. If the prompt mentions a position for this element, use those exact words. If the prompt does not mention position, use general painting knowledge to assign a reasonable canvas location (string)
- Order strictly back to front: all background elements before midground before foreground.
- Within the same conceptual layer, order by which must be painted first (elements painted under others come first).

LAYER ASSIGNMENT GUIDE:
- "background": the scene environment — sky, walls, floor, ground plane, distant landscape, ceiling. Anything that forms the "room" or "world" that objects and characters inhabit.
- "midground": objects, furniture, props, vehicles — things placed within the scene that are neither the ground/walls nor the main subject.
- "foreground": characters, figures, or elements that are the primary subject closest to the viewer.

LAYER NAMING RULES:
- If a conceptual group (background / midground / foreground) has 3 or fewer elements, use the group name directly: "background", "midground", "foreground".
- If a conceptual group has more than 3 elements, split them into descriptive named sub-layers instead of stacking everything on one layer. Use the format "{{group}} – {{descriptor}}", e.g. "foreground – trees", "foreground – character", "background – sky", "background – ground".
- Do not add, remove, or rename any element. Only assign layer, order, and position.
"""

_ORDER_EXAMPLES = [
    {
        "input": json.dumps({
            "scene_type": "sunset mountain lake",
            "description": "The sky is painted in deep amber and burnt orange. The sun is a flattened glowing disc in pale gold, half-submerged behind the mountains. The mountains are in three overlapping ridges. The lake in the foreground is perfectly still.",
            "elements": [
                {"name": "sky", "description": "deep amber and burnt orange with streaks of crimson near the horizon blending upward into a dusty rose"},
                {"name": "sun", "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo"},
                {"name": "mountains", "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze"},
                {"name": "lake", "description": "perfectly still and dark, near-black mirror surface reflecting the sky and sun"}
            ]
        }, indent=2),
        "output": json.dumps({
            "scene_type": "sunset mountain lake",
            "painting_order": [
                {"order": 1, "name": "sky", "layer": "background", "description": "deep amber and burnt orange with streaks of crimson near the horizon blending upward into a dusty rose", "position": "fills the upper two-thirds of the canvas"},
                {"order": 2, "name": "sun", "layer": "background", "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo", "position": "center of the horizon line, partially hidden behind the mountains"},
                {"order": 3, "name": "mountains", "layer": "midground", "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze", "position": "spans the full width at the horizon, mid-canvas vertically"},
                {"order": 4, "name": "lake", "layer": "foreground", "description": "perfectly still and dark, near-black mirror surface reflecting the sky and sun", "position": "fills the lower third of the canvas"}
            ]
        }, indent=2)
    }
]


def _build_order_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=_ORDER_EXAMPLES
    )
    return ChatPromptTemplate.from_messages([
        ("system", ORDER_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown, no bold text, no headers.\n\n{input}")
    ])


# ─── STAGE 4: ORDER SELF-REFLECTION ──────────────────────────────────────────

ORDER_REFLECT_SYSTEM = """You are a painting order validator.

You will receive the original painting description and a JSON painting order list.
Each entry has an order number, name, layer, description, and position.
Your task: check whether the painting order and positions are correct.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly three keys: "is_valid", "issues", "corrected_order".
- "is_valid": true if order and positions are correct, false otherwise (boolean).
- "issues": list of strings describing problems (array). Empty list if none.
- "corrected_order": the fixed painting_order array. If is_valid is true, return original unchanged.
- Flag these problems:
    - Any background element ordered after a midground or foreground element
    - Any midground element ordered after a foreground element
    - An element that visually sits beneath another but is assigned a higher order number, meaning it would be painted after the element that should cover it
    - A position that contradicts what the original prompt says about that element
    - A conceptual group (background / midground / foreground) that has more than 3 elements all assigned to the same single layer name — they should have been split into descriptive sub-layers
- Do not flag the position if the prompt does not mention it — LLM-inferred positions are acceptable.
"""


def _build_order_reflect_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", ORDER_REFLECT_SYSTEM),
        ("human", "/no_think\nORIGINAL DESCRIPTION:\n{description}\n\nPAINTING ORDER:\n{order_json}")
    ])


# ─── STAGE 5: ZONE LAYOUT MAP + STRUCTURAL GAP FILL + CONFLICT CHECK ─────────

# ── 5.1 Zone Layout Map ───────────────────────────────────────────────────────

ZONE_MAP_SYSTEM = """You are a canvas layout classifier for digital painting.

You will receive a JSON object containing a scene description, a scene type, and a painting_order list.
Your task: for each element in painting_order, classify it into ALL canvas grid cells it occupies based on its position field and description.
This is a classification task. Do NOT ask questions. Do NOT respond conversationally. Process the input and output JSON immediately.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly one key: "element_zones".
- "element_zones": array of objects, one per element, each with:
    - "name": element name (string)
    - "zones": array of ALL canvas grid cells this element occupies — use "row-col" format for each (array of strings). List every cell the element touches, not just the dominant one.
    - "layer_type": "background", "midground", or "foreground"

CANVAS ZONE FORMAT:
  Grid rows:    "upper" | "mid" | "lower"
  Grid columns: "left"  | "center" | "right"

  Format:   "<row>-<col>"   e.g. "upper-left", "mid-center", "lower-right"

List ALL cells an element spans. An element that covers two rows should have entries for each row.
"""

_ZONE_MAP_EXAMPLES = [
    {
        "input": json.dumps({
            "description": "A person reading alone in a cozy library with stacks of books and warm wooden walls.",
            "scene_type": "cozy library reading room",
            "painting_order": [
                {"order": 1, "name": "walls", "layer": "background", "description": "warm amber wood-paneled walls with a deep golden tone", "position": "spans full height across the back of the scene"},
                {"order": 2, "name": "floor", "layer": "background", "description": "dark hardwood flooring with visible grain lines running horizontally", "position": "lower third of the canvas"},
                {"order": 3, "name": "books", "layer": "midground", "description": "stacks of thick hardcover books in muted reds, greens, and gold spines clustered together", "position": "left side of the canvas, mid-height"},
                {"order": 4, "name": "person", "layer": "foreground", "description": "a figure leaning forward, face hidden behind an open book, softly lit from above", "position": "right side of the canvas, mid-height, seated"}
            ]
        }, indent=2),
        "output": json.dumps({
            "element_zones": [
                {"name": "walls", "zones": ["upper-left", "upper-center", "upper-right", "mid-left", "mid-center", "mid-right"], "layer_type": "background"},
                {"name": "floor", "zones": ["lower-left", "lower-center", "lower-right"], "layer_type": "background"},
                {"name": "books", "zones": ["mid-left"], "layer_type": "midground"},
                {"name": "person", "zones": ["mid-right"], "layer_type": "foreground"}
            ]
        }, indent=2)
    }
]


def _build_zone_map_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=_ZONE_MAP_EXAMPLES
    )
    return ChatPromptTemplate.from_messages([
        ("system", ZONE_MAP_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown, no questions, no explanation.\n\n{input}")
    ])


# ── 5.2 Structural Gap Fill ───────────────────────────────────────────────────

GAP_FILL_SYSTEM = """You are a structural completeness analyst for digital painting scenes.

You will receive the original painting description, the current painting order, and an element_zones map
showing which canvas grid cell each element occupies.

Your task: for EACH element in the painting_order, ask —
  "Does this element require a physical surface, seat, or structural support that is NOT already
   present in the painting_order under any name?"

Use the element_zones to understand spatial relationships (e.g. what is above/below another element).
Do NOT limit yourself to "empty" cells — a missing support element may share the same grid cell
as another element

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly two keys: "gaps_found" and "new_elements".
- "gaps_found": true if any missing structural elements were identified, false otherwise (boolean).
- "new_elements": array of proposed elements to add. Empty array [] if none needed. Each must have:
    - "name": short label (string)
    - "description": brief physical description in context of this scene (string)
    - "inferred_from": the existing element that structurally implies this one cannot be absent (string)
    - "support_cell": the canvas grid cell where this support element would sit (string)
    - "scene_role": the structural role this element plays — invent a descriptive name if needed (string)
    - "suggested_layer": "background", "midground", or "foreground" (string)
    - "suggested_position": canvas position for this element (string)
    - "insert_after_order": order number of the existing element after which this should be inserted (integer)

MANDATORY BACKGROUND RULE: Every painting must have at least one background element (sky, wall, ground, backdrop, etc.). If the painting_order contains no element with layer "background", you MUST propose an appropriate background element regardless of whether any other gaps exist. Infer the most fitting background from the scene type and existing elements.

CRITICAL CONSTRAINT — only propose a non-background element if BOTH are true:
1. Nothing in the current painting order already serves this structural role under any name
2. The proposed element is physically plausible for this scene type — it is a natural structural support given the setting and existing elements
Do NOT propose decorative, atmospheric, or merely probable elements.
If no structural gaps exist, return "gaps_found": false and "new_elements": [].
"""

_GAP_FILL_EXAMPLES = [
    {
        "input": json.dumps({
            "description": "A person reading alone in a cozy library with stacks of books and warm wooden walls.",
            "painting_order": [
                {"order": 1, "name": "walls", "layer": "background", "description": "warm amber wood-paneled walls with a deep golden tone", "position": "spans full height across the back of the scene"},
                {"order": 2, "name": "floor", "layer": "background", "description": "dark hardwood flooring with visible grain lines running horizontally", "position": "lower third of the canvas"},
                {"order": 3, "name": "books", "layer": "midground", "description": "stacks of thick hardcover books in muted reds, greens, and gold spines clustered together", "position": "left side of the canvas, mid-height"},
                {"order": 4, "name": "person", "layer": "foreground", "description": "a figure leaning forward, face hidden behind an open book, softly lit from above", "position": "right side of the canvas, mid-height, seated"}
            ],
            "element_zones": [
                {"name": "walls", "zones": ["upper-left", "upper-center", "upper-right", "mid-left", "mid-center", "mid-right"], "layer_type": "background"},
                {"name": "floor", "zones": ["lower-left", "lower-center", "lower-right"], "layer_type": "background"},
                {"name": "books", "zones": ["mid-left"], "layer_type": "midground"},
                {"name": "person", "zones": ["mid-right"], "layer_type": "foreground"}
            ]
        }, indent=2),
        "output": json.dumps({
            "gaps_found": True,
            "new_elements": [
                {
                    "name": "bookshelf",
                    "description": "tall wooden shelving unit with horizontal planks matching the amber wall tone, supporting the stacked books above",
                    "inferred_from": "books — stacked books at mid-left cannot float; a shelf beneath them is structurally required",
                    "support_cell": "lower-left",
                    "scene_role": "shelving-unit",
                    "suggested_layer": "midground",
                    "suggested_position": "lower-left, beneath the book stacks, resting on the floor",
                    "insert_after_order": 2
                },
                {
                    "name": "armchair",
                    "description": "cushioned reading chair in a warm muted fabric, low to the ground, supporting the seated figure",
                    "inferred_from": "person — the figure is described as seated at mid-right; a seated person cannot float; a chair beneath them is structurally required",
                    "support_cell": "lower-right",
                    "scene_role": "seating-surface",
                    "suggested_layer": "midground",
                    "suggested_position": "lower-right, directly beneath the seated figure, resting on the floor",
                    "insert_after_order": 3
                }
            ]
        }, indent=2)
    }
]


def _build_gap_fill_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=_GAP_FILL_EXAMPLES
    )
    return ChatPromptTemplate.from_messages([
        ("system", GAP_FILL_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown, no bold text, no headers.\n\n{input}")
    ])


# ── 5.3 Conflict Check + Merge ────────────────────────────────────────────────

CONFLICT_CHECK_SYSTEM = """You are a painting order integrator.

You will receive the original painting description, the current validated painting order, and a list of proposed new elements.
Your task: validate each proposed element and produce a final merged painting order.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly three keys: "merged_painting_order", "added", "rejected".
- "merged_painting_order": complete painting order with approved elements inserted, re-indexed from 1 (array). Each entry must have: "order", "name", "layer", "description", "position".
- "added": list of element names that were approved and inserted (array of strings).
- "rejected": array of objects with "name" and "reason" for each rejected element.

REJECT a proposed element if:
- An existing element already covers its structural role (same function under any name)
- Its layer assignment is inconsistent with its structural role in the scene
- It is decorative or not physically necessary

For each approved element: insert it after the element at "insert_after_order", use "suggested_layer" as "layer" and "suggested_position" as "position". Re-index all order numbers cleanly from 1 after all insertions.
"""

_CONFLICT_CHECK_EXAMPLES = [
    {
        "input": json.dumps({
            "description": "A person reading alone in a cozy library with stacks of books and warm wooden walls.",
            "painting_order": [
                {"order": 1, "name": "walls", "layer": "background", "description": "warm amber wood-paneled walls with a deep golden tone", "position": "spans full height across the back of the scene"},
                {"order": 2, "name": "floor", "layer": "background", "description": "dark hardwood flooring with visible grain lines running horizontally", "position": "lower third of the canvas"},
                {"order": 3, "name": "books", "layer": "midground", "description": "stacks of thick hardcover books in muted reds, greens, and gold spines clustered together", "position": "left side of the canvas, mid-height"},
                {"order": 4, "name": "person", "layer": "foreground", "description": "a figure leaning forward, face hidden behind an open book, softly lit from above", "position": "right side of the canvas, mid-height, seated"}
            ],
            "new_elements": [
                {
                    "name": "bookshelf",
                    "description": "tall wooden shelving unit with horizontal planks matching the amber wall tone, supporting the stacked books above",
                    "inferred_from": "books",
                    "suggested_layer": "midground",
                    "suggested_position": "lower-left, beneath the book stacks, resting on the floor",
                    "insert_after_order": 2
                },
                {
                    "name": "armchair",
                    "description": "cushioned reading chair in a warm muted fabric, low to the ground, supporting the seated figure",
                    "inferred_from": "person",
                    "suggested_layer": "midground",
                    "suggested_position": "lower-right, directly beneath the seated figure, resting on the floor",
                    "insert_after_order": 3
                }
            ]
        }, indent=2),
        "output": json.dumps({
            "merged_painting_order": [
                {"order": 1, "name": "walls", "layer": "background", "description": "warm amber wood-paneled walls with a deep golden tone", "position": "spans full height across the back of the scene"},
                {"order": 2, "name": "floor", "layer": "background", "description": "dark hardwood flooring with visible grain lines running horizontally", "position": "lower third of the canvas"},
                {"order": 3, "name": "bookshelf", "layer": "midground", "description": "tall wooden shelving unit with horizontal planks matching the amber wall tone, supporting the stacked books above", "position": "lower-left, beneath the book stacks, resting on the floor"},
                {"order": 4, "name": "books", "layer": "midground", "description": "stacks of thick hardcover books in muted reds, greens, and gold spines clustered together", "position": "left side of the canvas, mid-height"},
                {"order": 5, "name": "armchair", "layer": "midground", "description": "cushioned reading chair in a warm muted fabric, low to the ground, supporting the seated figure", "position": "lower-right, directly beneath the seated figure, resting on the floor"},
                {"order": 6, "name": "person", "layer": "foreground", "description": "a figure leaning forward, face hidden behind an open book, softly lit from above", "position": "right side of the canvas, mid-height, seated"}
            ],
            "added": ["bookshelf", "armchair"],
            "rejected": []
        }, indent=2)
    }
]


def _build_conflict_check_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=_CONFLICT_CHECK_EXAMPLES
    )
    return ChatPromptTemplate.from_messages([
        ("system", CONFLICT_CHECK_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown, no bold text, no headers.\n\n{input}")
    ])


# ─── STAGE 6: PAINTING STEP GENERATION ───────────────────────────────────────

STEPS_SYSTEM = """You are an expert digital painting process planner for Krita.

You will receive the original painting description and an ordered list of visual elements with their canvas positions.
Your task: generate a detailed, step-by-step painting process as JSON.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly two keys: "summary" and "steps".
- "summary": a short descriptive title for the painting (string).
- "steps": an ordered array of step objects. Each step must have exactly four keys:
    - "step": integer (1-based index)
    - "action": the concrete painting action — reference the Krita layer name, the canvas position from the painting_order, specific colors, and the element being painted. The position field from the painting_order must appear explicitly in the action (e.g. "fills the lower third", "spans the full width at the horizon") (string)
    - "reason": the painting principle that explains why this step happens at this point (string)
    - "zones": the list of canvas zone labels where this element lives, taken directly from element_zones in the input. If element_zones is not provided, set this to an empty list []. Do NOT invent zone labels — copy them exactly as given. Zone labels follow the format: "upper-left", "upper-center", "upper-right", "mid-left", "mid-center", "mid-right", "lower-left", "lower-center", "lower-right".
- Follow the painting_order strictly — all background elements before midground before foreground.
- Each element must appear in at least one step. Complex elements may span multiple steps (base coat first, then details on a separate pass).
- Use the exact layer name from each element's "layer" field in the painting_order when referencing Krita layers in the action.
- Do not skip any element from the painting_order.
- If an element's description does not mention specific colors or visual appearance details, infer appropriate ones that are consistent with the overall scene's style, mood, and color palette as established by the other elements. Do not leave the action vague — always specify colors, tones, or textures even when the prompt does not.
"""

_STEPS_EXAMPLES = [
    {
        "input": json.dumps({
            "scene_type": "sunset mountain lake",
            "description": "The sky is painted in deep amber and burnt orange, with streaks of crimson near the horizon. The sun is a flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo. The mountains are rendered in cool purple and slate blue, three overlapping ridges receding into atmospheric haze. The lake in the foreground is perfectly still and dark, its surface a near-black mirror reflecting the sky and sun.",
            "painting_order": [
                {"order": 1, "name": "sky", "layer": "background", "description": "deep amber and burnt orange with streaks of crimson near the horizon blending upward into a dusty rose", "position": "fills the upper two-thirds of the canvas"},
                {"order": 2, "name": "sun", "layer": "background", "description": "flattened glowing disc in pale gold, half-submerged behind the mountains, casting a warm radiant halo", "position": "center of the horizon line, partially hidden behind the mountains"},
                {"order": 3, "name": "mountains", "layer": "midground", "description": "cool purple and slate blue, three overlapping ridges receding into atmospheric haze", "position": "spans the full width at the horizon, mid-canvas vertically"},
                {"order": 4, "name": "lake", "layer": "foreground", "description": "perfectly still and dark, near-black mirror surface reflecting the sky and sun", "position": "fills the lower third of the canvas"}
            ]
        }, indent=2),
        "output": json.dumps({
            "summary": "Sunset Mountain Lake",
            "steps": [
                {"step": 1, "action": "On a new 'Sky' layer, fill the upper two-thirds of the canvas with a deep amber base using a large soft brush, then blend burnt orange toward the midpoint and dusty rose near the top", "reason": "The sky is the dominant background and must be laid first to anchor the overall warm color mood of the scene"},
                {"step": 2, "action": "On the 'Sky' layer, add streaks of crimson near the horizon using a dry textured brush, blending softly upward into the amber", "reason": "Horizon detail is refined after the base wash to avoid muddying the initial gradient"},
                {"step": 3, "action": "On a new 'Sun' layer above 'Sky', paint a flattened pale gold ellipse at the center of the horizon; apply a soft low-opacity airbrush halo radiating outward", "reason": "The sun is placed after the sky base so the glow blends naturally into the existing warm tones"},
                {"step": 4, "action": "On a new 'Mountains' layer, paint the farthest ridge in a light slate blue, then the middle ridge in mid-purple, then the nearest ridge in deeper cool purple, each overlapping the previous", "reason": "Painting mountain ridges back to front lets each closer ridge cover the edge of the previous, creating atmospheric depth"},
                {"step": 5, "action": "On a new 'Lake' layer, fill the lower third with a near-black base, then use a horizontal soft brush to mirror the amber and pale gold tones from the sky into the water surface", "reason": "The lake is painted last so the reflection accurately references the already-established sky and sun colors above"}
            ]
        }, indent=2)
    }
]


def _build_steps_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=_STEPS_EXAMPLES
    )
    return ChatPromptTemplate.from_messages([
        ("system", STEPS_SYSTEM),
        few_shot,
        ("human", "/no_think\nOutput ONLY valid JSON. No markdown, no bold text, no headers.\n\n{input}")
    ])


# ─── STAGE 7: STEP SELF-REFLECTION ───────────────────────────────────────────

STEPS_REFLECT_SYSTEM = """You are a painting step validator.

You will receive the original painting description, the ordered element list, and generated painting steps.
Your task: check whether the steps correctly and completely cover the painting process.

STRICT RULES:
- Output ONLY valid JSON. No markdown, no explanation.
- Output a JSON object with exactly three keys: "is_valid", "issues", "corrected_steps".
- "is_valid": true if steps are complete and correct, false otherwise (boolean).
- "issues": list of strings describing problems (array). Empty list if none.
- "corrected_steps": the fixed steps array. If is_valid is true, return original unchanged.
- Flag these problems:
    - Any element from the painting_order that has no corresponding step
    - Steps that paint a foreground or midground element before all background elements are done
    - Steps with vague actions that mention no color, brush type, or layer name
    - Step numbers that are out of sequence or duplicated
    - Steps where the "zones" field is missing or empty when element_zones were provided
    - Steps where "zones" contains labels that differ from what was given in element_zones (zones must be copied exactly, not reworded)
- Do not flag stylistic choices — only flag missing coverage, wrong paint order, non-actionable steps, or missing/wrong zone labels.
"""


def _build_steps_reflect_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", STEPS_REFLECT_SYSTEM),
        ("human", "/no_think\nORIGINAL DESCRIPTION:\n{description}\n\nPAINTING ORDER:\n{order_json}\n\nELEMENT ZONES (must appear unchanged in each step's zones field):\n{zones_json}\n\nGENERATED STEPS:\n{steps_json}")
    ])


# ─── JSON HELPER ──────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    # Strip <think>...</think> blocks (qwen3 may output these despite /no_think)
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


# ─── MAIN CLASS ───────────────────────────────────────────────────────────────

class ProcessGenerator2:
    def __init__(
        self,
        stage1_model:   str = "qwen3:8b",
        stage2_model:   str = "qwen3:8b",
        stage3_model:   str = "qwen3:8b",
        stage4_model:   str = "qwen3:8b",
        stage5_1_model: str = "qwen3:8b",
        stage5_2_model: str = "qwen3:30b-a3b",
        stage5_3_model: str = "qwen3:8b",
        stage6_model:   str = "qwen3:30b-a3b",
        stage7_model:   str = "qwen3:30b-a3b",
        max_reflect_rounds: int = 2
    ):
        self.max_reflect_rounds = max_reflect_rounds
        self.stage_models = {
            "1":   stage1_model,
            "2":   stage2_model,
            "3":   stage3_model,
            "4":   stage4_model,
            "5.1": stage5_1_model,
            "5.2": stage5_2_model,
            "5.3": stage5_3_model,
            "6":   stage6_model,
            "7":   stage7_model,
        }
        def _llm(model): return ChatOllama(model=model, temperature=0.3)
        self.extract_chain        = _build_extract_prompt()        | _llm(stage1_model)   | StrOutputParser()
        self.reflect_chain        = _build_reflect_prompt()        | _llm(stage2_model)   | StrOutputParser()
        self.order_chain          = _build_order_prompt()          | _llm(stage3_model)   | StrOutputParser()
        self.order_reflect_chain  = _build_order_reflect_prompt()  | _llm(stage4_model)   | StrOutputParser()
        self.zone_map_chain       = _build_zone_map_prompt()       | _llm(stage5_1_model) | StrOutputParser()
        self.gap_fill_chain       = _build_gap_fill_prompt()       | _llm(stage5_2_model) | StrOutputParser()
        self.conflict_check_chain = _build_conflict_check_prompt() | _llm(stage5_3_model) | StrOutputParser()
        self.steps_chain          = _build_steps_prompt()          | _llm(stage6_model)   | StrOutputParser()
        self.steps_reflect_chain  = _build_steps_reflect_prompt()  | _llm(stage7_model)   | StrOutputParser()

    def generate(self, description: str, output_path: str = None) -> dict:
        # ── Stage 1: Extract elements ──────────────────────────────────────────
        print(f"\n Stage 1 — Extracting elements...  [{self.stage_models['1']}]")
        print("─" * 50)

        raw = self.extract_chain.invoke({"input": description})
        extracted = _parse_json(raw)

        elements = extracted.get("elements", [])
        scene_type = extracted.get("scene_type", "unknown")
        print(f" Found {len(elements)} elements for scene: '{scene_type}'")

        # ── Stage 2: Self-reflection on elements ───────────────────────────────
        for round_num in range(1, self.max_reflect_rounds + 1):
            print(f"\n Stage 2 — Element reflection round {round_num}...  [{self.stage_models['2']}]")

            raw_reflect = self.reflect_chain.invoke({
                "description": description,
                "elements_json": json.dumps(elements, indent=2)
            })
            try:
                reflection = _parse_json(raw_reflect)
            except ValueError as e:
                print(f"  Reflection parse failed ({e}) — keeping current elements.")
                break

            if reflection.get("is_valid"):
                print(" Elements validated — no issues found.")
                break

            issues = reflection.get("issues", [])
            print(f"  {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"   • {issue}")
            elements = reflection.get("corrected_elements", elements)
            print(" Corrections applied.")

        # ── Stage 3: Assign painting order + positions ─────────────────────────
        print(f"\n Final elements: {[e['name'] for e in elements]}")
        print(f"\n Stage 3 — Assigning painting order and positions...  [{self.stage_models['3']}]")
        print("─" * 50)

        order_input = json.dumps({
            "scene_type": scene_type,
            "description": description,
            "elements": elements
        }, indent=2)

        raw_order = self.order_chain.invoke({"input": order_input})
        ordered = _parse_json(raw_order)
        painting_order = ordered.get("painting_order", [])
        print(f" Ordered {len(painting_order)} elements.")

        # ── Stage 4: Self-reflection on painting order ─────────────────────────
        for round_num in range(1, self.max_reflect_rounds + 1):
            print(f"\n Stage 4 — Order reflection round {round_num}...  [{self.stage_models['4']}]")

            raw_order_reflect = self.order_reflect_chain.invoke({
                "description": description,
                "order_json": json.dumps(painting_order, indent=2)
            })
            try:
                order_reflection = _parse_json(raw_order_reflect)
            except ValueError as e:
                print(f"  Reflection parse failed ({e}) — keeping current order.")
                break

            if order_reflection.get("is_valid"):
                print(" Painting order validated — no issues found.")
                break

            issues = order_reflection.get("issues", [])
            print(f"  {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"   • {issue}")
            painting_order = order_reflection.get("corrected_order", painting_order)
            print(" Corrections applied.")

        # ── Stage 5: Zone map + gap fill + conflict check ──────────────────────
        print(f"\n  Stage 5.1 — Canvas zone mapping...  [{self.stage_models['5.1']}]")
        print("─" * 50)

        zone_map_result = None
        try:
            zone_input = json.dumps({
                "description": description,
                "scene_type": scene_type,
                "painting_order": painting_order
            }, indent=2)
            raw_zone = self.zone_map_chain.invoke({"input": zone_input})
            zone_map_result = _parse_json(raw_zone)
            element_zones = zone_map_result.get("element_zones", [])
            zone_summary = [e["name"] + " → " + ", ".join(e.get("zones", [e.get("zone", "?")])) for e in element_zones]
            print(f" Zones mapped: {zone_summary}")
        except (ValueError, Exception) as e:
            print(f"  Zone mapping failed ({e}) — skipping Stage 5.")

        if zone_map_result:
            print(f"\n Stage 5.2 — Structural gap fill...  [{self.stage_models['5.2']}]")
            gap_result = None
            try:
                gap_input = json.dumps({
                    "description": description,
                    "painting_order": painting_order,
                    "element_zones": zone_map_result.get("element_zones", [])
                }, indent=2)
                raw_gap = self.gap_fill_chain.invoke({"input": gap_input})
                gap_result = _parse_json(raw_gap)
                new_elements = gap_result.get("new_elements", [])
                if gap_result.get("gaps_found") and new_elements:
                    print(f" {len(new_elements)} structural gap(s) found: {[e['name'] for e in new_elements]}")
                else:
                    print(" No structural gaps found.")
            except (ValueError, Exception) as e:
                print(f"  Gap fill failed ({e}) — skipping Stage 5.3.")

            if gap_result and gap_result.get("gaps_found") and gap_result.get("new_elements"):
                print(f"\n  Stage 5.3 — Conflict check + merge...  [{self.stage_models['5.3']}]")
                try:
                    conflict_input = json.dumps({
                        "description": description,
                        "painting_order": painting_order,
                        "new_elements": gap_result["new_elements"]
                    }, indent=2)
                    raw_conflict = self.conflict_check_chain.invoke({"input": conflict_input})
                    conflict_result = _parse_json(raw_conflict)
                    merged = conflict_result.get("merged_painting_order")
                    if merged:
                        painting_order = merged
                        added = conflict_result.get("added", [])
                        rejected = conflict_result.get("rejected", [])
                        print(f" Merged. Added: {added}. Rejected: {[r['name'] for r in rejected]}")

                        # Zone_map was built in Stage 5.1 BEFORE Stage 5.2/5.3 added
                        # new elements — so those new elements have no zone entry yet.
                        # Use each new element's support_cell (already computed by
                        # Stage 5.2) as its zone so Stage 6 can embed it in each step.
                        if zone_map_result:
                            added_names      = set(added)
                            existing_names   = {z["name"] for z in zone_map_result.get("element_zones", [])}
                            for new_el in gap_result.get("new_elements", []):
                                name = new_el.get("name", "")
                                if name in added_names and name not in existing_names:
                                    cell = new_el.get("support_cell", "")
                                    zone_map_result["element_zones"].append({
                                        "name":       name,
                                        "zones":      [cell] if cell else [],
                                        "layer_type": new_el.get("suggested_layer", "midground"),
                                    })
                                    print(f"  Zone added for new element '{name}': [{cell}]")
                    else:
                        print("  Conflict check returned no merged order — keeping current.")
                except (ValueError, Exception) as e:
                    print(f"  Conflict check failed ({e}) — keeping current painting order.")

        # ── Save intermediates ─────────────────────────────────────────────────
        if output_path:
            elements_path = output_path.replace(".json", "_elements.json")
            with open(elements_path, "w", encoding="utf-8") as f:
                json.dump({"scene_type": scene_type, "elements": elements}, f, indent=2)
            print(f"\n Elements saved to: {elements_path}")

            order_path = output_path.replace(".json", "_order.json")
            with open(order_path, "w", encoding="utf-8") as f:
                json.dump({"scene_type": scene_type, "painting_order": painting_order}, f, indent=2)
            print(f" Painting order saved to: {order_path}")

        # ── Stage 6: Generate painting steps ──────────────────────────────────
        print(f"\n  Stage 6 — Generating painting steps...  [{self.stage_models['6']}]")
        print("─" * 50)

        # Include element_zones from Stage 5 so Stage 6 can embed them in each step.
        steps_input_dict = {
            "scene_type": scene_type,
            "description": description,
            "painting_order": painting_order,
        }
        if zone_map_result:
            steps_input_dict["element_zones"] = zone_map_result.get("element_zones", [])
        steps_input = json.dumps(steps_input_dict, indent=2)

        raw_steps = self.steps_chain.invoke({"input": steps_input})
        steps_data = _parse_json(raw_steps)
        steps = steps_data.get("steps", [])
        summary = steps_data.get("summary", scene_type)
        print(f" Generated {len(steps)} painting steps.")

        # ── Stage 7: Self-reflection on steps ─────────────────────────────────
        for round_num in range(1, self.max_reflect_rounds + 1):
            print(f"\n Stage 7 — Step reflection round {round_num}...  [{self.stage_models['7']}]")

            raw_steps_reflect = self.steps_reflect_chain.invoke({
                "description": description,
                "order_json": json.dumps(painting_order, indent=2),
                "zones_json": json.dumps(
                    zone_map_result.get("element_zones", []) if zone_map_result else [],
                    indent=2
                ),
                "steps_json": json.dumps(steps, indent=2)
            })
            try:
                steps_reflection = _parse_json(raw_steps_reflect)
            except ValueError as e:
                print(f"  Reflection parse failed ({e}) — keeping current steps.")
                break

            if steps_reflection.get("is_valid"):
                print(" Steps validated — no issues found.")
                break

            issues = steps_reflection.get("issues", [])
            print(f"  {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"   • {issue}")
            steps = steps_reflection.get("corrected_steps", steps)
            print(" Corrections applied.")

        # element_zones from Stage 5 are preserved in the final output so that
        # plan_generator_v2.py can read each step's "zones" field and know
        # exactly which canvas regions each element belongs to.
        result = {
            "summary": summary,
            "element_zones": zone_map_result.get("element_zones", []) if zone_map_result else [],
            "steps": steps,
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f" Process saved to: {output_path}")

        return result


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python process_generator2.py \"your painting description\"")
        sys.exit(1)

    description = sys.argv[1]
    gen = ProcessGenerator2(extract_model="qwen3:8b", steps_model="qwen3:30b-a3b")

    try:
        safe_name = description[:25].replace(" ", "_").replace("'", "")
        result = gen.generate(description, output_path=f"process2_{safe_name}.json")
        print("\n" + "═" * 60)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
