# Layout Vocabulary

This file is loaded at runtime and injected into Stage 5.1 and Stage 6 prompts.
Edit here to update vocabulary for all stages at once.

---

## Canvas Zone Grid (3×3)

The canvas is divided into a 3×3 grid of named zones:

```
upper-left  | upper-center  | upper-right
mid-left    | mid-center    | mid-right
lower-left  | lower-center  | lower-right
```

When an element lists multiple zones, the union of those zones forms one bounding
rectangle on the canvas — that rectangle is what all fractions and alignments refer to.

---

## Field Meanings

### zones
Which grid cells the element occupies (array of zone name strings).
Multiple zones = element spans across them. Their union = one bounding rectangle.

### w_fraction
What fraction of the bounding rectangle's **width** the element fills:
- `"full"` — fills the entire width (element edge-to-edge horizontally)
- `"3/4"`  — fills 75 % of the width
- `"1/2"`  — fills 50 % of the width
- `"1/3"`  — fills 33 % of the width
- `"1/4"`  — fills 25 % of the width

### h_fraction
What fraction of the bounding rectangle's **height** the element fills (same values as w_fraction).

### h_align
**Only emit when w_fraction is NOT "full".**
Where to pin the element horizontally within the leftover space:
- `"left"`   — pushed flush to the left edge of the bounding rectangle
- `"center"` — centred horizontally inside the bounding rectangle
- `"right"`  — pushed flush to the right edge

When w_fraction = "full" there is no leftover horizontal space — do NOT emit h_align.

### v_align
**Only emit when h_fraction is NOT "full".**
Where to pin the element vertically within the leftover space:
- `"top"`    — pushed flush to the top edge of the bounding rectangle
- `"middle"` — centred vertically inside the bounding rectangle
- `"bottom"` — pushed flush to the bottom edge

When h_fraction = "full" there is no leftover vertical space — do NOT emit v_align.

---

## Worked Examples

**Sun in mid-center (small disc, top-centred):**
```json
{
  "zones": ["mid-center"],
  "w_fraction": "1/4", "h_fraction": "1/3",
  "h_align": "center", "v_align": "top"
}
```
→ Sun is 1/4 as wide and 1/3 as tall as the mid-center zone, pinned to the top-centre.
  Both fractions < full so both alignment fields are present.

**Mountains spanning full middle row (bottom half only):**
```json
{
  "zones": ["mid-left", "mid-center", "mid-right"],
  "w_fraction": "full", "h_fraction": "1/2",
  "v_align": "bottom"
}
```
→ Mountains fill the entire width of the three mid zones (w_fraction=full → no h_align),
  occupying the bottom half of that strip (v_align=bottom).

**Sky filling upper two rows completely:**
```json
{
  "zones": ["upper-left","upper-center","upper-right","mid-left","mid-center","mid-right"],
  "w_fraction": "full", "h_fraction": "full"
}
```
→ Sky fills all 6 zones edge-to-edge. Both fractions are full — neither h_align nor v_align is emitted.

**Person seated on the right (mid-height, half-width):**
```json
{
  "zones": ["mid-right"],
  "w_fraction": "1/2", "h_fraction": "3/4",
  "h_align": "center", "v_align": "middle"
}
```
→ Person occupies the centre of mid-right zone, half as wide and 3/4 as tall as the zone.
  Both fractions < full so both alignment fields are present.
