# image_to_plan.py
#
# Uses a Qwen VL model (via Hugging Face Inference, same client setup as
# Trial.py) to turn a reference image into an outline_generator.py-compatible
# plan JSON, and to apply correction rounds using the rendered output image.
#
# Correction is a two-pass critic/corrector split rather than one combined
# call: critique() only diagnoses errors in plain text (no JSON), and
# correct_plan() runs critique() itself (unless given precomputed text) then
# feeds that diagnosis, both images, and the existing JSON to a separate
# correction pass. Splitting diagnosis from fixing tends to catch more errors
# than asking a single call to spot-and-fix everything at once.
#
# Requires: HF_TOKEN environment variable set to a valid Hugging Face token.
#
# Usage:
#   python image_to_plan.py generate reference.png example.json --out plan.json
#   python image_to_plan.py critique generated.png reference.png
#   python image_to_plan.py correct plan.json generated.png reference.png --out plan.json

import base64
import json
import os
import re
import sys

from huggingface_hub import InferenceClient

MODEL = "Qwen/Qwen3-VL-32B-Instruct"
PROVIDER = "featherless-ai"

_PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "gpt_prompt.txt")
_RENDERER_FILE = os.path.join(os.path.dirname(__file__), "outline_generator.py")


def _load_prompts() -> tuple[str, str, str]:
    """Splits gpt_prompt.txt into (generation_prompt, correction_prompt, critic_prompt)."""
    text = open(_PROMPTS_FILE, "r", encoding="utf-8").read()
    correction_marker = "FOLLOW-UP CORRECTION PROMPT"
    critic_marker = "CRITIC PROMPT"

    gen_idx = text.index(correction_marker)
    critic_idx = text.index(critic_marker)

    generation_prompt = text[:gen_idx].strip()
    # Skip past each marker line and the "====" separator lines around it.
    correction_prompt = text[gen_idx:critic_idx].split("\n\n", 1)[1].strip()
    critic_prompt = text[critic_idx:].split("\n\n", 1)[1].strip()
    return generation_prompt, correction_prompt, critic_prompt


def _image_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext or "png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _extract_json(text: str) -> dict:
    """Strips markdown code fences if the model added them despite instructions,
    then parses the first complete JSON object found."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        debug_path = "last_raw_model_output.txt"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"JSON parse failed — raw model output saved to {debug_path} for inspection.")
        raise


def _call_model(content: list) -> str:
    client = InferenceClient(provider=PROVIDER, api_key=os.environ["HF_TOKEN"], timeout=120)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=16000,
    )
    return response.choices[0].message.content


def generate_plan(reference_image_path: str, example_json_path: str, out_path: str = "plan.json") -> dict:
    generation_prompt, _, _ = _load_prompts()
    renderer_source = open(_RENDERER_FILE, "r", encoding="utf-8").read()
    example_json = open(example_json_path, "r", encoding="utf-8").read()

    text_block = (
        f"{generation_prompt}\n\n"
        f"--- outline_generator.py (the renderer) ---\n{renderer_source}\n\n"
        f"--- example JSON (schema reference) ---\n{example_json}\n"
    )

    content = [
        {"type": "text", "text": text_block},
        {"type": "image_url", "image_url": {"url": _image_data_url(reference_image_path)}},
    ]

    raw = _call_model(content)
    plan = _extract_json(raw)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"Saved plan: {out_path}  ({len(plan.get('steps', []))} steps)")
    return plan


def critique(generated_image_path: str, reference_image_path: str) -> str:
    """Diagnosis-only pass: compares generated vs. reference image and returns
    a plain-text list of geometric errors, without touching any JSON. Kept
    separate from correct_plan() so diagnosis and fixing aren't asked of the
    model in the same breath."""
    _, _, critic_prompt = _load_prompts()

    content = [
        {"type": "text", "text": critic_prompt},
        {"type": "text", "text": "REFERENCE image:"},
        {"type": "image_url", "image_url": {"url": _image_data_url(reference_image_path)}},
        {"type": "text", "text": "GENERATED image:"},
        {"type": "image_url", "image_url": {"url": _image_data_url(generated_image_path)}},
    ]

    findings = _call_model(content).strip()
    print("--- CRITIC FINDINGS ---")
    print(findings)
    print("-----------------------")
    return findings


def correct_plan(
    plan_path: str,
    generated_image_path: str,
    reference_image_path: str,
    out_path: str | None = None,
    critique_text: str | None = None,
) -> dict:
    """Applies a correction round. If critique_text isn't passed in, runs
    critique() first so the corrector always works from an explicit written
    diagnosis rather than having to spot every error unaided from the images."""
    _, correction_prompt, _ = _load_prompts()
    existing_plan = open(plan_path, "r", encoding="utf-8").read()

    if critique_text is None:
        critique_text = critique(generated_image_path, reference_image_path)

    text_block = (
        f"{correction_prompt}\n\n"
        f"--- CRITIC FINDINGS (diagnosed by a separate review pass) ---\n{critique_text}\n\n"
        f"--- EXISTING JSON ---\n{existing_plan}\n"
    )

    content = [
        {"type": "text", "text": text_block},
        {"type": "text", "text": "Generated output image (from the current JSON):"},
        {"type": "image_url", "image_url": {"url": _image_data_url(generated_image_path)}},
        {"type": "text", "text": "Reference image (the target):"},
        {"type": "image_url", "image_url": {"url": _image_data_url(reference_image_path)}},
    ]

    raw = _call_model(content)
    plan = _extract_json(raw)

    out_path = out_path or plan_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(f"Saved corrected plan: {out_path}  ({len(plan.get('steps', []))} steps)")
    return plan


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] not in ("generate", "correct", "critique"):
        print("Usage:")
        print("  python image_to_plan.py generate reference.png example.json --out plan.json")
        print("  python image_to_plan.py correct plan.json generated.png reference.png --out plan.json")
        print("  python image_to_plan.py critique generated.png reference.png")
        sys.exit(1)

    mode = args[0]
    rest = args[1:]
    out = None
    positional = []
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--out":
            out = rest[i + 1]
            i += 2
        elif a.startswith("--out="):
            out = a.split("=", 1)[1]
            i += 1
        else:
            positional.append(a)
            i += 1

    if mode == "generate":
        ref, example = positional
        generate_plan(ref, example, out_path=out or "plan.json")
    elif mode == "critique":
        generated_img, reference_img = positional
        critique(generated_img, reference_img)
    else:
        plan_path, generated_img, reference_img = positional
        correct_plan(plan_path, generated_img, reference_img, out_path=out)
