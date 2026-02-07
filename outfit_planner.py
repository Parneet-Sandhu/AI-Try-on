"""
Stage 2: AI Outfit Planner (the stylist).
Takes color analysis + occasion + season + user's top/bottom choices.
Outputs a strict plan: top_color, bottom_color, top_type, bottom_type, reason.
Uses a small Hugging Face text model when available; else rule-based (deterministic).
NO image generation — only JSON plan for Stage 3.
"""

import json
import re


def _rule_based_plan(
    occasion: str,
    season: str,
    top_type: str,
    bottom_type: str,
    elite_colors: list,
    forbidden_colors: list,
    undertone: str,
) -> dict:
    """Deterministic plan from elite colors only. No bad combinations."""
    if not elite_colors:
        elite_colors = ["navy", "ivory"]
    if len(elite_colors) == 1:
        top_color = bottom_color = elite_colors[0].lower()
    else:
        top_color = elite_colors[0].lower()
        bottom_color = elite_colors[1].lower()

    return {
        "top_color": top_color,
        "bottom_color": bottom_color,
        "top_type": top_type,
        "bottom_type": bottom_type,
        "reason": f"Matches your {undertone} undertone and {occasion} for {season}. Colors from your palette only.",
    }


def _parse_json_from_text(text: str) -> dict | None:
    """Extract a JSON object from model output (may be wrapped in markdown)."""
    text = text.strip()
    # Find {...}
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def get_plan(
    occasion: str,
    season: str,
    top_type: str,
    bottom_type: str,
    analysis: dict,
    use_ai_stylist: bool = True,
) -> dict:
    """
    Stage 2: Produce outfit plan (colors + types) from color analysis and user choices.
    Returns dict with: top_color, bottom_color, top_type, bottom_type, reason.
    Colors are always from analysis elite list; forbidden colors are never used.
    """
    elite = analysis.get("elite_colors") or analysis.get("recommendations", {}).get("best_colors", [])[:6]
    forbidden = analysis.get("forbidden_colors") or analysis.get("recommendations", {}).get("avoid", [])
    undertone = analysis.get("undertone") or analysis.get("temperature", "neutral").lower()

    if use_ai_stylist:
        try:
            from transformers import pipeline
            pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=128,
                truncation=True,
            )
            prompt = (
                f"User has {undertone} undertone. Best colors: {', '.join(elite[:4])}. "
                f"Avoid: {', '.join(forbidden[:3])}. Occasion: {occasion}, Season: {season}. "
                f"Top: {top_type}, Bottom: {bottom_type}. "
                "Output JSON only with keys: top_color, bottom_color, top_type, bottom_type, reason. "
                "Use only colors from best colors list. One short reason."
            )
            out = pipe(prompt, max_new_tokens=120, do_sample=False)
            text = out[0].get("generated_text", "")
            plan = _parse_json_from_text(text)
            if plan and "top_color" in plan and "bottom_color" in plan:
                # Enforce: colors must be from elite; use full elite label (e.g. "Warm Burgundy")
                elite_lower = [c.lower() for c in elite]
                def pick_elite_color(model_color: str, prefer_index: int) -> str:
                    mc = str(model_color).lower()
                    for i, e in enumerate(elite_lower):
                        if mc in e or e in mc:
                            return elite[i]
                    return elite[prefer_index] if prefer_index < len(elite) else elite[0]

                plan["top_color"] = pick_elite_color(plan.get("top_color", elite[0]), 0)
                plan["bottom_color"] = pick_elite_color(plan.get("bottom_color", elite[1] if len(elite) > 1 else elite[0]), 1)
                plan["top_type"] = plan.get("top_type") or top_type
                plan["bottom_type"] = plan.get("bottom_type") or bottom_type
                return plan
        except Exception:
            pass

    return _rule_based_plan(
        occasion, season, top_type, bottom_type,
        elite, forbidden, undertone,
    )
