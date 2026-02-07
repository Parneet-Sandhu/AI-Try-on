"""
Theme detection and clothing recommendations.
Detects the context/theme a photo should be styled for and suggests outfits.
"""

import numpy as np
from PIL import Image


# Theme profiles: theme -> recommended (tops, bottoms, styles, colors)
THEME_PROFILES = {
    "office": {
        "name": "Office / Professional",
        "tops": ["Blazer", "Button-up Shirt", "Structured Blouse", "Turtleneck", "Cardigan"],
        "bottoms": ["Pencil Skirt", "Trousers", "Dress Pants", "Tailored Shorts"],
        "styles": ["Formal", "Structured", "Neutral ColorPalette"],
        "colors": ["Navy", "Black", "White", "Gray", "Camel", "Burgundy"],
        "emoji": "💼"
    },
    "casual": {
        "name": "Casual / Everyday",
        "tops": ["T-Shirt", "Sweater", "Sweatshirt", "Casual Blouse", "Polo"],
        "bottoms": ["Jeans", "Casual Pants", "Shorts", "Leggings", "Skirt"],
        "styles": ["Relaxed", "Comfortable", "Mixed Patterns"],
        "colors": ["Blue", "Gray", "White", "Earth Tones", "Pastels"],
        "emoji": "👕"
    },
    "party": {
        "name": "Party / Evening",
        "tops": ["Sequin Top", "Silk Blouse", "Crop Top", "Off-Shoulder Top", "Sparkly Camisole"],
        "bottoms": ["Dress Pants", "Metallic Skirt", "Leather Pants", "Sequin Shorts"],
        "styles": ["Glamorous", "Shiny Fabrics", "Bold Colors", "Statement Pieces"],
        "colors": ["Gold", "Silver", "Black", "Deep Red", "Jewel Tones"],
        "emoji": "✨"
    },
    "college": {
        "name": "College / Student",
        "tops": ["Hoodie", "Graphic T-Shirt", "Crop Top", "Casual Tee", "Oversized Sweatshirt"],
        "bottoms": ["Jeans", "Sweatpants", "Shorts", "Leggings", "Cargo Pants"],
        "styles": ["Casual", "Trendy", "Comfortable", "Youthful"],
        "colors": ["Black", "White", "Gray", "Pastels", "Bright Colors"],
        "emoji": "🎓"
    },
    "sports": {
        "name": "Sports / Gym",
        "tops": ["Sports Bra", "Tank Top", "Performance Top", "Hoodie", "T-Shirt"],
        "bottoms": ["Yoga Pants", "Leggings", "Shorts", "Sweatpants"],
        "styles": ["Athletic", "Breathable", "Stretchy", "Minimal"],
        "colors": ["Black", "White", "Gray", "Neon", "Earth Tones"],
        "emoji": "💪"
    },
    "date": {
        "name": "Date / Romantic",
        "tops": ["Silk Top", "Delicate Blouse", "Jewelry-Friendly Top", "V-Neck Top"],
        "bottoms": ["Skirt", "Dress Pants", "Tailored Shorts", "Midi Pants"],
        "styles": ["Elegant", "Flattering Silhouette", "Subtle Shine", "Romantic"],
        "colors": ["Jewel Tones", "Soft Pink", "Deep Red", "Navy", "Black"],
        "emoji": "💕"
    },
    "nature": {
        "name": "Outdoor / Nature",
        "tops": ["Flannel Shirt", "Windbreaker", "T-Shirt", "Sweater", "Jacket"],
        "bottoms": ["Jeans", "Cargo Pants", "Hiking Pants", "Shorts"],
        "styles": ["Practical", "Layerable", "Durable", "Earth Tones"],
        "colors": ["Olive", "Khaki", "Brown", "Gray", "Green"],
        "emoji": "🏕️"
    },
    "beach": {
        "name": "Beach / Summer",
        "tops": ["Bikini Top", "Tank Top", "Crop Top", "Linen Shirt", "Cover-up"],
        "bottoms": ["Shorts", "Beach Skirt", "Swim Shorts", "Linen Pants"],
        "styles": ["Light", "Breathable", "Colorful", "Fun"],
        "colors": ["Bright Colors", "Pastels", "White", "Ocean Blue", "Coral"],
        "emoji": "🏖️"
    },
    "supermarket": {
        "name": "Supermarket / Errands",
        "tops": ["T-Shirt", "Casual Top", "Sweater", "Simple Blouse"],
        "bottoms": ["Jeans", "Casual Pants", "Leggings", "Shorts"],
        "styles": ["Practical", "Easy to Move In", "Minimalist"],
        "colors": ["Neutral", "Blue", "Gray", "Black", "White"],
        "emoji": "🛒"
    },
}

# Season adjusts garment types (lighter vs heavier)
SEASON_PROFILES = {
    "summer": {
        "name": "Summer",
        "top_emphasis": ["Tank Top", "T-Shirt", "Linen Shirt", "Crop Top", "Sleeveless Blouse", "Polo"],
        "bottom_emphasis": ["Shorts", "Light Pants", "Skirt", "Linen Pants", "Capris"],
        "style_words": "light, breathable, summer",
    },
    "winter": {
        "name": "Winter",
        "top_emphasis": ["Hoodie", "Sweater", "Jacket", "Coat", "Cardigan", "Turtleneck", "Sweatshirt"],
        "bottom_emphasis": ["Jeans", "Trousers", "Sweatpants", "Warm Pants", "Leggings"],
        "style_words": "warm, layered, winter",
    },
    "spring": {
        "name": "Spring",
        "top_emphasis": ["Light Sweater", "Blouse", "T-Shirt", "Cardigan", "Jacket"],
        "bottom_emphasis": ["Jeans", "Chinos", "Skirt", "Light Pants"],
        "style_words": "light layers, spring",
    },
    "fall": {
        "name": "Fall / Autumn",
        "top_emphasis": ["Sweater", "Flannel", "Cardigan", "Long Sleeve Tee", "Jacket"],
        "bottom_emphasis": ["Jeans", "Cargo Pants", "Trousers", "Skirt"],
        "style_words": "cozy, autumn",
    },
}


def get_season_suggestions():
    """Return list of available seasons."""
    return list(SEASON_PROFILES.keys())


def detect_theme_heuristic(image: Image.Image) -> str:
    """
    Simple heuristic-based theme detection.
    Could be upgraded to use a lightweight image classifier.
    For now, returns a high-level guess based on image brightness/saturation.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=float)

    # Calculate average brightness
    brightness = arr.mean()

    # Calculate saturation (rough)
    max_c = arr.max(axis=2)
    min_c = arr.min(axis=2)
    saturation = ((max_c - min_c) / (max_c + 0.001)).mean()

    # Heuristic:
    # High brightness + high saturation -> outdoor/beach
    # High brightness + low saturation -> casual/everyday
    # Low brightness + high saturation -> party/evening
    # Otherwise -> office (neutral/professional)

    if brightness > 150 and saturation > 0.3:
        return "beach"
    elif brightness > 140 and saturation < 0.2:
        return "casual"
    elif brightness < 100 and saturation > 0.4:
        return "party"
    else:
        return "office"  # Default to professional


def get_theme_suggestions() -> list:
    """
    Return list of all available themes.
    """
    return list(THEME_PROFILES.keys())


def get_theme_outfit(theme: str) -> dict:
    """
    Get outfit recommendations for a given theme.
    """
    return THEME_PROFILES.get(theme, THEME_PROFILES["casual"])


def get_season_outfit(season: str) -> dict:
    """Get season-based garment emphasis."""
    return SEASON_PROFILES.get(season, SEASON_PROFILES["summer"])


def _pick_garments_for_theme_season(theme: str, season: str) -> tuple:
    """Pick one top and one bottom from theme + season. Returns (top, bottom)."""
    theme_outfit = get_theme_outfit(theme)
    season_outfit = get_season_outfit(season)
    theme_tops = [t.lower() for t in theme_outfit["tops"]]
    theme_bottoms = [b.lower() for b in theme_outfit["bottoms"]]
    # Prefer season items that appear in theme; else theme; else season
    top = None
    for t in season_outfit["top_emphasis"]:
        if t.lower() in theme_tops:
            top = t
            break
    if top is None:
        top = theme_outfit["tops"][0] if theme_outfit["tops"] else season_outfit["top_emphasis"][0]
    bottom = None
    for b in season_outfit["bottom_emphasis"]:
        if b.lower() in theme_bottoms:
            bottom = b
            break
    if bottom is None:
        bottom = theme_outfit["bottoms"][0] if theme_outfit["bottoms"] else season_outfit["bottom_emphasis"][0]
    return top, bottom


def get_outfit_prompt(
    theme: str,
    season: str,
    recommended_colors: list,
    theme_name: str = None,
) -> str:
    """
    Build a stable-diffusion style prompt for outfit generation.
    Uses theme + season for garment types and recommended_colors (from color analysis)
    for color words so the generated clothes suit the person.
    Returns a string like: "person wearing olive hoodie and blue jeans, college style, winter, ..."
    """
    top, bottom = _pick_garments_for_theme_season(theme, season)
    season_profile = get_season_outfit(season)
    theme_profile = get_theme_outfit(theme)
    name = theme_name or theme_profile.get("name", theme)

    # Use first 2 colors from analysis for top and bottom (e.g. olive, blue)
    color1 = recommended_colors[0] if recommended_colors else "neutral"
    color2 = recommended_colors[1] if len(recommended_colors) > 1 else recommended_colors[0] if recommended_colors else "neutral"
    # Normalize color names for prompt (lowercase, no spaces for consistency)
    c1 = color1.lower().replace(" ", "")
    c2 = color2.lower().replace(" ", "")

    prompt = (
        f"person wearing {color1} {top} and {color2} {bottom}, "
        f"{name} style, {season_profile['style_words']}, "
        "professional photo, good lighting, high quality, full body"
    )
    return prompt


def suggest_outfit_for_user(
    image: Image.Image, recommended_colors: list, theme: str = None
) -> dict:
    """
    Suggest a complete outfit for the user:
    - Theme (auto-detected if not provided)
    - Top recommendation
    - Bottom recommendation
    - Color suggestions
    - Overall styling advice
    """
    if theme is None:
        theme = detect_theme_heuristic(image)

    theme_outfit = get_theme_outfit(theme)

    # Pick first recommendation from each category
    suggested_top = theme_outfit["tops"][0] if theme_outfit["tops"] else "Blouse"
    suggested_bottom = theme_outfit["bottoms"][0] if theme_outfit["bottoms"] else "Pants"

    # Filter theme colors to intersect with user's recommended colors
    theme_colors = theme_outfit["colors"]
    good_colors = [c for c in theme_colors if any(
        c.lower() in user_c.lower() or user_c.lower() in c.lower()
        for user_c in recommended_colors
    )]
    if not good_colors:
        good_colors = theme_colors[:3]  # Fallback

    return {
        "theme": theme,
        "theme_name": theme_outfit["name"],
        "emoji": theme_outfit["emoji"],
        "suggested_top": suggested_top,
        "suggested_bottom": suggested_bottom,
        "suggested_colors": good_colors,
        "styling_tips": f"Go for {theme_outfit['styles'][0]} and {theme_outfit['styles'][1] if len(theme_outfit['styles']) > 1 else 'elegant'} pieces.",
    }
