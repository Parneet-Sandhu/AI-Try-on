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
