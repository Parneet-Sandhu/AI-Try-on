"""
Color analysis module: skin tone detection and color recommendations.
Lightweight, no ML models required — uses PIL + numpy only.
"""

import numpy as np
from PIL import Image


def extract_skin_tone(image: Image.Image, top_k: int = 100) -> tuple:
    """
    Extract dominant skin tone from face region (upper-middle area of image).
    Returns (R, G, B) tuple.
    """
    # Convert to RGB if needed
    img = image.convert("RGB")
    arr = np.array(img)

    # Sample from face region (rough: upper-middle 40% width, upper 50% height)
    h, w = arr.shape[:2]
    x1, x2 = int(w * 0.2), int(w * 0.8)
    y1, y2 = 0, int(h * 0.4)
    face_region = arr[y1:y2, x1:x2]

    # Flatten and find dominant colors (avoid extremes like white/black)
    pixels = face_region.reshape(-1, 3).astype(float)
    
    # Filter out very bright (>220) and very dark (<50) pixels
    brightness = pixels.mean(axis=1)
    valid_pixels = pixels[(brightness > 50) & (brightness < 220)]

    if len(valid_pixels) == 0:
        # Fallback: use all pixels
        valid_pixels = pixels

    # Find centroid (mean color) as skin tone
    dominant = valid_pixels.mean(axis=0).astype(int)
    return tuple(dominant)


def classify_skin_tone(rgb: tuple) -> str:
    """
    Classify skin tone as Fair/Medium/Deep based on luminance.
    Returns one of: 'Fair', 'Medium', 'Deep'
    """
    r, g, b = rgb
    # ITU-R BT.709 luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    if luminance > 180:
        return "Fair"
    elif luminance > 100:
        return "Medium"
    else:
        return "Deep"


def get_color_temperature(rgb: tuple) -> str:
    """
    Classify skin tone as Warm/Cool/Neutral based on red vs. red+green balance.
    Returns one of: 'Warm', 'Cool', 'Neutral'
    """
    r, g, b = rgb
    
    # Warm tones have more red relative to green
    # Cool tones have more blue relative to red
    red_green_diff = r - g
    red_blue_diff = r - b
    
    if red_green_diff > 10:
        return "Warm"
    elif red_blue_diff < -10:
        return "Cool"
    else:
        return "Neutral"


def recommend_colors(skin_tone_class: str, temperature: str) -> dict:
    """
    Recommend flattering colors based on skin tone classification and temperature.
    Returns dict with color names and hex values.
    """
    recommendations = {
        ("Fair", "Warm"): {
            "name": "Warm Fair Skin",
            "best_colors": ["Gold", "Warm Orange", "Coral", "Warm Red", "Cream", "Warm Brown"],
            "hex": ["#FFD700", "#FF8C00", "#FF7F50", "#DC143C", "#FFFDD0", "#8B4513"],
            "avoid": ["Cool Blue", "Silver", "Black"],
        },
        ("Fair", "Cool"): {
            "name": "Cool Fair Skin",
            "best_colors": ["Silver", "Cool Blue", "Jewel Tones", "White", "Rose", "Cool Purple"],
            "hex": ["#C0C0C0", "#4169E1", "#000080", "#FFFFFF", "#FF007F", "#800080"],
            "avoid": ["Gold", "Warm Orange", "Rust"],
        },
        ("Fair", "Neutral"): {
            "name": "Neutral Fair Skin",
            "best_colors": ["Navy", "Emerald", "Pink", "Taupe", "Ivory", "Wine Red"],
            "hex": ["#000080", "#50C878", "#FFC0CB", "#B38B6D", "#FFFFF0", "#722F37"],
            "avoid": ["Neon colors"],
        },
        ("Medium", "Warm"): {
            "name": "Warm Medium Skin",
            "best_colors": ["Olive", "Terracotta", "Warm Burgundy", "Warm Yellow", "Rust", "Burnt Orange"],
            "hex": ["#808000", "#E2725B", "#800020", "#FFD700", "#B7410E", "#CC5500"],
            "avoid": ["Pale pink", "Cool gray"],
        },
        ("Medium", "Cool"): {
            "name": "Cool Medium Skin",
            "best_colors": ["Deep Blue", "Teal", "Purple", "Magenta", "Bright White", "Violet"],
            "hex": ["#00008B", "#008080", "#800080", "#FF00FF", "#FFFFFF", "#EE82EE"],
            "avoid": ["Warm orange", "Gold"],
        },
        ("Medium", "Neutral"): {
            "name": "Neutral Medium Skin",
            "best_colors": ["Caramel", "Jade", "Plum", "Warm White", "Rich Brown", "Deep Red"],
            "hex": ["#A0522D", "#00A86B", "#DDA0DD", "#F5F5DC", "#654321", "#8B0000"],
            "avoid": ["Neon colors"],
        },
        ("Deep", "Warm"): {
            "name": "Warm Deep Skin",
            "best_colors": ["Gold", "Deep Orange", "Warm Red", "Mustard", "Bronze", "Terracotta"],
            "hex": ["#FFD700", "#FF6347", "#FF0000", "#FFDB58", "#CD7F32", "#CD5C5C"],
            "avoid": ["Pale colors", "Cool gray"],
        },
        ("Deep", "Cool"): {
            "name": "Cool Deep Skin",
            "best_colors": ["Jewel Tones", "Royal Blue", "Emerald", "Deep Purple", "Bright White", "Silver"],
            "hex": ["#000080", "#4169E1", "#50C878", "#800080", "#FFFFFF", "#C0C0C0"],
            "avoid": ["Pale colors", "Warm orange"],
        },
        ("Deep", "Neutral"): {
            "name": "Neutral Deep Skin",
            "best_colors": ["Chocolate Brown", "Deep Teal", "Sapphire", "Cream", "Charcoal", "Wine"],
            "hex": ["#3E2C1E", "#367588", "#0F52BA", "#FFFDD0", "#36454F", "#722F37"],
            "avoid": ["Very pale colors"],
        },
    }

    key = (skin_tone_class, temperature)
    return recommendations.get(key, {
        "name": "Default",
        "best_colors": ["Navy", "Cream", "Charcoal"],
        "hex": ["#000080", "#FFFDD0", "#36454F"],
        "avoid": ["Neon"],
    })


def get_contrast_level(skin_class: str) -> str:
    """Soft vs high contrast (for styling). Fair/Deep = high, Medium = soft."""
    return "high" if skin_class in ("Fair", "Deep") else "soft"


def analyze_person(image: Image.Image) -> dict:
    """
    Stage 1: Color analysis. Extract skin tone, undertone, elite colors, forbidden colors.
    Returns dict with everything the AI Stylist (Stage 2) needs.
    """
    skin_rgb = extract_skin_tone(image)
    skin_class = classify_skin_tone(skin_rgb)
    temperature = get_color_temperature(skin_rgb)
    recommendations = recommend_colors(skin_class, temperature)

    elite = list(recommendations.get("best_colors", [])[:6])
    forbidden = list(recommendations.get("avoid", []))

    return {
        "skin_tone_rgb": skin_rgb,
        "skin_tone_class": skin_class,
        "temperature": temperature,
        "undertone": temperature.lower(),
        "contrast": get_contrast_level(skin_class),
        "elite_colors": elite,
        "forbidden_colors": forbidden,
        "recommendations": recommendations,
    }
