import streamlit as st
from PIL import Image
from model import load_models, generate_tryon_from_prompt
from color_analysis import analyze_person
from theme_detection import (
    detect_theme_heuristic,
    get_theme_outfit,
    get_theme_suggestions,
    get_season_suggestions,
    get_tops_options,
    get_bottoms_options,
    get_outfit_prompt,
    suggest_outfit_for_user,
    _pick_garments_for_theme_season,
)

# Page configuration
st.set_page_config(
    page_title="Virtual Closet – AI Styling Lab",
    page_icon="👗",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #fff5f9 0%, #fff 100%);
    }
    .title {
        color: #d946a6;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    .subtitle {
        color: #666;
        text-align: center;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<p class="title">👗 Virtual Closet – AI Styling Lab</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your photo → Color analysis → Theme & season → You pick top & bottom → AI generates you in your colors only</p>', unsafe_allow_html=True)

# Sidebar: Settings
with st.sidebar:
    st.header("⚙️ Settings")
    show_color_analysis = st.checkbox("Show Color Analysis", value=True)
    show_theme_detection = st.checkbox("Show Theme Detection", value=True)
    selected_theme = st.selectbox(
        "Theme",
        ["Auto-detect"] + get_theme_suggestions(),
        help="e.g. College, Office, Casual — outfit type will match this.",
    )
    selected_season = st.selectbox(
        "Season",
        get_season_suggestions(),
        help="Summer = lighter clothes; Winter = hoodies, layers, etc.",
    )

st.divider()

# Single column: person photo only (no cloth upload)
st.subheader("📸 Your Photo")
person_file = st.file_uploader("Upload your full-body photo", type=["jpg", "png"])
analysis = None
theme_final = None
rec_colors = []
outfit = None
selected_top = selected_bottom = None

if person_file:
    person_img = Image.open(person_file).convert("RGB")
    st.image(person_img, caption="Your Photo", use_container_width=True)

    # Theme for this session (used for suggestions and options)
    theme_final = selected_theme if selected_theme != "Auto-detect" else detect_theme_heuristic(person_img)

    # ========== COLOR ANALYSIS ==========
    if show_color_analysis:
        st.divider()
        st.markdown("### 🎨 AI Color Analysis")
        with st.spinner("Analyzing your skin tone and colors..."):
            analysis = analyze_person(person_img)
            skin_class = analysis["skin_tone_class"]
            temp = analysis["temperature"]
            rec = analysis["recommendations"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Skin Tone", skin_class)
        with col_b:
            st.metric("Temperature", temp)

        st.write("**✨ Colors That Suit You:**")
        colors_display = ", ".join(rec["best_colors"][:5])
        st.success(colors_display)
        rec_colors = rec["best_colors"]

    # ========== THEME + SEASON & SUGGESTION ==========
    if show_theme_detection:
        st.divider()
        st.markdown("### 🎭 Theme & Suggestion")
        outfit = suggest_outfit_for_user(person_img, rec_colors or [], theme_final)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Theme", outfit["emoji"])
        with col_t2:
            st.write(f"**{outfit['theme_name']}** · **Season:** {selected_season.title()}")
        col_outfit1, col_outfit2 = st.columns(2)
        with col_outfit1:
            st.write(f"**Suggested top:** {outfit['suggested_top']}")
        with col_outfit2:
            st.write(f"**Suggested bottom:** {outfit['suggested_bottom']}")
        st.info(f"**Styling Tip:** {outfit['styling_tips']}")

    # ========== CHOOSE YOUR OUTFIT (your colors only) ==========
    st.divider()
    st.markdown("### 👔 Choose Your Outfit")
    tops_list = get_tops_options(theme_final, selected_season)
    bottoms_list = get_bottoms_options(theme_final, selected_season)
    if outfit:
        sug_top, sug_bottom = outfit["suggested_top"], outfit["suggested_bottom"]
    else:
        sug_top, sug_bottom = _pick_garments_for_theme_season(theme_final, selected_season)
    idx_top = tops_list.index(sug_top) if sug_top in tops_list else 0
    idx_bottom = bottoms_list.index(sug_bottom) if sug_bottom in bottoms_list else 0

    selected_top = st.selectbox("Top wear", tops_list, index=idx_top, key="top_wear")
    selected_bottom = st.selectbox("Bottom wear", bottoms_list, index=idx_bottom, key="bottom_wear")

    color1 = (rec_colors[0] if rec_colors else "neutral")
    color2 = (rec_colors[1] if len(rec_colors) > 1 else rec_colors[0] if rec_colors else "neutral")
    st.success(f"**Outfit preview (from your color analysis only):** {color1} **{selected_top}** + {color2} **{selected_bottom}**")

st.divider()

# Generate button: only needs person photo; uses theme + season + color analysis
if st.button("✨ Generate Try-On", type="primary", use_container_width=True):
    if not person_file:
        st.warning("⚠️ Please upload your full-body photo first.")
    else:
        try:
            with st.spinner("Loading AI model (first time may take ~1 min)..."):
                class Opt:
                    pass
                opt = Opt()
                pipeline = load_models(opt)

            person_img = Image.open(person_file).convert("RGB")
            theme_for_prompt = theme_final if theme_final else detect_theme_heuristic(person_img)
            if not rec_colors:
                analysis_run = analyze_person(person_img)
                rec_colors = analysis_run["recommendations"]["best_colors"]
            colors_for_prompt = rec_colors if rec_colors else ["neutral", "gray"]
            # Use the exact top and bottom the user picked; colors only from color analysis
            prompt = get_outfit_prompt(
                theme_for_prompt, selected_season, colors_for_prompt,
                top_choice=selected_top or None, bottom_choice=selected_bottom or None,
            )

            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            time_estimate = "~30–60 s (GPU)" if device == "cuda" else "~1–3 min (CPU)"
            st.info(f"⏳ Generating outfit with your colors + {theme_for_prompt} + {selected_season}… {time_estimate}")

            with st.spinner("Generating your outfit..."):
                output = generate_tryon_from_prompt(person_img, prompt, pipeline, opt, num_steps=15, guidance_scale=7.0)

            st.success("✓ Try-on image generated!")
            st.divider()
            st.subheader("🎉 Your Try-On Result")
            st.image(output, caption="Virtual Try-On Result", use_container_width=True)
            output_path = "tryon_result.png"
            output.save(output_path)
            with open(output_path, "rb") as f:
                st.download_button(label="⬇️ Download Result", data=f, file_name="tryon_result.png", mime="image/png")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Make sure your image is clear and shows full body. Try theme/season and run again.")

st.divider()
st.markdown("""
### About this app
- **Flow**: Upload your photo → **color analysis** (skin tone + colors that suit you) → set **Theme** and **Season** → **choose your top and bottom** from the lists → generate. The outfit is always in **your analysis colors only** (no random or bad combinations).
- **Model**: Stable Diffusion Inpainting (Hugging Face, free).
- **Speed**: GPU ~30–60 s; CPU ~1–3 min per image. First run downloads the model once.
""")

