import streamlit as st
from PIL import Image
from model import load_models, generate_tryon_from_plan
from color_analysis import analyze_person
from outfit_planner import get_plan
from theme_detection import (
    detect_theme_heuristic,
    get_theme_suggestions,
    get_season_suggestions,
    get_tops_options,
    get_bottoms_options,
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
st.markdown('<p class="subtitle">Stage 1: Color analysis → Stage 2: AI outfit plan → Stage 3: Controlled try-on</p>', unsafe_allow_html=True)

# Cache pipeline so we load once (next runs 30–60s instead of 10 min)
class Opt:
    pass

@st.cache_resource
def get_pipeline():
    return load_models(Opt())

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_color_analysis = st.checkbox("Show Color Analysis", value=True)
    show_theme_detection = st.checkbox("Show Theme Detection", value=True)
    use_ai_stylist = st.checkbox("Use AI Stylist (Stage 2)", value=True, help="Small HF model for plan; off = rule-based only.")
    selected_theme = st.selectbox("Theme", ["Auto-detect"] + get_theme_suggestions())
    selected_season = st.selectbox("Season", get_season_suggestions())

st.divider()

# Photo
st.subheader("📸 Your Photo")
person_file = st.file_uploader("Upload your full-body photo", type=["jpg", "png"])
analysis = None
theme_final = None
elite_colors = []
forbidden_colors = []
outfit = None
selected_top = selected_bottom = None
plan = None

if person_file:
    person_img = Image.open(person_file).convert("RGB")
    st.image(person_img, caption="Your Photo", use_container_width=True)
    theme_final = selected_theme if selected_theme != "Auto-detect" else detect_theme_heuristic(person_img)

    # ========== STAGE 1: COLOR ANALYSIS ==========
    with st.spinner("Stage 1: Analyzing your colors..."):
        analysis = analyze_person(person_img)
    elite_colors = analysis.get("elite_colors") or analysis.get("recommendations", {}).get("best_colors", [])
    forbidden_colors = analysis.get("forbidden_colors") or analysis.get("recommendations", {}).get("avoid", [])

    if show_color_analysis:
        st.divider()
        st.markdown("### 🎨 Stage 1 – Color Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Skin Tone", analysis["skin_tone_class"])
        with col_b:
            st.metric("Undertone", analysis["undertone"].title())
        st.success(f"**✅ Colors that suit you:** {', '.join(elite_colors[:5])}")
        if forbidden_colors:
            st.caption(f"❌ Avoid: {', '.join(forbidden_colors)}")

    # ========== THEME + SUGGESTION ==========
    rec_colors = elite_colors
    if show_theme_detection:
        st.divider()
        st.markdown("### 🎭 Theme & Suggestion")
        outfit = suggest_outfit_for_user(person_img, rec_colors or [], theme_final)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Theme", outfit["emoji"])
        with col_t2:
            st.write(f"**{outfit['theme_name']}** · **Season:** {selected_season.title()}")
        st.info(f"**Suggested:** {outfit['suggested_top']} + {outfit['suggested_bottom']}")

    # ========== CHOOSE TOP & BOTTOM ==========
    st.divider()
    st.markdown("### 👔 Choose Your Outfit")
    tops_list = get_tops_options(theme_final, selected_season)
    bottoms_list = get_bottoms_options(theme_final, selected_season)
    sug_top, sug_bottom = (outfit["suggested_top"], outfit["suggested_bottom"]) if outfit else _pick_garments_for_theme_season(theme_final, selected_season)
    idx_top = tops_list.index(sug_top) if sug_top in tops_list else 0
    idx_bottom = bottoms_list.index(sug_bottom) if sug_bottom in bottoms_list else 0

    selected_top = st.selectbox("Top wear", tops_list, index=idx_top, key="top_wear")
    selected_bottom = st.selectbox("Bottom wear", bottoms_list, index=idx_bottom, key="bottom_wear")

    # ========== STAGE 2: AI OUTFIT PLAN ==========
    with st.spinner("Stage 2: Planning your outfit..."):
        plan = get_plan(
            theme_final, selected_season, selected_top, selected_bottom,
            analysis, use_ai_stylist=use_ai_stylist,
        )
    st.success(
        f"**Outfit plan:** {plan['top_color']} **{plan['top_type']}** + {plan['bottom_color']} **{plan['bottom_type']}**  \n"
        f"_{plan.get('reason', '')}_"
    )

st.divider()

# ========== STAGE 3: GENERATE (controlled try-on) ==========
if st.button("✨ Generate Try-On", type="primary", use_container_width=True):
    if not person_file:
        st.warning("⚠️ Please upload your full-body photo first.")
    elif not plan:
        st.warning("⚠️ Upload a photo and wait for the outfit plan above.")
    else:
        try:
            pipeline = get_pipeline()
            person_img = Image.open(person_file).convert("RGB")

            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"⏳ Stage 3: Generating (exact plan: {plan['top_color']} {plan['top_type']} + {plan['bottom_color']} {plan['bottom_type']})… {('~30–60 s' if device == 'cuda' else '~1–3 min')}")

            with st.spinner("Generating your outfit..."):
                output = generate_tryon_from_plan(
                    person_img, plan, forbidden_colors, selected_season,
                    pipeline, Opt(), num_steps=18, guidance_scale=5.5,
                )

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
            st.info("💡 Make sure your image is clear and shows full body.")

st.divider()
st.markdown("""
### Pipeline (3 stages)
- **Stage 1:** Color analysis → your undertone + elite colors + forbidden colors (no heavy model).
- **Stage 2:** AI Stylist → exact plan (top_color, bottom_color, top_type, bottom_type) from your choices + analysis; optional small HF text model.
- **Stage 3:** Controlled diffusion try-on → prompt uses plan only; low CFG + negative prompt so colors stay correct. Pipeline cached so next runs are fast.
""")

