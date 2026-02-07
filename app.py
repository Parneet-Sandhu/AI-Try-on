import streamlit as st
from PIL import Image
from model import load_models, generate_tryon, generate_tryon_fast
from color_analysis import analyze_person
from theme_detection import detect_theme_heuristic, get_theme_outfit, get_theme_suggestions, suggest_outfit_for_user
import time

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
st.markdown('<p class="subtitle">AI Color Analysis + Theme Detection + Virtual Try-On</p>', unsafe_allow_html=True)

# Sidebar: Settings
with st.sidebar:
    st.header("⚙️ Settings")
    lightweight = st.checkbox("Lightweight Mode (instant, ~50MB)", value=False)
    show_color_analysis = st.checkbox("Show Color Analysis", value=True)
    show_theme_detection = st.checkbox("Show Theme Detection", value=True)
    selected_theme = st.selectbox(
        "Force Theme (optional)",
        ["Auto-detect"] + get_theme_suggestions(),
    )

st.divider()

# Create two columns for image uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Your Photo")
    person_file = st.file_uploader("Upload your full-body photo", type=["jpg", "png"])
    if person_file:
        person_img = Image.open(person_file).convert("RGB")
        st.image(person_img, caption="Your Photo", use_container_width=True)

        # ========== COLOR ANALYSIS ==========
        analysis = None
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

            st.write(f"**✨ Colors That Suit You:**")
            colors_display = ", ".join(rec["best_colors"][:5])
            st.success(colors_display)

        # ========== THEME DETECTION & OUTFIT SUGGESTION ==========
        if show_theme_detection:
            st.divider()
            st.markdown("### 🎭 Theme Detection & Outfit Suggestions")
            
            # Get recommended colors from analysis
            rec_colors = analysis["recommendations"]["best_colors"] if analysis else []
            
            # Determine theme
            if selected_theme != "Auto-detect":
                theme_final = selected_theme
            else:
                theme_final = detect_theme_heuristic(person_img)
            
            outfit = suggest_outfit_for_user(person_img, rec_colors, theme_final)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.metric("Suggested Theme", outfit["emoji"])
            with col_t2:
                st.write(f"**{outfit['theme_name']}**")
            
            col_outfit1, col_outfit2 = st.columns(2)
            with col_outfit1:
                st.write(f"**👔 Top:** {outfit['suggested_top']}")
            with col_outfit2:
                st.write(f"**👖 Bottom:** {outfit['suggested_bottom']}")
            
            st.info(f"**Styling Tip:** {outfit['styling_tips']}")
            st.success(f"**Colors for {outfit['theme_name']}:** {', '.join(outfit['suggested_colors'])}")

with col2:
    st.subheader("👔 Clothing Item")
    cloth_file = st.file_uploader("Upload clothing image", type=["jpg", "png"])
    if cloth_file:
        cloth_img = Image.open(cloth_file).convert("RGB")
        st.image(cloth_img, caption="Clothing Item", use_container_width=True)

st.divider()

# Generate button
if st.button("✨ Generate Try-On", type="primary", use_container_width=True):
    if person_file and cloth_file:
        try:
            pipeline = None
            if not lightweight:
                with st.spinner("Loading AI model (this may take a minute)..."):
                    # Load models with simple config
                    class Opt:
                        pass
                    opt = Opt()
                    pipeline = load_models(opt)
            
            with st.spinner("Generating your try-on image..."):
                person_img = Image.open(person_file).convert("RGB")
                cloth_img = Image.open(cloth_file).convert("RGB")
                
                # Determine expected time
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                time_estimate = "5-10 seconds (GPU)" if device == 'cuda' else "30-60 seconds (CPU)"
                
                st.info(f"⏳ This may take {time_estimate}. Please wait...")
                
                # Generate try-on (lightweight or ML)
                if lightweight:
                    st.info("Using Lightweight Mode (instant compositing)")
                    output = generate_tryon_fast(person_img, cloth_img)
                else:
                    st.info("Using AI ML Model (Stable Diffusion Inpainting - better quality but slower)")
                    output = generate_tryon(person_img, cloth_img, pipeline, opt)
                
                # Display result
                st.success("✓ Try-on image generated!")
                st.divider()
                st.subheader("🎉 Your Try-On Result")
                st.image(output, caption="Virtual Try-On Result", use_container_width=True)
                
                # Download button
                output_path = "tryon_result.png"
                output.save(output_path)
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Result",
                        data=file,
                        file_name="tryon_result.png",
                        mime="image/png"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Tips: Make sure your image is clear and shows full body. Try different poses or clothing!")
    
    else:
        st.warning("⚠️ Please upload both a photo and a clothing item")

st.divider()
st.markdown("""
### About this app
This is a virtual try-on application powered by AI diffusion models.
- **Model**: Stable Diffusion Inpainting (Hugging Face)
- **Purpose**: Help you visualize how different clothing would look on you!
- **Note**: Results are AI-generated and for visualization purposes

### ⏳ Why is generation slow?
**On CPU**: ~30-60 seconds (default, no GPU)  
**On GPU**: ~5-10 seconds (10x faster!)

**Tips to speed up:**
- **Use a GPU**: Install CUDA for PyTorch
- **First run takes longer**: Model downloads on first use (~3 min)
- **Subsequent runs are faster**: Model is cached

### 🚀 Want faster generation?
Install GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
This will make generation 10x faster!
""")

