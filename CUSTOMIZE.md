# Customization Guide

## Prompts

Edit the prompts in `model.py` to control image generation:

```python
# In generate_tryon() function, around line 60:

prompt = "a person wearing stylish clothing, high quality, professional photo"
negative_prompt = "blurry, low quality, distorted"
```

### Prompt Examples

**Professional Fashion:**
```python
prompt = "a person wearing elegant designer clothing, studio lighting, professional fashion photo"
negative_prompt = "blurry, low quality, amateur"
```

**Casual Wear:**
```python
prompt = "a person in casual comfortable clothing, natural lighting, lifestyle photo"
negative_prompt = "formal, bright, overly edited"
```

**Luxury Fashion:**
```python
prompt = "a person in luxury designer wear, high-end fashion, sophisticated styling"
negative_prompt = "cheap, casual, wrinkled"
```

**Streetwear:**
```python
prompt = "a person in trendy streetwear, urban style, cool fashion"
negative_prompt = "formal, boring, common"
```

## Model Parameters

### Inference Steps
Controls how many diffusion steps to run. More = better quality but slower.

```python
# Line 70 in model.py:
num_inference_steps=50  # Default: 50 (30-60 seconds)
```

**Recommended values:**
- **10**: Ultra-fast (~5-10s), lower quality
- **25**: Fast (~15-30s), good quality  ← Good balance
- **50**: Slow (~30-60s), high quality ← Default
- **75+**: Very slow (~60-90s), maximum quality

### Guidance Scale
How strongly the model follows the prompt. Higher = follows prompt more closely.

```python
# Line 71 in model.py:
guidance_scale=7.5  # Default
```

**Recommended values:**
- **3.0**: Creative, more variation
- **5.0**: Balanced (slightly more creative)
- **7.5**: Balanced (strict prompt following) ← Default
- **10.0**: Very strict prompt following
- **15.0**: Extreme - may distort image

## Image Size

Currently locked at 512x512 (good balance of quality and speed). To change:

```python
# Line 55-56 in model.py:
size = (512, 512)  # Change both values equally
```

**Recommended:**
- **256x256**: Ultra-fast but low quality
- **384x384**: Fast, decent quality
- **512x512**: Good balance ← Default
- **768x768**: Slow, high memory usage
- **1024x1024**: Very slow, requires lots of RAM

**Remember:** Size must be multiple of 8 (256, 384, 512, 640, 768, 896, 1024)

## Clothing Mask

The mask controls which part of the image to replace. Edit `create_clothing_mask()`:

### Current: Simple Rectangle Mask
```python
# Covers center region (torso + legs)
left = int(width * 0.15)      # Start 15% from left
top = int(height * 0.15)      # Start 15% from top
right = int(width * 0.85)     # End 85% from left
bottom = int(height * 0.85)   # End 85% from top
```

### Experiment: CircularMask
```python
from PIL import ImageDraw

def create_clothing_mask(image, cloth_image):
    width, height = image.size
    mask = Image.new('L', (width, height), 0)
    
    # Draw circle at center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        [center_x - radius, center_y - radius, 
         center_x + radius, center_y + radius],
        fill=255
    )
    return mask
```

### Experiment: Full Body Mask
```python
def create_clothing_mask(image, cloth_image):
    width, height = image.size
    mask = Image.new('L', (width, height), 0)
    
    # Mask entire body
    draw = ImageDraw.Draw(mask)
    draw.rectangle([0, 0, width, height], fill=255)
    
    return mask
```

## UI Customization

### Colors

Edit CSS in `app.py`:

```python
# Line 18-26 in app.py:
st.markdown("""
<style>
    .title {
        color: #d946a6;  # Change title color (hex code)
    }
    .subtitle {
        color: #666;    # Change subtitle color
    }
</style>
""", unsafe_allow_html=True)
```

**Color Codes:**
- Pink: `#d946a6` or `#ff69b4`
- Blue: `#0066cc` or `#0099ff`
- Purple: `#9933cc` or `#aa00ff`
- Green: `#00cc66` or `#00ff99`
- Black: `#000000`
- Gray: `#666666` or `#999999`

### Titles and Labels

```python
# Just edit the text in app.py:

st.markdown('<p class="title">👗 Virtual Closet – AI Styling Lab</p>', unsafe_allow_html=True)
# Change to:
st.markdown('<p class="title">✨ MyApp – Fashion AI</p>', unsafe_allow_html=True)

st.subheader("📸 Your Photo")
# Change to:
st.subheader("📸 Upload Your Picture")
```

### Emojis

Replace emojis in strings:
- 👗 Dress
- 👔 Suit
- 👕 Shirt
- 👖 Jeans
- 👠 Shoes
- 🎩 Hat
- ✨ Sparkle
- 🎉 Party
- ⬇️ Download
- And many more!

## Performance Tuning

### For CPU (No GPU)

```python
# model.py - use fewer steps:
num_inference_steps = 25  # Instead of 50

# Also reduce size:
size = (384, 384)  # Instead of 512x512

# Lower guidance:
guidance_scale = 5.0  # Instead of 7.5
```

### For GPU (CUDA)

```python
# model.py - can increase quality:
num_inference_steps = 75  # Higher quality

# Larger size:
size = (768, 768)  # Better quality

# Higher guidance:
guidance_scale = 10.0  # Stricter prompt following
```

### For Limited Memory (4GB RAM)

```python
# model.py:
num_inference_steps = 15
size = (256, 256)
guidance_scale = 5.0

# Also in model.py, add after line 15:
pipeline.enable_sequential_cpu_offload()
```

## Adding Features

### Example: Custom Style Transfer

```python
# In model.py, modify generate_tryon():

def generate_tryon(person_img, cloth_img, pipeline, opt=None, style="professional"):
    size = (512, 512)
    person_img = person_img.resize(size, Image.Resampling.LANCZOS)
    cloth_img = cloth_img.resize(size, Image.Resampling.LANCZOS)
    mask_img = create_clothing_mask(person_img, cloth_img)
    mask_img = mask_img.resize(size, Image.Resampling.LANCZOS)
    
    # Different prompts by style
    prompts = {
        "professional": "a person in professional business wear, office lighting",
        "casual": "a person in casual clothing, relaxed pose",
        "luxury": "a person in luxury designer clothing, high-end",
        "sporty": "a person in athletic wear, active pose",
    }
    
    prompt = prompts.get(style, prompts["professional"])
    
    # Rest of function...
```

Then in `app.py`:

```python
# Add style selection:
style = st.selectbox("Choose a style:", ["professional", "casual", "luxury", "sporty"])

# Pass to generate_tryon:
result = generate_tryon(person_img, cloth_img, pipeline, opt, style=style)
```

### Example: Multiple Clothing Items

```python
# Add second file uploader in app.py:
cloth_file_2 = st.file_uploader("Upload second clothing (optional)", type=["jpg", "png"])

# Process both:
if cloth_file_2:
    cloth_img_2 = Image.open(cloth_file_2).convert("RGB")
    result_2 = generate_tryon(person_img, cloth_img_2, pipeline, opt)
    
    # Display side by side:
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(result, caption="Outfit 1")
    with col_b:
        st.image(result_2, caption="Outfit 2")
```

## Deployment Customization

### For Streamlit Cloud

```python
# Add settings file: .streamlit/config.toml

[theme]
primaryColor = "#d946a6"
backgroundColor = "#fff5f9"
secondaryBackgroundColor = "#ffffff"
textColor = "#000000"

[server]
maxUploadSize = 200
```

### For Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Advanced: Custom Model

To use a different diffusion model:

```python
# model.py - change model ID:

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "different-model/stable-diffusion-inpainting",
    # or try other models:
    # "nitrosocke/Virtually-Trained-Diffusers-Inpainting"
    # "stabilityai/stable-diffusion-2-inpainting"
)
```

## Testing Customizations

After changes, test locally:

```bash
streamlit run app.py
```

Then try:
1. Different prompts
2. Different image sizes
3. Different inference steps

## Common Issues When Customizing

**Issue**: Image looks distorted
- **Fix**: Lower `guidance_scale` to 5.0

**Issue**: Doesn't follow prompt
- **Fix**: Increase `num_inference_steps` to 75

**Issue**: Out of memory
- **Fix**: Reduce `size` to (384, 384) or enable CPU offloading

**Issue**: Slow on CPU
- **Fix**: Reduce `num_inference_steps` to 15-25

---

**Need more help?** Check [README.md](README.md) or [QUICKSTART.md](QUICKSTART.md)!
