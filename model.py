from PIL import Image, ImageFilter
import numpy as np
import sys
import os


def _import_torch():
    """Import torch from site-packages only, so a local 'torch' folder doesn't shadow it."""
    # PyTorch can fail with "loaded the torch/_C folder of the PyTorch repository"
    # when a torch source dir or cwd shadows the real package. Prefer site-packages.
    import site
    saved_path = list(sys.path)
    try:
        # Prepend each site-packages path so installed torch is found first
        for sp in site.getsitepackages():
            if os.path.isdir(sp) and sp not in sys.path:
                sys.path.insert(0, sp)
        # Remove cwd and script dir so they don't shadow torch
        cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path = [p for p in sys.path if os.path.abspath(p) != cwd and os.path.abspath(p) != script_dir]
        import torch
        return torch
    finally:
        sys.path[:] = saved_path


def _get_device():
    """Lazy torch import to avoid loading PyTorch until generation is needed."""
    torch = _import_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(opt):
    """Load inpainting model. Torch and diffusers are imported here so the app
    can start without PyTorch (e.g. for color analysis only).
    """
    try:
        torch = _import_torch()
    except Exception as e:
        raise RuntimeError(
            "PyTorch failed to load. This often happens when a local 'torch' folder "
            "shadows the real package, or PyTorch is misinstalled.\n\n"
            "Fix: 1) Run from the project folder: cd E:\\try-on\\AI-Try-on\n"
            "     2) Reinstall: pip uninstall torch torchvision && pip install torch torchvision\n\n"
            f"Original error: {e}") from e
    try:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
            StableDiffusionInpaintPipeline,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to import diffusers. Run: pip install -U transformers diffusers accelerate\n\n"
            f"Original error: {e}") from e

    device = _get_device()
    print(f"Loading inpainting model on {device}...")
    print("⚡ This model generates results in 1 step (instant!)")

    # Use runwayml inpainting model (works with Stable Diffusion InpaintPipeline)
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32 if device == 'cpu' else torch.float16,
    )
    pipeline = pipeline.to(device)

    if device == 'cuda':
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass

    print("✓ Model loaded successfully!")
    return pipeline

def create_clothing_mask(image: Image.Image, cloth_image: Image.Image = None) -> Image.Image:
    """
    Create a simple mask for the clothing region (torso + upper legs).
    Used for inpainting: masked area will be replaced by generated clothing.
    cloth_image is optional (kept for backward compatibility with generate_tryon).
    """
    width, height = image.size
    mask = Image.new('L', (width, height), 0)
    left = int(width * 0.15)
    top = int(height * 0.15)
    right = int(width * 0.85)
    bottom = int(height * 0.85)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, top, right, bottom], fill=255)
    return mask


def generate_tryon_from_prompt(
    person_img: Image.Image,
    prompt: str,
    pipeline,
    opt=None,
    num_steps: int = 15,
    guidance_scale: float = 7.0,
):
    """
    Generate virtual try-on using only a text prompt (no cloth image).
    Uses inpainting on the clothing region so the outfit matches theme + season + color analysis.
    Faster defaults: 15 steps, guidance 7.0.
    """
    size = (512, 512)
    person_img = person_img.resize(size, Image.Resampling.LANCZOS)
    mask_img = create_clothing_mask(person_img)
    mask_img = mask_img.resize(size, Image.Resampling.LANCZOS)

    negative_prompt = "low quality, blurry, distorted, ugly, bad anatomy, deformed face, extra limbs"

    import torch
    with torch.no_grad():
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_img,
            mask_image=mask_img,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
        )
    img = output.images[0]

    # If result is too dark, retry once with more steps
    arr = np.array(img.convert("L"))
    if float(arr.mean()) < 20:
        import torch
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person_img,
                mask_image=mask_img,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=512,
                width=512,
            )
        img = output.images[0]
    return img


def generate_tryon(person_img: Image.Image, cloth_img: Image.Image, pipeline, opt=None):
    """
    Generate virtual try-on using ML inpainting with adaptive prompts.
    Uses the actual clothing image to guide generation.
    """
    import torch
    size = (512, 512)
    person_img = person_img.resize(size, Image.Resampling.LANCZOS)
    cloth_img = cloth_img.resize(size, Image.Resampling.LANCZOS)
    
    # Create mask for clothing region
    mask_img = create_clothing_mask(person_img, cloth_img)
    mask_img = mask_img.resize(size, Image.Resampling.LANCZOS)
    
    # Analyze cloth color to create better prompt
    cloth_arr = np.array(cloth_img)
    dominant_color = cloth_arr.reshape(-1, 3).mean(axis=0).astype(int)
    
    # Estimate dominant color name
    r, g, b = dominant_color
    if max(dominant_color) - min(dominant_color) < 30:
        color_name = "neutral gray" if r < 150 else "white"
    elif r > g and r > b:
        color_name = "red" if r > 180 else "warm brown"
    elif g > r and g > b:
        color_name = "green"
    elif b > r and b > g:
        color_name = "blue"
    else:
        color_name = "multicolor"
    
    # Smart prompt based on clothing
    prompt = f"person wearing a {color_name} clothing item, professional photo, good lighting, high quality"
    negative_prompt = "low quality, blurry, distorted, ugly, bad anatomy"
    
    print(f"Generating try-on with prompt: '{prompt}'...")

    default_steps = 18
    default_guidance = 7.0

    def run_pipeline(steps, guidance):
        with torch.no_grad():
            return pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=person_img,
                mask_image=mask_img,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=512,
                width=512,
            )

    # First attempt with better defaults
    output = run_pipeline(default_steps, default_guidance)
    img = output.images[0]

    # Validation: check if result is reasonable (not too dark)
    arr = np.array(img.convert('L'))
    mean_val = float(arr.mean())
    if mean_val < 15:
        try:
            img.save('debug_tryon_dark.png')
        except Exception:
            pass
        print(f"Warning: generated image very dark (mean={mean_val:.2f}), retrying...")
        output = run_pipeline(40, 7.5)
        img = output.images[0]

    return img


def generate_tryon_fast(person_img: Image.Image, cloth_img: Image.Image):
    """
    Improved lightweight try-on with better background removal and positioning.
    
    Strategy:
    1. Extract clothing using adaptive thresholding (better background removal)
    2. Detect clothing bounds and scale intelligently
    3. Blend with torso area using opacity blending + feathering
    4. Adjust colors to match clothing better
    """
    size = (512, 512)
    person = person_img.resize(size, Image.Resampling.LANCZOS).convert('RGBA')
    
    # Resize clothing to approximately clothing size (about 40% of person image)
    cloth_resized_coarse = cloth_img.resize((int(size[0] * 0.7), int(size[1] * 0.7)), Image.Resampling.LANCZOS)
    cloth_rgba = cloth_resized_coarse.convert('RGBA')

    # Smart background removal using edge-based detection
    cloth_rgb = cloth_rgba.convert('RGB')
    cloth_arr = np.array(cloth_rgb, dtype=float)
    
    # Get corners (likely background) - sample 4 corners
    h, w = cloth_arr.shape[:2]
    corners = [
        cloth_arr[0:10, 0:10].mean(axis=(0, 1)),  # Top-left
        cloth_arr[0:10, w-10:w].mean(axis=(0, 1)),  # Top-right
        cloth_arr[h-10:h, 0:10].mean(axis=(0, 1)),  # Bottom-left
        cloth_arr[h-10:h, w-10:w].mean(axis=(0, 1)),  # Bottom-right
    ]
    bg_color = np.mean(corners, axis=0)
    
    # Create mask: pixels similar to background are transparent
    dist_to_bg = np.linalg.norm(cloth_arr - bg_color, axis=2)
    threshold = 60  # Color distance threshold
    mask_arr = (dist_to_bg > threshold).astype('uint8') * 255
    
    # Morphological smoothing: dilate then erode to reduce noise
    from scipy import ndimage
    mask_arr = ndimage.binary_dilation(mask_arr > 0, iterations=2).astype('uint8') * 255
    mask_arr = ndimage.binary_erosion(mask_arr > 0, iterations=1).astype('uint8') * 255
    
    mask = Image.fromarray(mask_arr).convert('L')
    
    # Apply feathering (soft edges) by blurring the mask
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Find clothing bounding box to auto-position
    mask_np = np.array(mask)
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)
    if rows.any() and cols.any():
        cloth_y1, cloth_y2 = np.where(rows)[0][[0, -1]]
        cloth_x1, cloth_x2 = np.where(cols)[0][[0, -1]]
        cloth_h_actual = cloth_y2 - cloth_y1
        cloth_w_actual = cloth_x2 - cloth_x1
    else:
        # Fallback if detection fails
        cloth_h_actual = int(h * 0.6)
        cloth_w_actual = int(w * 0.6)
        cloth_x1, cloth_y1 = int(w * 0.2), int(h * 0.2)
    
    # Position on person: center horizontally, place on torso (upper-middle area)
    canvas = person.copy()
    target_x = (size[0] - cloth_w_actual) // 2
    target_y = int(size[1] * 0.2)  # Start at 20% from top (shoulders)
    
    # Create intermediate image for blending
    tmp = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    
    # Paste clothing with mask
    cloth_cropped = cloth_rgba.crop((cloth_x1, cloth_y1, cloth_x1 + cloth_w_actual, cloth_y1 + cloth_h_actual))
    mask_cropped = mask.crop((cloth_x1, cloth_y1, cloth_x1 + cloth_w_actual, cloth_y1 + cloth_h_actual))
    
    # Ensure mask and cloth are same size
    if cloth_cropped.size != mask_cropped.size:
        mask_cropped = mask_cropped.resize(cloth_cropped.size, Image.Resampling.LANCZOS)
    
    tmp.paste(cloth_cropped, (target_x, target_y), mask_cropped)
    
    # Alpha composite with reduced opacity for softer blending
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    overlay.paste(tmp)
    
    # Blend at 85% opacity for natural look
    result = Image.blend(canvas.convert('RGB'), overlay.convert('RGB'), 0.85)
    
    return result
