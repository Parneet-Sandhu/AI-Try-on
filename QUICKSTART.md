# Quick Start Guide

## What Changed?

We've migrated from custom neural networks to **Hugging Face's Stable Diffusion Inpainting** model. This provides:

✅ **Better Results** - Professional-quality image generation
✅ **Simpler Code** - No complex custom architectures  
✅ **No Tensor Issues** - Pre-tested, production-ready model
✅ **Active Support** - Maintained by the AI community

## File Changes Summary

### Before (Custom Models)
- Used GMM (Geometric Matching Module) for cloth warping
- Used U2NET for clothing segmentation
- Required custom training data
- Had tensor dimension mismatches

### After (Hugging Face)
- Uses Stable Diffusion Inpainting
- Simple mask generation for clothing region
- Pre-trained on millions of images
- Clean, maintainable code

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the App
```bash
streamlit run app.py
```

### 3. Use the App
1. Open http://localhost:8501
2. Upload your photo
3. Upload clothing item
4. Click "Generate Try-On"
5. Wait 30-60 seconds (longer on CPU)
6. Download the result!

## Model Details

- **Model ID**: `runwayml/stable-diffusion-inpainting`
- **Size**: ~5GB (downloaded on first run)
- **Type**: Latent diffusion for image-to-image
- **Input**: Image + Mask
- **Output**: Generated image with inpainted region

## Key Functions

### model.py

```python
# Load the pipeline
pipeline = load_models(opt)

# Generate try-on
result_image = generate_tryon(
    person_img,      # PIL Image
    cloth_img,       # PIL Image  
    pipeline,        # Diffusers pipeline
    opt              # Config object
)
```

### app.py

- **User uploads** → PIL Images
- **Button click** → `load_models()` + `generate_tryon()`
- **Display result** → Streamlit `st.image()`
- **Download** → Save using PIL

## Performance Tips

| Setting | Speed | Quality |
|---------|-------|---------|
| 50 steps (default) | 30-60s | High |
| 25 steps | 15-30s | Medium |
| 10 steps | 5-10s | Lower |

Edit in `model.py`:
```python
num_inference_steps=25  # Lower = faster
guidance_scale=7.5      # Higher = follow prompt more
```

## Troubleshooting

### "Module not found: diffusers"
```bash
pip install diffusers transformers accelerate
```

### Slow on CPU
- Use GPU: Install CUDA + PyTorch GPU version
- Reduce `num_inference_steps` to 25
- Use smaller input images

### Out of Memory
- Close other applications
- Reduce batch processing
- Use CPU offloading in diffusers

### Model won't download
- Check internet connection
- Set HuggingFace token: `huggingface-cli login`
- Ensure 6GB free space

## What's Legacy?

These files are from the old custom model approach (kept for reference):

- `cloth_mask.py` - U2NET clothing segmentation
- `network.py` - Custom GMM module  
- `networks/` - U2NET implementation
- `dataset.py` - Dataset utilities

These are **not used** by the current Streamlit app and can be removed.

## Next Steps

1. ✅ Run the app locally
2. Test with different photos
3. Customize prompts in `generate_tryon()`
4. Enhance mask generation for better results
5. Deploy to Streamlit Cloud or cloud provider

## Resources

- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [Stable Diffusion Model Card](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- [Streamlit Docs](https://docs.streamlit.io/)
- [PyTorch Docs](https://pytorch.org/docs/)

---

**Questions?** Check the full README.md for detailed documentation!
