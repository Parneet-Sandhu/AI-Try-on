# Virtual Closet – AI Styling Lab

## Overview

A virtual try-on application powered by AI that allows users to upload their photo and clothing items to see how the clothes would look on them. Built with Streamlit and Hugging Face's Stable Diffusion Inpainting model.

## Features

✨ **AI-Powered Try-On**: Uses Stable Diffusion Inpainting to generate realistic try-on images
📸 **Simple Upload**: Drag-and-drop interface for photos and clothing
💾 **Download Results**: Save generated try-on images
🎨 **Clean UI**: Inspired by the movie Clueless - minimalist design with pink accents

## Tech Stack

### Frontend
- **Streamlit**: Web UI framework for rapid prototyping
- **Pillow**: Image processing library

### Backend
- **Python 3.12**
- **PyTorch 2.0+**: Deep learning framework
- **Hugging Face Diffusers**: Pre-trained generative models
- **Transformers**: NLP models for text conditioning

## Installation

### Prerequisites
- Python 3.10+
- ~6GB disk space for model download (on first run)
- GPU recommended (runs on CPU but slower)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Parneet-Sandhu/AI-Try-on.git
cd AI-Try-on
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then:
1. Open browser to `http://localhost:8501`
2. Upload your full-body photo
3. Upload a clothing item image
4. Click "Generate Try-On"
5. Download the result!

### First Run
⏳ The first run will download the Stable Diffusion Inpainting model (~5GB). This may take 2-3 minutes depending on your connection.

## How It Works

### Model: Stable Diffusion Inpainting
- **Source**: [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- **Type**: Latent diffusion model optimized for inpainting tasks
- **Process**:
  1. User uploads full-body photo and clothing image
  2. App creates a mask for the clothing region
  3. Model generates new image with clothing inpainted
  4. Result maintains user's pose and identity

### Clothing Mask
The app creates a center-focused mask that tells the model which areas to modify. Currently uses a simple geometric mask - can be enhanced with clothing segmentation models.

## Project Structure

```
AI-Try-on/
├── app.py                 # Main Streamlit application
├── model.py              # Model loading and inference
├── cloth_mask.py         # Clothing mask utilities (legacy)
├── network.py            # Network architectures (legacy)
├── dataset.py            # Dataset utilities (legacy)
├── utils.py              # General utilities
├── requirements.txt      # Python dependencies
├── networks/             # U2NET network module (legacy)
│   ├── __init__.py
│   └── u2net.py
└── README.md             # This file
```

## Key Files

### app.py
Main Streamlit interface:
- File upload widgets
- Model loading UI
- Result display
- Download functionality

### model.py
Core AI pipeline:
- `load_models()`: Downloads and initializes Stable Diffusion pipeline
- `create_clothing_mask()`: Generates region mask for inpainting
- `generate_tryon()`: Runs the inpainting model

## Performance

| Component | Time |
|-----------|------|
| Model Download (first run) | ~2-3 min |
| Model Load | ~30 sec |
| Image Generation | ~30-60 sec (CPU), ~5-10 sec (GPU) |

## Limitations

- Works best with clear, full-body photos
- Clothing should show clearly in upload
- Generated images are AI-created (not photorealistic)
- Model works on 512x512 resolution internally

## Future Enhancements

- [ ] Real-time pose detection
- [ ] Multiple clothing item support (mix & match)
- [ ] Cloth segmentation for precise masking
- [ ] Cloud deployment (Streamlit Cloud / HuggingFace Spaces)
- [ ] Batch generation for multiple outfits
- [ ] Style transfer options
- [ ] Wardrobe management system

## Troubleshooting

### Model won't download
- Check internet connection
- Try setting HuggingFace token: `huggingface-cli login`
- Ensure 6GB free disk space

### Slow generation
- GPU is recommended (50-100x faster)
- Install CUDA: https://pytorch.org/get-started/locally/
- Reduce `num_inference_steps` in model.py (lower quality, faster)

### Memory issues
- Run on machine with 8GB+ RAM
- Close other applications
- Use `accelerate` library for distributed inference

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
```bash
streamlit run app.py --logger.level=debug
```
Then deploy via https://streamlit.io/cloud

### Docker
```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
```

## Credits

- **Model**: [Runway ML's Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- **Framework**: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- **UI**: [Streamlit](https://streamlit.io/)
- **Inspiration**: Clueless (1995) 👗

## License

MIT License - feel free to use for personal projects

## Contributing

Contributions welcome! Areas for improvement:
- Better clothing segmentation
- More sophisticated mask generation
- Performance optimizations
- UI/UX enhancements

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers)
3. Open an issue on GitHub

---

**Made with ❤️ for fashion tech enthusiasts**
