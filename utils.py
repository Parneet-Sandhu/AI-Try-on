from PIL import Image
import torch
from torchvision import transforms

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return img

def pil_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)

def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor)
