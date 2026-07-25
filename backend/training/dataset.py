import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from backend.config import settings

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a painting in the style of cool {}",
    "a painting in the style of {}",
    "a painting in the style of bright {}",
    "a painting in the style of clean {}",
    "a painting in the style of {}",
    "a nice painting in the style of {}",
    "a painting in the style of good {}",
    "a painting in the style of small {}",
]


class TextualInversionDataset(Dataset):
    """
    Loads images from a gallery directory (settings.images_dir).
    Applies random horizontal flip, center crop, and resize.
    Returns {pixel_values, input_ids} pairs.
    """

    def __init__(
        self,
        image_names: list[str],
        tokenizer,
        placeholder_token: str,
        size: int = 512,
        repeats: int = 100,
        learnable_property: str = "object",
        center_crop: bool = False,
    ):
        self.image_paths = [settings.images_dir / name for name in image_names]
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.size = size
        self.center_crop = center_crop
        self._length = len(self.image_paths) * repeats
        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i % len(self.image_paths)]).convert("RGB")
        text = random.choice(self.templates).format(self.placeholder_token)
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": self.transform(image), "input_ids": input_ids}
