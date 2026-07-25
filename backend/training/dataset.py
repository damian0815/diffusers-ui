import random
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from backend.config import settings
from backend.training.aspects import get_aspect_buckets, compute_resolution_options

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
    Applies aspect-ratio-aware resize (via aspects.compute_resolution_options) then
    random crop, random horizontal flip.
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
        class_prompt: str = "",
    ):
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.size = size
        self.class_prompt = class_prompt
        self._length = len(image_names) * repeats
        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )

        buckets = get_aspect_buckets(size)
        aspects_per_resolution = {size: buckets}

        self._entries: list[tuple[Path, int, int]] = []
        for name in image_names:
            path = settings.images_dir / name
            try:
                with Image.open(path) as img:
                    w, h = img.size
                result = compute_resolution_options(w, h, aspects_per_resolution)
                target_wh = result[size].target_wh
                bw, bh = target_wh
                if bw >= bh:
                    tw, th = round(size * bw / bh), size
                else:
                    tw, th = size, round(size * bh / bw)
            except Exception:
                tw, th = size, size
            self._entries.append((path, tw, th))

    def _build_transform(self, target_w: int, target_h: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((target_h, target_w),
                               interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        path, tw, th = self._entries[i % len(self._entries)]
        image = Image.open(path).convert("RGB")
        transform = self._build_transform(tw, th)

        suffix = f" {self.class_prompt}" if self.class_prompt else ""
        text = random.choice(self.templates).format(self.placeholder_token) + suffix
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": transform(image), "input_ids": input_ids}
