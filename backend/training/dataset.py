import math
import random
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from backend.config import settings
from backend.prompt_utils import expand_prompt
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
        num_vectors: int = 1,
        augment_max_rotation_degrees: float = 30,
    ):
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.size = size
        self.class_prompt = class_prompt
        self.num_vectors = num_vectors
        self.augment_max_rotation_degrees = augment_max_rotation_degrees
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

    @staticmethod
    def _compute_max_rotation_angle(img_w: int, img_h: int, box_w: int, box_h: int, max_degrees: float) -> float:
        """
        Compute the maximum safe rotation angle (in degrees) such that a
        center axis-aligned crop of `box_w x box_h` after rotating the
        `img_w x img_h` image by that angle contains no border fill.

        The safe-rotation constraints are:
          box_w * cos(θ) + box_h * sin(θ) ≤ img_w
          box_h * cos(θ) + box_w * sin(θ) ≤ img_h

        Each is of the form A·cos(θ) + B·sin(θ) ≤ C, which rewrites as
        R·sin(θ + φ) ≤ C with R = √(A²+B²), φ = atan2(A, B).

        Returns the smaller of the two bound angles, capped at max_degrees.
        If both constraints are satisfied for all θ in [0, max_degrees],
        returns max_degrees.
        """
        if box_w > img_w or box_h > img_h:
            return 0.0

        bounds = []

        for a, b, c in [(box_w, box_h, img_w), (box_h, box_w, img_h)]:
            r = math.sqrt(a * a + b * b)
            if c >= r:
                continue
            phi = math.atan2(a, b)
            bound = math.asin(c / r) - phi
            bounds.append(bound)

        if not bounds:
            return max_degrees

        result = min(bounds)
        result_degrees = math.degrees(result)
        return max(0.0, min(result_degrees, max_degrees))

    @staticmethod
    def _apply_rotation(image: Image.Image, angle_degrees: float) -> Image.Image:
        """
        Rotate a PIL image by `angle_degrees` (positive = counterclockwise)
        around its center, using fill color (0, 0, 0) for exposed regions.
        """
        return image.rotate(angle_degrees, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

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

        if self.augment_max_rotation_degrees > 0:
            img_w, img_h = image.size
            max_angle = self._compute_max_rotation_angle(
                img_w, img_h, self.size, self.size, self.augment_max_rotation_degrees,
            )
            if max_angle > 0:
                angle = random.uniform(-max_angle, max_angle)
                image = self._apply_rotation(image, angle)

        transform = self._build_transform(tw, th)

        suffix = f" {self.class_prompt}" if self.class_prompt else ""
        text = random.choice(self.templates).format(self.placeholder_token) + suffix
        bare_name = self.placeholder_token.strip("<>")
        text = expand_prompt(text, {bare_name: self.num_vectors})
        #print("tokenzining text:", text)
        tokenization_output = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {"pixel_values": transform(image), "attention_mask": tokenization_output.attention_mask[0], "input_ids": tokenization_output.input_ids[0]}
