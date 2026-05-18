import io
import json
import base64

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Latent → RGB preview helpers
# (technique by @erucipe / @keturn / @torridgristle / @StAlKeR7779)
# ---------------------------------------------------------------------------

# SD 1.x per-channel RGB projection factors (updated for v1.5 by @torridgristle)
_SD1_LATENT_RGB_FACTORS = [
    #    R        G        B
    [0.3444, 0.1385, 0.0670],   # L1
    [0.1247, 0.4027, 0.1494],   # L2
    [-0.3192, 0.2513, 0.2103],  # L3
    [-0.1307, -0.1874, -0.7445],  # L4
]

# SDXL per-channel RGB projection factors (by @StAlKeR7779)
_SDXL_LATENT_RGB_FACTORS = [
    #    R        G        B
    [0.3816, 0.4930, 0.5320],
    [-0.3753, 0.1631, 0.1739],
    [0.1770, 0.3588, -0.2048],
    [-0.4350, -0.2644, -0.4289],
]

# Optional spatial-smoothing kernel applied after the SDXL projection
_SDXL_SMOOTH_MATRIX = [
    [0.0358, 0.0964, 0.0358],
    [0.0964, 0.4711, 0.0964],
    [0.0358, 0.0964, 0.0358],
]


def _sample_to_lowres_estimated_image(
    samples: torch.Tensor,
    latent_rgb_factors: torch.Tensor,
    smooth_matrix: torch.Tensor | None = None,
) -> Image.Image:
    """
    Project a latent sample to a low-resolution RGB preview via a linear
    colour-factor matrix, with an optional smoothing convolution pass.

    Args:
        samples:            shape (1, C, H, W) — single latent from step callback
        latent_rgb_factors: (C, 3) projection matrix
        smooth_matrix:      optional (3, 3) spatial smoothing kernel

    Returns:
        PIL Image (RGB).
    """
    latent_image = samples[0].permute(1, 2, 0) @ latent_rgb_factors  # H×W×3

    if smooth_matrix is not None:
        latent_image = latent_image.unsqueeze(0).permute(3, 0, 1, 2)  # 3×1×H×W
        latent_image = torch.nn.functional.conv2d(
            latent_image,
            smooth_matrix.reshape((1, 1, 3, 3)),
            padding=1,
        )
        latent_image = latent_image.permute(1, 2, 3, 0).squeeze(0)  # H×W×3

    latents_ubyte = (
        ((latent_image + 1) / 2)  # scale -1..1 → 0..1
        .clamp(0, 1)
        .mul(0xFF)
        .byte()
    ).cpu()

    return Image.fromarray(latents_ubyte.numpy())


def latents_to_preview_image(latents: torch.Tensor, *, is_sdxl: bool = False) -> bytes:
    """
    Convert a latent tensor to a low-quality preview JPEG without using the VAE.
    Uses a per-model RGB factor matrix for a coloured, recognisable approximation.

    Args:
        latents:  shape (1, C, H, W) — single-item latent from a step callback
        is_sdxl:  use SDXL projection factors + smoothing when True,
                  otherwise use SD 1.x factors.

    Returns:
        JPEG bytes suitable for WebSocket transmission.
    """
    with torch.no_grad():
        if is_sdxl:
            rgb_factors = torch.tensor(
                _SDXL_LATENT_RGB_FACTORS,
                dtype=latents.dtype,
                device=latents.device,
            )
            smooth = torch.tensor(
                _SDXL_SMOOTH_MATRIX,
                dtype=latents.dtype,
                device=latents.device,
            )
            img = _sample_to_lowres_estimated_image(latents, rgb_factors, smooth)
        else:
            rgb_factors = torch.tensor(
                _SD1_LATENT_RGB_FACTORS,
                dtype=latents.dtype,
                device=latents.device,
            )
            img = _sample_to_lowres_estimated_image(latents, rgb_factors)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return buf.getvalue()


class PreviewBroadcaster:
    """
    Maintains a set of active WebSocket connections and broadcasts
    preview frames to all of them.

    This class knows nothing about FastAPI — it just holds connections
    and sends bytes. The WebSocket router manages add/remove.
    """

    def __init__(self):
        self._connections: set = set()

    def add(self, ws) -> None:
        self._connections.add(ws)

    def remove(self, ws) -> None:
        self._connections.discard(ws)

    async def broadcast(
        self,
        item_id: int,
        step: int,
        total_steps: int,
        jpeg_bytes: bytes,
    ) -> None:
        """Broadcast a preview frame to all connected clients."""
        payload = json.dumps({
            "item_id": item_id,
            "step": step,
            "total_steps": total_steps,
            "image_b64": base64.b64encode(jpeg_bytes).decode(),
        })
        dead: set = set()
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._connections -= dead
