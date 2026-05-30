import io
import sys
import json
import base64
import asyncio
import logging
import threading
from pathlib import Path

import torch
from PIL import Image

from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TAESD submodule path — project-root/taesd/
# ---------------------------------------------------------------------------

_TAESD_DIR = Path(__file__).parent.parent.parent / "taesd"
if str(_TAESD_DIR) not in sys.path:
    sys.path.insert(0, str(_TAESD_DIR))


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


def _latents_to_approx_image(latents: torch.Tensor, *, is_sdxl: bool = False) -> bytes:
    """
    Linear RGB-projection approximation (no neural decoder required).
    Fast but lower quality — used as fallback when TAESD is unavailable.
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


# ---------------------------------------------------------------------------
# TAESD decoder — lazy-loaded, one instance per (variant, device) pair
# ---------------------------------------------------------------------------

_taesd_cache: dict = {}       # (variant, device_str) → nn.Sequential
_taesd_lock = threading.Lock()


def _load_taesd_decoder(variant: str, device: str):
    """
    Load (or return a cached) TAESD decoder for *variant* on *device*.

    variant: "sd1" → taesd_decoder.pth (4-channel SD 1.x / SDXL-compatible)
             "sdxl" → taesdxl_decoder.pth
    """
    key = (variant, device)
    if key in _taesd_cache:
        return _taesd_cache[key]

    with _taesd_lock:
        if key in _taesd_cache:          # double-checked
            return _taesd_cache[key]

        weight_name = "taesdxl_decoder.pth" if variant == "sdxl" else "taesd_decoder.pth"
        weight_path = _TAESD_DIR / weight_name

        try:
            import taesd as _taesd_module  # noqa: PLC0415 (imported inside fn intentionally)
            dec = _taesd_module.Decoder().to(device).requires_grad_(False)
            dec.load_state_dict(
                torch.load(weight_path, map_location=device, weights_only=True)
            )
            dec.eval()
            _taesd_cache[key] = dec
            logger.info("Loaded TAESD decoder (%s) from %s", variant, weight_path)
            return dec
        except Exception as exc:
            logger.warning(
                "Could not load TAESD decoder from %s: %s — falling back to approx preview",
                weight_path, exc,
            )
            return None


def _latents_to_taesd_image(latents: torch.Tensor, *, is_sdxl: bool = False) -> bytes | None:
    """
    Decode latents with TAESD and return JPEG bytes, or return None if the
    decoder could not be loaded (caller falls back to approx).
    """
    variant = "sdxl" if is_sdxl else "sd1"
    device = str(latents.device)
    dec = _load_taesd_decoder(variant, device)
    if dec is None:
        return None

    with torch.no_grad():
        # Decoder outputs roughly [0, 1] RGB; clamp for safety.
        rgb = dec(latents.float()).clamp(0, 1)   # (1, 3, H, W)

    rgb_np = rgb[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
    img = Image.fromarray(rgb_np, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def latents_to_preview_image(latents: torch.Tensor, *, is_sdxl: bool = False) -> bytes:
    """
    Convert a latent tensor to a preview JPEG.

    Dispatches to TAESD (higher quality) or the linear-approximation method
    depending on ``settings.preview_method``.

    Args:
        latents:  shape (1, C, H, W) — single-item latent from a step callback
        is_sdxl:  select SDXL weights / projection factors when True

    Returns:
        JPEG bytes suitable for WebSocket transmission.
    """
    if settings.preview_method == "taesd":
        result = _latents_to_taesd_image(latents, is_sdxl=is_sdxl)
        if result is not None:
            return result
        # Fall through to approx if TAESD is unavailable

    return _latents_to_approx_image(latents, is_sdxl=is_sdxl)


class PreviewBroadcaster:
    """
    Maintains a set of active WebSocket connections and broadcasts
    preview frames to all of them.

    Sends are fire-and-forget (asyncio.create_task).  If a connection still has
    a send in-flight when the next frame arrives the new frame is silently
    dropped for that client — this prevents slow consumers from stalling the
    worker while keeping latency near-zero for fast consumers.
    """

    def __init__(self):
        self._connections: set = set()
        self._sending: set = set()    # connections with a send currently in-flight

    def add(self, ws) -> None:
        self._connections.add(ws)

    def remove(self, ws) -> None:
        self._connections.discard(ws)
        self._sending.discard(ws)

    async def _deliver(self, ws, payload: str) -> None:
        """Single-connection fire-and-forget send."""
        try:
            await ws.send_text(payload)
        except Exception:
            # Connection died — evict it.
            self._connections.discard(ws)
        finally:
            self._sending.discard(ws)

    async def broadcast(
        self,
        item_id: int,
        step: int,
        total_steps: int,
        jpeg_bytes: bytes,
    ) -> None:
        """Broadcast a preview frame to all connected clients."""
        if not self._connections:
            return
        payload = json.dumps({
            "type": "preview",
            "item_id": item_id,
            "step": step,
            "total_steps": total_steps,
            "image_b64": base64.b64encode(jpeg_bytes).decode(),
        })
        for ws in list(self._connections):
            if ws in self._sending:
                continue          # client hasn't drained last frame — drop this one
            self._sending.add(ws)
            asyncio.create_task(self._deliver(ws, payload))

    async def broadcast_event(self, event_type: str, data: dict) -> None:
        """Broadcast a domain event to all connected clients (fire-and-forget).

        Unlike preview frames, events are never dropped — each connected client
        gets its own send task regardless of in-flight state.
        """
        if not self._connections:
            return
        payload = json.dumps({"type": "event", "event": event_type, "data": data})
        for ws in list(self._connections):
            asyncio.create_task(self._deliver(ws, payload))

