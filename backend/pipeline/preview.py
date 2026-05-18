import io
import json
import base64

import torch
from PIL import Image

# Standard SD/SDXL latent scaling constant
VAE_SCALING_FACTOR = 0.18215


def latents_to_preview_image(latents: torch.Tensor) -> bytes:
    """
    Convert a latent tensor to a low-quality preview JPEG without using the VAE.
    Uses a channel-average approximation: fast but blurry. Good enough for progress.

    Args:
        latents: shape (1, C, H, W) — single-item latent from a step callback

    Returns:
        JPEG bytes suitable for WebSocket transmission.
    """
    with torch.no_grad():
        # Bring into roughly [0, 1] range using mean across channels
        preview = (latents[0].float() / VAE_SCALING_FACTOR).mean(dim=0)
        preview = (preview - preview.min()) / (preview.max() - preview.min() + 1e-8)
        preview_np = (preview.cpu().numpy() * 255).clip(0, 255).astype("uint8")

    img = Image.fromarray(preview_np, mode="L").convert("RGB")
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

