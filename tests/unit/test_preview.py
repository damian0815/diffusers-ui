"""Unit tests for backend/pipeline/preview.py"""
import asyncio
import io
import pytest
import torch
from PIL import Image
from backend.pipeline.preview import latents_to_preview_image, PreviewBroadcaster


def test_returns_bytes():
    latents = torch.randn(1, 4, 64, 80)
    result = latents_to_preview_image(latents)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_output_is_valid_jpeg():
    latents = torch.randn(1, 4, 64, 80)
    result = latents_to_preview_image(latents)
    img = Image.open(io.BytesIO(result))
    assert img.format == "JPEG"


def test_output_is_rgb():
    latents = torch.randn(1, 4, 64, 80)
    result = latents_to_preview_image(latents)
    img = Image.open(io.BytesIO(result))
    assert img.mode == "RGB"


def test_handles_all_zeros():
    latents = torch.zeros(1, 4, 64, 80)
    result = latents_to_preview_image(latents)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_handles_large_values():
    latents = torch.randn(1, 4, 64, 80) * 1000
    result = latents_to_preview_image(latents)
    assert isinstance(result, bytes)


@pytest.mark.asyncio
async def test_broadcaster_add_and_remove():
    broadcaster = PreviewBroadcaster()

    class FakeWs:
        messages = []
        async def send_text(self, msg):
            self.messages.append(msg)

    ws = FakeWs()
    broadcaster.add(ws)
    assert ws in broadcaster._connections
    broadcaster.remove(ws)
    assert ws not in broadcaster._connections


@pytest.mark.asyncio
async def test_broadcaster_sends_to_all_connections():
    broadcaster = PreviewBroadcaster()

    class FakeWs:
        def __init__(self):
            self.messages = []
        async def send_text(self, msg):
            self.messages.append(msg)

    ws1, ws2 = FakeWs(), FakeWs()
    broadcaster.add(ws1)
    broadcaster.add(ws2)
    await broadcaster.broadcast(item_id=1, step=5, total_steps=20, jpeg_bytes=b"fake")
    await asyncio.sleep(0)
    assert len(ws1.messages) == 1
    assert len(ws2.messages) == 1


@pytest.mark.asyncio
async def test_broadcaster_removes_dead_connections():
    broadcaster = PreviewBroadcaster()

    class DeadWs:
        async def send_text(self, msg):
            raise RuntimeError("connection closed")

    ws = DeadWs()
    broadcaster.add(ws)
    await broadcaster.broadcast(item_id=1, step=1, total_steps=10, jpeg_bytes=b"x")
    await asyncio.sleep(0)
    assert ws not in broadcaster._connections


