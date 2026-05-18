from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/preview")
async def preview_ws(websocket: WebSocket):
    """
    WebSocket endpoint for live latent preview frames.
    The worker broadcasts JPEG frames as base64-encoded JSON payloads.
    Clients send 'ping' messages to keep the connection alive.
    """
    broadcaster = websocket.app.state.broadcaster
    await websocket.accept()
    broadcaster.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive; client sends pings
    except WebSocketDisconnect:
        broadcaster.remove(websocket)
    except Exception:
        broadcaster.remove(websocket)


