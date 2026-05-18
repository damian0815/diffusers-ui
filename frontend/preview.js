/* frontend/preview.js — WebSocket live preview client */

const PING_INTERVAL_MS = 20_000;
const RECONNECT_DELAY_MS = 2_000;

/**
 * Connect to the preview WebSocket and call onFrame for each received frame.
 *
 * @param {function} onFrame - Called with {item_id, step, total_steps, image_b64}
 * @returns {WebSocket}
 */
function connectPreview(onFrame) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${location.host}/ws/preview`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onFrame(data);
    } catch (e) {
      // ignore malformed frames
    }
  };

  ws.onerror = () => {
    // Errors are handled by onclose
  };

  // Keep-alive ping
  const ping = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send('ping');
  }, PING_INTERVAL_MS);

  ws.onclose = () => {
    clearInterval(ping);
    // Reconnect after a short delay
    setTimeout(() => connectPreview(onFrame), RECONNECT_DELAY_MS);
  };

  return ws;
}

