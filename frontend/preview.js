/* frontend/preview.js — WebSocket live preview client */

/**
 * Connect to the preview WebSocket and call onFrame for each frame that
 * should be rendered.
 *
 * Frames are coalesced via requestAnimationFrame: if multiple messages arrive
 * between two paint opportunities only the most recent one is delivered to
 * onFrame.  This keeps the main thread free and prevents the preview from
 * falling behind when the backend is faster than the display refresh rate.
 *
 * @param {function} onFrame - Called with {item_id, step, total_steps, image_b64}
 */
function connectPreview(onFrame) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';

  let latestFrame = null;   // most recent parsed frame waiting to be rendered
  let rafPending  = false;  // true while a rAF callback is already scheduled

  function scheduleRender() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      if (latestFrame !== null) {
        onFrame(latestFrame);
        latestFrame = null;
      }
    });
  }

  function connect() {
    const ws = new WebSocket(`${protocol}//${location.host}/ws/preview`);

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'event') {
          // Domain event (gallery/image operations) — dispatch immediately, no coalescing.
          window.dispatchEvent(new CustomEvent('ws-gallery-event', { detail: msg }));
        } else {
          // Preview frame (type === 'preview' or legacy without type field).
          latestFrame = msg; // overwrite — keep only latest
          scheduleRender();
        }
      } catch (e) { /* ignore malformed frames */ }
    };

    ws.onerror = () => { /* handled by onclose */ };

    // Keep-alive ping
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 20_000);

    ws.onclose = () => {
      clearInterval(ping);
      setTimeout(connect, 2_000); // auto-reconnect
    };
  }

  connect();
}
