/* frontend/persist.js — localStorage helpers */
window.Persist = (() => {
  const _timers = {};

  function get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (_) { return fallback; }
  }

  function set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) {}
  }

  function setDebounced(key, value, ms = 250) {
    clearTimeout(_timers[key]);
    _timers[key] = setTimeout(() => { _timers[key] = null; set(key, value); }, ms);
  }

  function flush(key, value) {
    clearTimeout(_timers[key]);
    _timers[key] = null;
    set(key, value);
  }

  return { get, set, setDebounced, flush };
})();
