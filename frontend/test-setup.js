/* frontend/test-setup.js — vitest environment bootstrap.
   jsdom provides a real Storage on its own window, but Node 22+ ships an
   experimental global `localStorage` getter that returns undefined (warning:
   "--localstorage-file was not provided"), and vitest skips copying jsdom's
   localStorage onto globalThis because that key already exists. Re-expose it. */
if (globalThis.jsdom && globalThis.jsdom.window) {
  Object.defineProperty(globalThis, 'localStorage', {
    value: globalThis.jsdom.window.localStorage,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: globalThis.jsdom.window.sessionStorage,
    configurable: true,
    writable: true,
  });
}
