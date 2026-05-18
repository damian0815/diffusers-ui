/* layout.js — Resizable panel drag handles + collapse/expand
 *
 * Exposed global: Layout.init()  Layout.toggleLeft()  Layout.toggleRight()
 */

const Layout = (() => {
  const COLLAPSED_W = 36;
  const MIN_W       = 36;
  const MAX_W       = 600;

  let leftW  = 280, rightW  = 300;
  let leftPW = 280, rightPW = 300;   // previous widths before collapse
  let leftCollapsed  = false;
  let rightCollapsed = false;

  const root = document.documentElement;

  function applyWidths() {
    root.style.setProperty('--left-w',  leftW  + 'px');
    root.style.setProperty('--right-w', rightW + 'px');
  }

  function syncClass(side, collapsed) {
    const id = side === 'left' ? 'controls-panel' : 'gallery-panel';
    document.getElementById(id)?.classList.toggle('panel-collapsed', collapsed);
  }

  function toggleLeft() {
    if (leftCollapsed) {
      leftW = leftPW; leftCollapsed = false;
    } else {
      leftPW = leftW; leftW = COLLAPSED_W; leftCollapsed = true;
    }
    syncClass('left', leftCollapsed);
    applyWidths();
  }

  function toggleRight() {
    if (rightCollapsed) {
      rightW = rightPW; rightCollapsed = false;
    } else {
      rightPW = rightW; rightW = COLLAPSED_W; rightCollapsed = true;
    }
    syncClass('right', rightCollapsed);
    applyWidths();
  }

  function makeHandle(id, isRight) {
    const el = document.getElementById(id);
    if (!el) return;

    el.addEventListener('mousedown', e => {
      e.preventDefault();
      const startX = e.clientX;
      const startW = isRight ? rightW : leftW;
      document.body.classList.add('is-resizing');

      const onMove = e => {
        const delta = e.clientX - startX;
        const newW  = Math.max(MIN_W, Math.min(MAX_W, startW + (isRight ? -delta : delta)));
        if (isRight) {
          rightW = newW;
          rightCollapsed = newW <= COLLAPSED_W + 8;
          syncClass('right', rightCollapsed);
        } else {
          leftW = newW;
          leftCollapsed = newW <= COLLAPSED_W + 8;
          syncClass('left', leftCollapsed);
        }
        applyWidths();
      };

      const onUp = () => {
        document.body.classList.remove('is-resizing');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      };

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  function init() {
    applyWidths();
    makeHandle('left-handle',  false);
    makeHandle('right-handle', true);
  }

  return { init, toggleLeft, toggleRight };
})();

