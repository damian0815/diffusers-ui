import { describe, it, expect, afterAll } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

/* Integration test: renders the REAL move-dialog dropdown block extracted
   from index.html under Alpine (jsdom), then drives it: open → filter →
   Esc twice → reopen → Enter. The production markup itself is under test —
   if the block is renamed or restructured, extraction fails loudly. */

const GALLERIES = [
  { board_id: 'b1', board_name: 'Sunsets',    image_count: 3,  latest_image: 'a.jpg' },
  { board_id: 'b2', board_name: 'city lights', image_count: 0 },
  { board_id: 'b3', board_name: 'Portraits',  image_count: 12 },
];

function readIndexHtml() {
  // vitest cwd is the repo root; import.meta.url is an http URL under jsdom
  return readFileSync(path.resolve(process.cwd(), 'frontend', 'index.html'), 'utf8');
}

// Slice out one balanced <div>…</div> subtree starting at `start`.
function sliceBlock(html, start) {
  const tag = /<div\b|<\/div>/g;
  tag.lastIndex = start;
  let m, depth = 0;
  while ((m = tag.exec(html))) {
    depth += m[0] === '</div>' ? -1 : 1;
    if (depth === 0) return html.slice(start, tag.lastIndex);
  }
  throw new Error('unbalanced <div> starting at ' + start);
}

// Every `<div … class="board-dropdown">` wrapper (the three pickers).
function dropdownBlocks(html) {
  const re = /<div\b[^>]*\bclass="board-dropdown"/g;
  const blocks = [];
  let m;
  while ((m = re.exec(html))) blocks.push(sliceBlock(html, m.index));
  return blocks;
}

function realMoveBlock() {
  const block = dropdownBlocks(readIndexHtml())
    .find(b => b.includes('moveModal.targetBoardId'));
  if (!block) throw new Error('move-dialog boardDropdown block not found in index.html');
  return block;
}

const tick = (ms = 30) => new Promise(r => setTimeout(r, ms));

/* Poll until condition true (or cap); x-show leave transitions in jsdom
   finish on an unpredictable fallback timer, so don't hardcode waits. */
async function waitFor(fn, capMs = 1500) {
  while (!fn()) {
    capMs -= 20;
    if (capMs <= 0) throw new Error('waitFor timed out');
    await tick(20);
  }
}

const menuEl   = () => document.querySelector('.board-dropdown-menu');
const labelEl  = () => document.querySelector('.board-dropdown-label');
const inputEl  = () => document.querySelector('.board-dropdown-search input');
const itemEls  = () =>
  [...document.querySelectorAll('.board-dropdown-menu .board-dropdown-item')];
const highlightedItem = () => itemEls().find(el => el.classList.contains('highlighted'));

const visible = (el) => el.style.display !== 'none';

function pressKey(el, key) {
  el.dispatchEvent(new window.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

function typeFilter(el, value) {
  el.value = value;
  el.dispatchEvent(new window.Event('input', { bubbles: true }));
}

async function mount(overrides = {}) {
  window.hostPicker = function hostPicker() {
    return Object.assign({
      galleries: [...GALLERIES],
      selectedBoardId: 'b1',           // excluded from the move-dialog list
      moveModal: { open: true, targetBoardId: null },
      onBoardChange() {},
    }, overrides);
  };

  await import('./boardDropdown.js');  // registers window.boardDropdown
  if (!window.__bdAlpineLoaded) {
    // jsdom reports every element as 0×0, and Alpine's @click.outside
    // suppresses the handler for zero-size elements. Fake real dimensions.
    for (const dim of ['offsetWidth', 'offsetHeight']) {
      Object.defineProperty(window.HTMLElement.prototype, dim, {
        configurable: true,
        get() { return 100; },
      });
    }
    // jsdom has no scrollIntoView at all; record calls so the initial-scroll
    // behaviour can be asserted.
    window.__scrollCalls = [];
    window.HTMLElement.prototype.scrollIntoView = function (opts) {
      window.__scrollCalls.push({ el: this, opts });
    };
    // Let Alpine auto-start against an empty body so it never pre-processes
    // a container we then hand to initTree (double directive registration).
    document.body.innerHTML = '';
    await import('./vendor/alpine.min.js');
    await tick(30);
    window.__bdAlpineLoaded = true;
  }
  window.__scrollCalls.length = 0;

  document.body.innerHTML = '';
  const container = document.createElement('div');
  container.setAttribute('x-data', 'hostPicker()');
  container.innerHTML = realMoveBlock();
  document.body.appendChild(container);
  await window.Alpine.initTree(container);  // public API for dynamically added trees
  await tick(30);
}

describe('index.html board dropdowns share one component', () => {
  it('all three carry the shared filter + highlight markup', () => {
    const html = readIndexHtml();
    const blocks = dropdownBlocks(html);
    expect(blocks.length).toBe(3);
    for (const b of blocks) {
      expect(b).toContain('x-data="boardDropdown({');
      expect(b).toContain('visibleOptions()');
      expect(b).toContain('highlighted: idx === highlight');
      expect(b).toContain('@input="resetHighlight()"');
      expect(b).toContain('x-ref="menu"');
      expect(b).toContain('x-ref="filterInput"');
      expect(b).toContain('@mousedown.prevent');
    }
    // No leftover per-instance open flags.
    expect(html).not.toContain('DropdownOpen');
  });
});

describe('move-dialog dropdown (real index.html markup + Alpine)', () => {
  // Alpine's document MutationObserver would otherwise fire during jsdom
  // teardown (after `document` is gone) and spew ReferenceErrors.
  afterAll(() => {
    window.Alpine?.stopObservingMutations?.();
  });

  it('opens with a fixed-position menu and autofocused filter', async () => {
    await mount();
    expect(visible(menuEl())).toBe(false);

    document.querySelector('.board-dropdown-trigger').click();
    await tick();

    expect(visible(menuEl())).toBe(true);
    // The move-dialog clipping bug: absolute menu truncated by
    // .modal-box{overflow:hidden}. Under Alpine the open menu must now
    // carry the JS-computed fixed placement.
    expect(menuEl().style.position).toBe('fixed');
    expect(menuEl().style.top).toBeTruthy();
    // b1 is the current board, so only two destinations are listed
    expect(itemEls().length).toBe(2);
  });

  it('type-to-filter narrows items; Esc clears filter then closes', async () => {
    await mount();
    document.querySelector('.board-dropdown-trigger').click();
    await tick();

    inputEl().value = 'port';
    inputEl().dispatchEvent(new window.Event('input', { bubbles: true }));
    await tick();
    expect(itemEls().length).toBe(1);
    expect(itemEls()[0].textContent).toContain('Portraits');

    pressKey(inputEl(), 'Escape');       // 1st Esc: clear filter, stay open
    await tick();
    expect(inputEl().value).toBe('');
    expect(itemEls().length).toBe(2);
    expect(visible(menuEl())).toBe(true);

    pressKey(inputEl(), 'Escape');       // 2nd Esc: close
    await waitFor(() => !visible(menuEl()));
    expect(visible(menuEl())).toBe(false);
  });

  it('Enter selects the first filtered item; click selects too', async () => {
    await mount();
    document.querySelector('.board-dropdown-trigger').click();
    await tick();

    inputEl().value = 'port';
    inputEl().dispatchEvent(new window.Event('input', { bubbles: true }));
    await tick();
    pressKey(inputEl(), 'Enter');
    await waitFor(() => !visible(menuEl()));

    expect(labelEl().textContent).toContain('Portraits (12)');
    expect(visible(menuEl())).toBe(false);

    // Click path: reopen and pick "city lights" directly.
    document.querySelector('.board-dropdown-trigger').click();
    await tick();
    itemEls().find(el => el.textContent.includes('city lights')).click();
    await waitFor(() => !visible(menuEl()));
    expect(labelEl().textContent).toContain('city lights (0)');
    expect(visible(menuEl())).toBe(false);
  });

  it('closes on click outside and resets the filter for the next open', async () => {
    await mount();
    document.querySelector('.board-dropdown-trigger').click();
    await tick();

    inputEl().value = 'zzz';             // no matches
    inputEl().dispatchEvent(new window.Event('input', { bubbles: true }));
    await tick();
    expect(itemEls().length).toBe(0);
    expect(document.querySelector('.board-dropdown-empty').style.display).not.toBe('none');

    document.body.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    await waitFor(() => !visible(menuEl()));
    expect(visible(menuEl())).toBe(false);

    document.querySelector('.board-dropdown-trigger').click();
    await tick();
    expect(inputEl().value).toBe('');    // stale filter not persisted
    expect(itemEls().length).toBe(2);
  });

  it('opens on the remembered destination and scrolls it into view', async () => {
    await mount({ moveModal: { open: true, targetBoardId: 'b3' } });
    document.querySelector('.board-dropdown-trigger').click();
    await tick();

    // Highlight must land on the current selection (b3), not the first row.
    const hi = highlightedItem();
    expect(hi).toBeTruthy();
    expect(hi.textContent).toContain('Portraits');
    // ...and that row must have been scrolled into view. The x-transition
    // defers $nextTick callbacks until it settles, so poll rather than sleep.
    await waitFor(() => window.__scrollCalls.some(c => c.el === hi));
    expect(window.__scrollCalls.some(c => c.el === hi)).toBe(true);
  });

  it('arrow keys move the highlight and Enter selects the highlighted item', async () => {
    await mount();                       // no remembered destination
    document.querySelector('.board-dropdown-trigger').click();
    await tick();

    // Items are [b2, b3]; highlight starts on the first row.
    expect(highlightedItem().textContent).toContain('city lights');

    pressKey(inputEl(), 'ArrowDown');
    await tick();
    expect(highlightedItem().textContent).toContain('Portraits');

    pressKey(inputEl(), 'ArrowUp');
    await tick();
    expect(highlightedItem().textContent).toContain('city lights');

    pressKey(inputEl(), 'ArrowDown');
    await tick();
    pressKey(inputEl(), 'Enter');
    await waitFor(() => !visible(menuEl()));
    expect(labelEl().textContent).toContain('Portraits (12)');
  });

  it('typing re-seeds the highlight onto the top match', async () => {
    // Regression guard: the filter input's x-model update must land before
    // the @input highlight reset. Start highlighted on a non-first row
    // (b3) so a stale-filter reset would leave nothing highlighted.
    await mount({ moveModal: { open: true, targetBoardId: 'b3' } });
    document.querySelector('.board-dropdown-trigger').click();
    await tick();
    expect(highlightedItem().textContent).toContain('Portraits');

    typeFilter(inputEl(), 'c');          // → only "city lights" (b2)
    await tick();
    expect(itemEls().length).toBe(1);
    expect(highlightedItem()).toBeTruthy();
    expect(highlightedItem().textContent).toContain('city lights');

    pressKey(inputEl(), 'Enter');
    await waitFor(() => !visible(menuEl()));
    expect(labelEl().textContent).toContain('city lights (0)');
  });
});
