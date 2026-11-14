import { describe, it, expect } from 'vitest';
import './boardDropdown.js';

const GALLERIES = [
  { board_id: 'b1', board_name: 'Sunsets',    image_count: 3,  latest_image: 'a.jpg' },
  { board_id: 'b2', board_name: 'city lights', image_count: 0 },
  { board_id: 'b3', board_name: 'Portraits',  image_count: 12, latest_image_name: 'c.png' },
];

function makeDropdown(overrides = {}) {
  const host = { selected: null, selectedHistory: [] };
  const cfg = {
    items: () => GALLERIES,
    selected: () => host.selected,
    onSelect: (id) => { host.selected = id; host.selectedHistory.push(id); },
    placeholder: 'Select gallery',
    ...overrides,
  };
  return { dd: window.boardDropdown(cfg), host };
}

function keyEvent(key) {
  const e = { key, preventDefault() {}, stopPropagation() { e.stopped = true; } };
  return e;
}

describe('filteredItems', () => {
  it('returns all items when filter is empty or whitespace', () => {
    const { dd } = makeDropdown();
    expect(dd.filteredItems()).toEqual(GALLERIES);
    dd.filter = '   ';
    expect(dd.filteredItems()).toEqual(GALLERIES);
  });

  it('filters by case-insensitive name substring', () => {
    const { dd } = makeDropdown();
    dd.filter = 'CITY';
    expect(dd.filteredItems().map(g => g.board_id)).toEqual(['b2']);
    dd.filter = 's';   // all three names contain an "s"
    expect(dd.filteredItems().map(g => g.board_id)).toEqual(['b1', 'b2', 'b3']);
    dd.filter = 'sun';
    expect(dd.filteredItems().map(g => g.board_id)).toEqual(['b1']);
  });
});

describe('trigger display', () => {
  it('shows placeholder when nothing is selected or selection is missing', () => {
    const { dd } = makeDropdown();
    expect(dd.triggerLabel()).toBe('Select gallery');
    host_select(dd, 'nope');
    expect(dd.triggerLabel()).toBe('Select gallery');
  });

  it('shows "name (count)" for the selected gallery', () => {
    const { dd } = makeDropdown();
    host_select(dd, 'b1');
    expect(dd.triggerLabel()).toBe('Sunsets (3)');
    // zero counts are normalized like the legacy markup did
    host_select(dd, 'b2');
    expect(dd.triggerLabel()).toBe('city lights (0)');
  });

  it('resolves thumb url from latest_image or latest_image_name', () => {
    const { dd } = makeDropdown();
    host_select(dd, 'b1');
    expect(dd.triggerThumbSrc()).toBe('/api/v1/images/i/a.jpg/thumbnail');
    host_select(dd, 'b3');
    expect(dd.triggerThumbSrc()).toBe('/api/v1/images/i/c.png/thumbnail');
    host_select(dd, 'b2');
    expect(dd.triggerThumbSrc()).toBeNull();
  });
});

function host_select(dd, id) { dd.select(id); } // selection is just cfg state

describe('open/close/select', () => {
  it('select() routes through cfg.onSelect and closes the menu', () => {
    const { dd, host } = makeDropdown();
    dd.open = true;
    dd.select('b1');
    expect(host.selected).toBe('b1');
    expect(host.selectedHistory).toEqual(['b1']);
    expect(dd.open).toBe(false);
  });

  it('select(null) supports the allowNone clear row', () => {
    const { dd, host } = makeDropdown({ allowNone: true });
    host_select(dd, 'b1');
    dd.select(null);
    expect(host.selected).toBeNull();
    expect(dd.allowNone()).toBe(true);
  });

  it('openMenu resets a stale filter and closes via toggle', () => {
    const { dd } = makeDropdown();
    dd.filter = 'sun';
    dd.openMenu();
    expect(dd.open).toBe(true);
    expect(dd.filter).toBe('');
    dd.toggle();
    expect(dd.open).toBe(false);
  });
});

describe('onFilterKeydown — Esc semantics', () => {
  it('first Esc clears the filter, second Esc closes; both stop propagation', () => {
    const { dd } = makeDropdown();
    dd.openMenu();
    dd.filter = 'sun';

    const e1 = keyEvent('Escape');
    dd.onFilterKeydown(e1);
    expect(dd.filter).toBe('');
    expect(dd.open).toBe(true);
    expect(e1.stopped).toBe(true);

    const e2 = keyEvent('Escape');
    dd.onFilterKeydown(e2);
    expect(dd.open).toBe(false);
    expect(e2.stopped).toBe(true);
  });

  it('Enter selects the first filtered item', () => {
    const { dd, host } = makeDropdown();
    dd.openMenu();
    dd.filter = 'po';
    dd.onFilterKeydown(keyEvent('Enter'));
    expect(host.selected).toBe('b3');
    expect(dd.open).toBe(false);
  });

  it('Enter with no matches does nothing', () => {
    const { dd, host } = makeDropdown();
    dd.openMenu();
    dd.filter = 'zzz';
    dd.onFilterKeydown(keyEvent('Enter'));
    expect(host.selectedHistory).toEqual([]);
  });
});

describe('positionMenu — viewport-safe placement', () => {
  const triggerAt = (top, bottom) => ({
    $refs: { trigger: { getBoundingClientRect: () => ({ top, bottom, left: 40, width: 220 }) } },
  });

  it('opens downward when there is room below', () => {
    const { dd } = makeDropdown();
    Object.assign(dd, triggerAt(100, 130));
    dd.positionMenu();
    expect(dd.menuStyle.position).toBe('fixed');
    expect(dd.menuStyle.left).toBe('40px');
    expect(dd.menuStyle.width).toBe('220px');
    expect(dd.menuStyle.top).toBe('134px');
    expect(dd.menuStyle.maxHeight).toBe('260px');
  });

  it('flips upward near the bottom of the viewport', () => {
    const { dd } = makeDropdown();
    Object.assign(dd, triggerAt(window.innerHeight - 80, window.innerHeight - 50));
    dd.positionMenu();
    expect(dd.menuStyle.top).toBeUndefined();
    expect(dd.menuStyle.bottom).toBeDefined();
  });

  it('caps max-height to the available space below', () => {
    const { dd } = makeDropdown();
    Object.assign(dd, triggerAt(500, 530)); // 768-530-12 = 226px below → down, capped to 226
    dd.positionMenu();
    expect(dd.menuStyle.top).toBe('534px');
    expect(dd.menuStyle.maxHeight).toBe('226px');
  });
});
