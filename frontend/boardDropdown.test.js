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
  const e = {
    key,
    preventDefault() { e.prevented = true; },
    stopPropagation() { e.stopped = true; },
  };
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

describe('visibleOptions — clear row handling', () => {
  it('prepends the clear row only when allowNone and no filter', () => {
    const { dd } = makeDropdown({ allowNone: true });
    expect(dd.visibleOptions().map(o => o.board_id)).toEqual([null, 'b1', 'b2', 'b3']);
    dd.filter = 'port';
    // The clear row is a state, not a search hit — hidden while filtering.
    expect(dd.visibleOptions().map(o => o.board_id)).toEqual(['b3']);
  });

  it('omits the clear row when allowNone is false', () => {
    const { dd } = makeDropdown();
    expect(dd.visibleOptions().map(o => o.board_id)).toEqual(['b1', 'b2', 'b3']);
  });
});

describe('initialHighlight — open on the current selection', () => {
  it('points at the selected item', () => {
    const { dd, host } = makeDropdown();
    host.selected = 'b3';
    expect(dd.initialHighlight()).toBe(2);
  });

  it('falls back to the first row when nothing is selected', () => {
    const { dd } = makeDropdown();
    expect(dd.initialHighlight()).toBe(0);
  });

  it('points at the clear row when allowNone and nothing is selected', () => {
    const { dd } = makeDropdown({ allowNone: true });
    expect(dd.visibleOptions()[dd.initialHighlight()].board_id).toBeNull();
  });

  it('is -1 when there is nothing to show', () => {
    const { dd } = makeDropdown({ items: () => [] });
    expect(dd.initialHighlight()).toBe(-1);
  });

  it('openMenu seeds the highlight from the selection', () => {
    const { dd, host } = makeDropdown();
    host.selected = 'b2';
    dd.openMenu();
    expect(dd.highlight).toBe(1);
  });
});

describe('moveHighlight — clamped arrow movement', () => {
  it('moves down and up by one, clamped at both ends (no wrap)', () => {
    const { dd } = makeDropdown();
    dd.highlight = 0;
    dd.moveHighlight(1);
    expect(dd.highlight).toBe(1);
    dd.moveHighlight(1);
    dd.moveHighlight(1);          // past the end
    expect(dd.highlight).toBe(2);
    dd.moveHighlight(-1);
    expect(dd.highlight).toBe(1);
    dd.moveHighlight(-5);         // past the start
    expect(dd.highlight).toBe(0);
  });

  it('starts from the top when nothing was highlighted', () => {
    const { dd } = makeDropdown();
    dd.highlight = -1;
    dd.moveHighlight(1);
    expect(dd.highlight).toBe(1);
  });

  it('does nothing when there are no options', () => {
    const { dd } = makeDropdown({ items: () => [] });
    dd.highlight = -1;
    dd.moveHighlight(1);
    expect(dd.highlight).toBe(-1);
  });
});

describe('resetHighlight — after the filter changes', () => {
  it('returns to the current selection when the filter is cleared', () => {
    const { dd, host } = makeDropdown();
    host.selected = 'b3';
    dd.filter = '';
    dd.resetHighlight();
    expect(dd.highlight).toBe(2);
  });

  it('jumps to the top match while filtering', () => {
    const { dd } = makeDropdown();
    dd.filter = 'port';
    dd.resetHighlight();
    expect(dd.highlight).toBe(0);
  });

  it('clears the highlight when nothing matches', () => {
    const { dd } = makeDropdown();
    dd.filter = 'zzz';
    dd.resetHighlight();
    expect(dd.highlight).toBe(-1);
  });
});

describe('onFilterKeydown — arrows and Enter', () => {
  it('ArrowDown/ArrowUp move the highlight and prevent default', () => {
    const { dd } = makeDropdown();
    dd.openMenu();
    const down = keyEvent('ArrowDown');
    dd.onFilterKeydown(down);
    expect(dd.highlight).toBe(1);
    expect(down.prevented).toBe(true);
    dd.onFilterKeydown(keyEvent('ArrowUp'));
    expect(dd.highlight).toBe(0);
  });

  it('Enter selects the highlighted item, not necessarily the first', () => {
    const { dd, host } = makeDropdown();
    dd.openMenu();                 // highlight 0 (nothing selected)
    dd.onFilterKeydown(keyEvent('ArrowDown'));   // → 1 (b2)
    dd.onFilterKeydown(keyEvent('Enter'));
    expect(host.selected).toBe('b2');
    expect(dd.open).toBe(false);
  });

  it('Enter does nothing when the highlight is cleared (no matches)', () => {
    const { dd, host } = makeDropdown();
    dd.openMenu();
    dd.filter = 'zzz';
    dd.resetHighlight();           // → -1
    dd.onFilterKeydown(keyEvent('Enter'));
    expect(host.selectedHistory).toEqual([]);
  });
});
