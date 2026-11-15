/* frontend/boardDropdown.js — one shared gallery board-picker dropdown.

   Global Alpine data factory used as `x-data="boardDropdown({...})"`
   (parenthesized direct call per the repo convention — NOT Alpine.data()
   registration, which races alpine:init). Every custom gallery picker in
   index.html (Generate "Target Gallery", gallery header board-select,
   move dialog) instantiates this same component; sites differ only in the
   config object. Config closures are evaluated inside the surrounding
   Alpine scope, so reads/writes of host state stay reactive:

     boardDropdown({
       items:      () => [...],        // candidate galleries
       selected:   () => idOrNull,     // current selection
       onSelect:   (id) => {...},      // mutates host state (id may be null)
       placeholder: 'Select gallery',  // trigger text when nothing selected
       allowNone:   true,              // optional clear row (default false)
     })

   Shared behavior:
   - Opening highlights the current selection and scrolls it into view, so
     the menu always starts on the entry that matters (last-used target
     gallery, the open board, the remembered move destination).
   - Type-to-filter: opening the menu focuses a filter input; typing narrows
     the list by case-insensitive name substring.
   - Arrow keys move the highlight (clamped at both ends), Enter selects the
     highlighted entry, hovering an entry highlights it too.
   - Esc with a filter clears the filter; Esc with no filter closes the menu.
     Handled Escapes are stopPropagation()ed so window-level listeners
     (e.g. the Escape-closes-move-modal handler in gallery.js) don't fire.
   - The menu is position:fixed, placed from the trigger's viewport rect and
     flipped up / height-capped to stay on-screen. position:absolute menus
     were clipped by overflow-hidden/auto ancestors — most visibly the move
     dialog, where .modal-box{overflow:hidden} truncated the list to ~1 row.
*/

function boardDropdown(cfg) {
  return {
    open: false,
    filter: '',
    highlight: 0,   // index into visibleOptions(); -1 = nothing highlighted
    menuStyle: {},

    // ── Item data ───────────────────────────────────────────────────────────

    allItems() { return cfg.items() || []; },

    filteredItems() {
      const q = this.filter.trim().toLowerCase();
      const items = this.allItems();
      return q ? items.filter(g => (g.board_name || '').toLowerCase().includes(q)) : items;
    },

    noneOption() { return { board_id: null, board_name: cfg.placeholder }; },

    visibleOptions() {
      const items = this.filteredItems();
      // The clear row is a state, not a search hit: hide it while filtering so
      // that "no matching galleries" stays meaningful.
      return cfg.allowNone && !this.filter ? [this.noneOption(), ...items] : items;
    },

    selectedId() { return cfg.selected() ?? null; },

    selectedItem() {
      const id = this.selectedId();
      if (id == null) return null;
      return this.allItems().find(g => g.board_id === id) || null;
    },

    // cfg is invisible to markup — expose the two static options as methods.
    allowNone() { return !!cfg.allowNone; },
    placeholder() { return cfg.placeholder; },

    // ── Trigger display ─────────────────────────────────────────────────────

    triggerLabel() {
      const g = this.selectedItem();
      return g ? `${g.board_name} (${g.image_count || 0})` : cfg.placeholder;
    },

    thumbSrc(g) {
      const name = g && (g.latest_image || g.latest_image_name);
      return name ? `/api/v1/images/i/${name}/thumbnail` : null;
    },

    triggerThumbSrc() { return this.thumbSrc(this.selectedItem()); },

    // ── Highlight (keyboard) ────────────────────────────────────────────────

    initialHighlight() {
      const opts = this.visibleOptions();
      if (!opts.length) return -1;
      const i = opts.findIndex(o => o.board_id === this.selectedId());
      return i === -1 ? 0 : i;
    },

    highlightedOption() {
      const opts = this.visibleOptions();
      if (!opts.length || this.highlight < 0) return null;
      return opts[Math.min(this.highlight, opts.length - 1)];
    },

    moveHighlight(delta) {
      const opts = this.visibleOptions();
      if (!opts.length) return;
      const cur = this.highlight < 0 ? 0 : Math.min(this.highlight, opts.length - 1);
      this.highlight = Math.min(Math.max(cur + delta, 0), opts.length - 1);
      this.scrollHighlightIntoView();
    },

    resetHighlight() {
      this.highlight = this.filter
        ? (this.filteredItems().length ? 0 : -1)
        : this.initialHighlight();
      this.scrollHighlightIntoView();
    },

    scrollHighlightIntoView() {
      this.$nextTick?.(() => {
        const items = this.$refs?.menu?.querySelectorAll('.board-dropdown-item');
        if (!items || !items.length) return;
        const i = this.highlight < 0 ? 0 : Math.min(this.highlight, items.length - 1);
        items[i]?.scrollIntoView?.({ block: 'nearest' });
      });
    },

    // ── Open / close / select ───────────────────────────────────────────────

    toggle() { this.open ? this.closeMenu() : this.openMenu(); },

    openMenu() {
      this.open = true;
      this.filter = '';
      this.highlight = this.initialHighlight();
      this.positionMenu();
      if (!this._onViewportChange) this._onViewportChange = () => this.positionMenu();
      window.addEventListener('scroll', this._onViewportChange, true);  // capture: panel scrolls too
      window.addEventListener('resize', this._onViewportChange);
      this.$nextTick?.(() => this.$refs?.filterInput?.focus());
      this.scrollHighlightIntoView();
    },

    closeMenu() {
      this.open = false;
      if (this._onViewportChange) {
        window.removeEventListener('scroll', this._onViewportChange, true);
        window.removeEventListener('resize', this._onViewportChange);
      }
    },

    select(id) {
      cfg.onSelect(id);
      this.closeMenu();
    },

    // ── Viewport-safe positioning ───────────────────────────────────────────

    positionMenu() {
      const trigger = this.$refs?.trigger || this.$el?.querySelector('.board-dropdown-trigger');
      if (!trigger) return;
      const rect = trigger.getBoundingClientRect();
      const gap = 4, pad = 8, maxH = 260;
      const spaceBelow = window.innerHeight - rect.bottom - gap - pad;
      const spaceAbove = rect.top - gap - pad;
      const style = { position: 'fixed', left: `${rect.left}px`, width: `${rect.width}px` };
      // Prefer opening downward; flip up when below is cramped and above has room.
      if (spaceBelow >= Math.min(maxH, 160) || spaceBelow >= spaceAbove) {
        style.top = `${rect.bottom + gap}px`;
        style.maxHeight = `${Math.max(120, Math.min(maxH, spaceBelow))}px`;
      } else {
        style.bottom = `${window.innerHeight - rect.top + gap}px`;
        style.maxHeight = `${Math.max(120, Math.min(maxH, spaceAbove))}px`;
      }
      this.menuStyle = style;
    },

    // ── Keyboard: filter, highlight, select, close ──────────────────────────

    onFilterKeydown(e) {
      if (e.key === 'Escape') {
        if (this.filter) {
          this.filter = '';
          this.resetHighlight();
        } else {
          this.closeMenu();
        }
        e.preventDefault();
        e.stopPropagation();  // never let Esc fall through to window-level handlers
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const opt = this.highlightedOption();
        if (opt) this.select(opt.board_id);
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();  // keep caret moves and the gallery-grid arrow nav out of it
        this.moveHighlight(e.key === 'ArrowDown' ? 1 : -1);
      }
    },
  };
}

window.boardDropdown = boardDropdown;
