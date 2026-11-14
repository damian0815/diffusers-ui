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
   - Type-to-filter: opening the menu focuses a filter input; typing narrows
     the list by case-insensitive name substring. The filter resets on open.
   - Esc with a filter clears the filter; Esc with no filter closes the menu.
     Handled Escapes are stopPropagation()ed so window-level listeners
     (e.g. the Escape-closes-move-modal handler in gallery.js) don't fire.
   - Enter selects the first filtered item.
   - The menu is position:fixed, placed from the trigger's viewport rect and
     flipped up / height-capped to stay on-screen. position:absolute menus
     were clipped by overflow-hidden/auto ancestors — most visibly the move
     dialog, where .modal-box{overflow:hidden} truncated the list to ~1 row.
*/

function boardDropdown(cfg) {
  return {
    open: false,
    filter: '',
    menuStyle: {},

    // ── Item data ───────────────────────────────────────────────────────────

    allItems() { return cfg.items() || []; },

    filteredItems() {
      const q = this.filter.trim().toLowerCase();
      const items = this.allItems();
      return q ? items.filter(g => (g.board_name || '').toLowerCase().includes(q)) : items;
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

    // ── Open / close / select ───────────────────────────────────────────────

    toggle() { this.open ? this.closeMenu() : this.openMenu(); },

    openMenu() {
      this.open = true;
      this.filter = '';
      this.positionMenu();
      if (!this._onViewportChange) this._onViewportChange = () => this.positionMenu();
      window.addEventListener('scroll', this._onViewportChange, true);  // capture: panel scrolls too
      window.addEventListener('resize', this._onViewportChange);
      this.$nextTick?.(() => this.$refs?.filterInput?.focus());
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

    // ── Keyboard: filter clear, close, enter-to-pick ────────────────────────

    onFilterKeydown(e) {
      if (e.key === 'Escape') {
        if (this.filter) this.filter = '';
        else this.closeMenu();
        e.preventDefault();
        e.stopPropagation();  // never let Esc fall through to window-level handlers
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const first = this.filteredItems()[0];
        if (first) this.select(first.board_id);
      }
    },
  };
}

window.boardDropdown = boardDropdown;
